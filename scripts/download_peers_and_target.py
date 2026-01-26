#!/usr/bin/env python3
"""Download 10-Ks for a target ticker and its industry peers.

This script selects up to N peers using a strict 4-digit SIC match (see
documentation/feature-peer-comparison/ISSUE83_PEER_DOWNLOAD.md) and then
uses the existing TenKDownloader to fetch 10-K HTMLs into the repo's
`data/filings/{ticker}/{year}` layout.

Usage examples:
    PYTHONPATH=src python scripts/download_peers_and_target.py NVDA 2024
    PYTHONPATH=src python scripts/download_peers_and_target.py NVDA 2024 --max-peers 6 --require-filing-year

This file leverages `src/sigmak/downloads/tenk_downloader.py` and the
`PeerDiscoveryService` + `peers` DB for candidate selection.
"""
from __future__ import annotations

import argparse
import logging
from typing import List, Optional, Tuple

from sigmak.downloads.tenk_downloader import (
    TenKDownloader,
    resolve_ticker_to_cik,
    fetch_company_submissions,
)
from sigmak.peer_discovery import PeerDiscoveryService
from sigmak.filings_db import get_peers_by_sic, get_peer


logger = logging.getLogger(__name__)


def select_peers_strict_sic(
    svc: PeerDiscoveryService,
    db_path: str,
    target_ticker: str,
    year: int,
    max_peers: int = 6,
    require_filing_year: bool = True,
) -> List[str]:
    """Return up to `max_peers` tickers that share the exact 4-digit SIC.

    Tie-breakers implemented (in order):
    1. Filing availability for `year` (if required)
    2. Market-cap proximity to the target (absolute difference)
    3. Recent filing date
    """
    # Ensure target info is in DB (upsert happens inside find_peers_for_ticker)
    try:
        svc.find_peers_for_ticker(target_ticker, top_n=0)
    except Exception:
        # best-effort: continue even if on-demand scan fails
        logger.debug("Non-fatal: on-demand upsert failed for %s", target_ticker)

    target_row = get_peer(db_path, target_ticker)
    if not target_row:
        raise ValueError(f"Target ticker not present in peers DB: {target_ticker}")

    sic = target_row.get("sic")
    if not sic:
        raise ValueError(f"Target has no SIC: {target_ticker}")

    candidates = get_peers_by_sic(db_path, sic, limit=500)
    # Exclude the target
    candidates = [c for c in candidates if c["ticker"].upper() != target_ticker.upper()]

    # Annotate availability for requested year
    for c in candidates:
        latest_10k = c.get("latest_10k_date")
        c["has_year"] = False
        if latest_10k and isinstance(latest_10k, str) and str(year) in latest_10k:
            c["has_year"] = True

    # Filter by filing-year requirement
    if require_filing_year:
        primary = [c for c in candidates if c.get("has_year")]
        fallback = [c for c in candidates if not c.get("has_year")]
    else:
        primary = candidates
        fallback = []

    selected: List[Tuple[str, dict]] = []

    def market_cap_key(c: dict) -> float:
        try:
            targ = float(target_row.get("market_cap") or 0)
            val = float(c.get("market_cap") or 0)
            return abs(targ - val)
        except Exception:
            return float("inf")

    # Sort primary by market-cap proximity then recent filing date
    primary_sorted = sorted(primary, key=lambda c: (market_cap_key(c), -(int(c.get("latest_10k_date", "0")[:4]) if c.get("latest_10k_date") else 0)))

    for c in primary_sorted:
        selected.append((c["ticker"].upper(), c))
        if len(selected) >= max_peers:
            break

    # If still underfilled, consider fallback candidates
    if len(selected) < max_peers and fallback:
        fallback_sorted = sorted(fallback, key=lambda c: (market_cap_key(c), -(int(c.get("latest_10k_date", "0")[:4]) if c.get("latest_10k_date") else 0)))
        for c in fallback_sorted:
            selected.append((c["ticker"].upper(), c))
            if len(selected) >= max_peers:
                break

    return [t for t, _ in selected]


def download_for_ticker(
    downloader: TenKDownloader,
    ticker: str,
    year: int,
    force_refresh: bool = False,
) -> Tuple[str, str]:
    """Attempt to download the 10-K for `ticker` for `year`.

    Returns (ticker, status) where status is one of: downloaded, skipped, missing, error
    """
    try:
        cik = resolve_ticker_to_cik(ticker)
    except ValueError as e:
        logger.warning("Could not resolve %s: %s", ticker, e)
        return ticker, "error:unknown_ticker"

    try:
        filings = fetch_company_submissions(cik, form_type="10-K")
    except Exception as e:
        logger.warning("Failed to fetch submissions for %s: %s", ticker, e)
        return ticker, "error:submissions"

    # Find filing matching year
    chosen = None
    for f in filings:
        if str(year) == f.filing_date.split("-")[0]:
            chosen = f
            break

    if not chosen:
        # fallback: pick the most recent 10-K
        if filings:
            chosen = sorted(filings, key=lambda x: x.filing_date, reverse=True)[0]
            return_status = "fallback_recent"
        else:
            return ticker, "missing"
    else:
        return_status = "downloaded"

    try:
        # Insert filing record and download
        filing_id = downloader.db.insert_filing(chosen)
        # Check existing downloads
        existing = downloader.db.get_downloads_for_filing(filing_id)
        if existing and not force_refresh:
            logger.info("Skipping existing download for %s %s", ticker, chosen.filing_date)
            return ticker, "skipped"

        dl = downloader.download_filing(chosen, filing_id)
        return ticker, return_status

    except Exception as e:
        logger.exception("Download failed for %s", ticker)
        return ticker, f"error:{str(e)}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download 10-K for a target and its peers")
    parser.add_argument("ticker", help="Target ticker (e.g., NVDA)")
    parser.add_argument("--year", type=int, default=None, help="Filing year (e.g., 2024). If omitted, downloads the latest available 10-K for each company.")
    parser.add_argument("--peers", nargs="+", help="Explicit peer tickers (optional)")
    parser.add_argument("--max-peers", type=int, default=6, help="Max peers to download (default: 6)")
    parser.add_argument("--require-filing-year", action="store_true", help="Require peer to have a filing for the given year")
    parser.add_argument("--force-refresh", action="store_true", help="Force re-download even if present")
    parser.add_argument("--db-path", type=str, default="./database/sec_filings.db", help="Path to filings DB")
    parser.add_argument("--download-dir", type=str, default="./data/filings", help="Base download dir")
    parser.add_argument("--cache-dir", type=str, default="./data/peer_discovery", help="Peer discovery cache dir")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    downloader = TenKDownloader(db_path=args.db_path, download_dir=args.download_dir)
    svc = PeerDiscoveryService(cache_dir=args.cache_dir, db_path=args.db_path)

    target = args.ticker.upper()
    year = args.year  # may be None -> use latest available per-company

    if args.peers:
        peers = [p.upper() for p in args.peers]
    else:
        peers = select_peers_strict_sic(
            svc,
            args.db_path,
            target,
            year,
            max_peers=args.max_peers,
            require_filing_year=args.require_filing_year,
        )

    # Always include target first
    to_process = [target] + [p for p in peers if p != target]

    summary = {}
    for t in to_process:
        ticker, status = download_for_ticker(downloader, t, year, force_refresh=args.force_refresh)
        summary[ticker] = status

    # Print summary
    print("\nDownload Summary:")
    for t, s in summary.items():
        print(f" - {t}: {s}")


if __name__ == "__main__":
    main()
