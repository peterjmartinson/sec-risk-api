from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from sigmak.filings_db import upsert_peer, ensure_peers_table

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# Test injection point: tests can set _svc_for_test to a dummy PeerDiscoveryService
_svc_for_test = None

try:
    from sigmak.peer_discovery import PeerDiscoveryService  # type: ignore
except Exception:
    PeerDiscoveryService = None  # pragma: no cover


def _extract_tickers(subs: Dict) -> Optional[str]:
    t = subs.get("tickers")
    if isinstance(t, list) and len(t) > 0:
        return t[0].upper()
    t2 = subs.get("ticker") or subs.get("tradingSymbol")
    return str(t2).upper() if t2 else None


def _parse_recent_filings(subs: Dict) -> Dict:
    recent = subs.get("filings", {}).get("recent", {})
    result = {"recent_filing_count": None, "latest_filing_date": None, "latest_10k_date": None}
    if not recent:
        return result
    dates = recent.get("filingDate") or recent.get("filingDates") or []
    forms = recent.get("form") or []
    if dates:
        try:
            latest = max(dates)
            result["latest_filing_date"] = latest
        except Exception:
            pass
    if forms and dates:
        latest_10k = None
        for form, dt in zip(forms, dates):
            if str(form).upper().startswith("10") and ("K" in str(form).upper() or str(form).upper().startswith("10-K")):
                latest_10k = dt
        if latest_10k:
            result["latest_10k_date"] = latest_10k
    result["recent_filing_count"] = len(dates) if dates else None
    return result


def _process_submission_dict(data: Dict, fp_stat_mtime: Optional[int], db_path: str) -> None:
    ticker = _extract_tickers(data)
    cik = data.get("cik") or data.get("cik_str")
    company_info = data.get("companyInfo") or {}
    name = company_info.get("companyName") or company_info.get("title") or data.get("entityName") or data.get("companyName")
    sic = company_info.get("sic") or data.get("sic")
    sic_desc = company_info.get("sicDescription") or data.get("sicDescription")
    state = company_info.get("stateOfIncorporation") or data.get("stateOfIncorporation")
    fy = company_info.get("fiscalYearEnd") or data.get("fiscalYearEnd")
    website = None
    phone = None
    bus = company_info.get("businessAddress") or data.get("businessAddress") or {}
    if isinstance(bus, dict):
        website = bus.get("website") or bus.get("webSite")
        phone = bus.get("phone") or data.get("phone")

    filings_info = _parse_recent_filings(data)

    if not ticker:
        return

    upsert_peer(
        db_path,
        ticker.upper(),
        str(cik) if cik is not None else None,
        str(sic) if sic is not None else None,
        industry=company_info.get("title") or data.get("industry"),
        market_cap=None,
        last_updated=int(datetime.utcnow().timestamp()),
        name=name,
        sic_description=sic_desc,
        state_of_incorporation=state,
        fiscal_year_end=fy,
        website=website,
        phone=phone,
        owner_org=None,
        entity_type=None,
        category=None,
        recent_filing_count=filings_info.get("recent_filing_count"),
        latest_filing_date=filings_info.get("latest_filing_date"),
        latest_10k_date=filings_info.get("latest_10k_date"),
        submissions_cached_at=fp_stat_mtime,
    )


def prefetch_from_cache(cache_dir: str, db_path: str, limit: Optional[int] = None, *, fetch_missing: bool = False, max_fetch: Optional[int] = None) -> int:
    """
    Scan `cache_dir` for submissions_*.json files and upsert peers into DB.

    If `fetch_missing` is True, the function will call the SEC for companies that
    appear in the `company_tickers` listing but do not have a submissions_{cik}.json
    present in `cache_dir`. Fetched submissions are written to `cache_dir` before upsert.

    Returns number of peer rows processed (existing files + newly fetched).
    """
    ensure_peers_table(db_path)
    p = Path(cache_dir)
    if not p.exists():
        raise FileNotFoundError(f"Cache dir not found: {cache_dir}")

    processed = 0
    processed_existing = 0
    processed_fetched = 0
    logger.info(
        "Prefetch start: cache_dir=%s db=%s fetch_missing=%s limit=%s max_fetch=%s",
        cache_dir,
        db_path,
        fetch_missing,
        limit,
        max_fetch,
    )
    # First, process existing submissions files found on disk
    files = sorted([x for x in p.glob("submissions_*.json")])
    for idx, fp in enumerate(files, start=1):
        if limit is not None and processed >= limit:
            break
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Skipping unreadable/corrupt JSON: %s", fp)
            continue
        try:
            mtime = int(fp.stat().st_mtime)
        except Exception:
            mtime = None
        _process_submission_dict(data, mtime, db_path)
        processed += 1
        processed_existing += 1
        if processed_existing % 50 == 0:
            logger.info("Processed %d existing submission files", processed_existing)

    # If allowed, fetch missing submissions for companies listed in company_tickers
    if fetch_missing:
        svc = _svc_for_test or (PeerDiscoveryService() if PeerDiscoveryService is not None else None)
        if svc is None:
            raise RuntimeError("PeerDiscoveryService not available to fetch missing submissions")
        entries = svc.fetch_company_tickers()
        total = len(entries) if isinstance(entries, list) else 0
        fetched_count = 0
        logger.info("Fetching missing submissions from SEC (total tickers=%s)", total)
        for idx, e in enumerate(entries, start=1):
            if max_fetch is not None and fetched_count >= max_fetch:
                break
            cik = e.get("cik_str") or e.get("cik")
            if not cik:
                continue
            cache_fp = p / f"submissions_{cik}.json"
            if cache_fp.exists():
                continue
            logger.debug("Fetching submissions for CIK %s (%d/%d)", cik, idx, total)
            try:
                subs = svc.get_company_submissions(cik, write_cache=True)
            except Exception:
                logger.warning("Failed to fetch submissions for CIK %s, skipping", cik)
                continue
            # write_cache=True in PeerDiscoveryService already writes file; but to be safe:
            try:
                cache_fp.write_text(json.dumps(subs), encoding="utf-8")
            except Exception:
                logger.debug("Could not write cache file %s (may have been written by service)", cache_fp)
            mtime = None
            try:
                mtime = int(cache_fp.stat().st_mtime)
            except Exception:
                mtime = None
            _process_submission_dict(subs, mtime, db_path)
            processed += 1
            fetched_count += 1
            processed_fetched += 1
            if fetched_count % 20 == 0:
                logger.info("Fetched %d missing submissions so far", fetched_count)
            if max_fetch is not None and fetched_count >= max_fetch:
                break
            if idx % 200 == 0:
                logger.info("Prefetch: scanned %d/%d company tickers, fetched %d so far", idx, total, fetched_count)

    logger.info("Prefetch complete: processed=%d existing=%d fetched=%d", processed, processed_existing, processed_fetched)
    return processed


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/peer_discovery")
    ap.add_argument("--db", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fetch-missing", action="store_true", help="Fetch missing submissions from SEC and write to cache")
    ap.add_argument("--max-fetch", type=int, default=None, help="Maximum number of missing submissions to fetch")
    args = ap.parse_args()
    from sigmak.filings_db import DEFAULT_DB_PATH

    dbp = args.db or DEFAULT_DB_PATH
    n = prefetch_from_cache(args.cache_dir, dbp, limit=args.limit, fetch_missing=args.fetch_missing, max_fetch=args.max_fetch)
    print(f"Prefetched {n} submission files into peers table")
