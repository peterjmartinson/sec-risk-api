from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import logging

import requests

logger = logging.getLogger(__name__)


class PeerDiscoveryService:
    """Simple peer discovery with caching of SEC data.

    - Caches `company_tickers.json` (SEC static mapping) and per-company
      submissions JSON under `cache_dir`.
    - Uses `https://www.sec.gov/files/company_tickers.json` and
      `https://data.sec.gov/submissions/CIK{cik_padded}.json` for live lookups.
    - Minimizes calls by caching results to disk.

    This implementation is intentionally small and testable. It is the
    scaffold requested in Issue #84 and will be iterated in follow-up PRs.
    """

    SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik_padded}.json"

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        user_agent: Optional[str] = None,
        db_path: Optional[str] = None,
        timeout: Optional[float] = 30.0,
    ) -> None:
        self.cache_dir = Path(cache_dir or Path("data/peer_discovery")).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # SEC requires a User-Agent header for polite use
        self.user_agent = user_agent or "sigmak-peer-discovery/1.0 (+https://example.com)"
        # HTTP request timeout (seconds)
        self.timeout = timeout or 30.0
        # path to the filings DB where we'll store peers
        from sigmak.filings_db import DEFAULT_DB_PATH

        # By default, prefer a DB colocated with the cache directory so tests
        # that pass a temporary `cache_dir` get an isolated DB. If the caller
        # explicitly provides `db_path`, use it; otherwise fall back to the
        # repo-wide default.
        if db_path:
            self.db_path = db_path
        else:
            # colocate with cache for isolation in tests and local runs
            self.db_path = str(self.cache_dir / "peers.db")

    def _headers(self) -> Dict[str, str]:
        return {"User-Agent": self.user_agent}

    def _cache_path(self, name: str) -> Path:
        return self.cache_dir / name

    def _fetch_json(self, url: str, cache_name: str, write_cache: bool = True) -> Dict:
        dst = self._cache_path(cache_name)
        if write_cache and dst.exists():
            try:
                logger.debug("Cache hit for %s", cache_name)
                return json.loads(dst.read_text(encoding="utf-8"))
            except Exception:
                logger.warning("Corrupt cache %s, removing and refetching", cache_name)
                try:
                    dst.unlink()
                except Exception:
                    logger.debug("Failed to remove cache %s", cache_name)
        logger.info("Fetching %s", url)
        try:
            resp = requests.get(url, headers=self._headers(), timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.exception("Error fetching JSON from %s: %s", url, exc)
            raise
        if write_cache:
            dst.write_text(json.dumps(data), encoding="utf-8")
        return data

    def fetch_company_tickers(self) -> List[Dict]:
        """Return the list mapping tickers to CIKs. Cached to disk."""
        data = self._fetch_json(self.SEC_TICKERS_URL, "company_tickers.json")
        # The SEC serves this mapping as an object with numeric keys; convert to list if needed
        if isinstance(data, dict):
            # older format: dict of index->entry
            try:
                return list(data.values())
            except Exception:
                return []
        return data

    def ticker_to_cik(self, ticker: str) -> Optional[str]:
        ticker = ticker.upper()
        entries = self.fetch_company_tickers()
        for e in entries:
            if e.get("ticker", "").upper() == ticker:
                cik_val = e.get("cik_str")
                return str(cik_val) if cik_val is not None else None
        return None

    def _pad_cik(self, cik: str) -> str:
        # Accept ints or strings
        return str(cik).zfill(10)

    def get_company_submissions(self, cik: str, write_cache: bool = True) -> Dict:
        """Fetch and cache the company submissions JSON for a CIK."""
        padded = self._pad_cik(cik)
        url = self.SEC_SUBMISSIONS_URL.format(cik_padded=padded)
        cache_name = f"submissions_{cik}.json"
        return self._fetch_json(url, cache_name, write_cache=write_cache)

    def get_company_sic(self, cik: str) -> Optional[str]:
        subs = self.get_company_submissions(cik)
        # SEC submissions often include a "companyInfo" block
        company_info = subs.get("companyInfo") or {}
        sic = company_info.get("sic") or subs.get("sic")
        return str(sic) if sic is not None else None

    def find_peers_for_ticker(self, ticker: str, top_n: int = 10) -> List[str]:
        """Find peer tickers that share the same SIC code.

        This is a brute-force approach that leverages cached `company_tickers.json`.
        It will fetch submissions for other companies as needed but will cache
        responses so subsequent calls are cheap.
        """
        from sigmak.filings_db import upsert_peer, get_peers_by_sic

        cik = self.ticker_to_cik(ticker)
        if not cik:
            return []
        # Ensure target info is upserted into DB
        # Prefer cached company submissions when available to avoid live requests
        target_subs = self.get_company_submissions(cik, write_cache=True)
        target_company_info = target_subs.get("companyInfo") or {}
        target_sic = target_company_info.get("sic") or target_subs.get("sic")
        target_sic = str(target_sic) if target_sic is not None else None
        # Upsert target into DB
        upsert_peer(self.db_path, ticker.upper(), cik, target_sic, target_company_info.get("title"))
        if not target_sic:
            return []
        # Query DB for existing peers with this SIC and return results.
        db_peers = get_peers_by_sic(self.db_path, target_sic, limit=top_n)
        # Exclude the target ticker itself from the returned peers.
        peers: List[str] = [p["ticker"].upper() for p in db_peers if p.get("ticker", "").upper() != ticker.upper()]
        if peers:
            return peers[:top_n]

        # No peers found in DB for this SIC. Do not perform thousands of live SEC requests here.
        logger.info("No peers found in DB for SIC %s. Falling back to an on-demand scan of company_tickers.json (cached or live).", target_sic)

        # Fallback: scan company_tickers and fetch submissions on-demand to discover peers.
        # This respects cached submissions when present but will perform requests
        # (monkeypatchable in tests) when needed. Results are upserted into the DB.
        entries = self.fetch_company_tickers()
        discovered: List[str] = []
        for e in entries:
            t = e.get("ticker")
            other_cik = e.get("cik_str")
            if not t or not other_cik:
                continue
            t_up = t.upper()
            if t_up == ticker.upper():
                continue
            try:
                subs = self.get_company_submissions(other_cik, write_cache=True)
            except Exception:
                subs = {}
            company_info = subs.get("companyInfo") or {}
            other_sic = company_info.get("sic") or subs.get("sic")
            other_sic = str(other_sic) if other_sic is not None else None
            if other_sic and other_sic == target_sic:
                upsert_peer(self.db_path, t_up, other_cik, other_sic, company_info.get("title"))
                discovered.append(t_up)
                if len(discovered) >= top_n:
                    break
        return discovered

    def refresh_peers_for_ticker(self, ticker: str, max_fetch: Optional[int] = None) -> int:
        """Scan company tickers and upsert peers for the target ticker's SIC into the DB.

        Returns the number of peer rows inserted/updated.
        """
        from sigmak.filings_db import upsert_peer

        cik = self.ticker_to_cik(ticker)
        if not cik:
            return 0

        target_subs = self.get_company_submissions(cik, write_cache=True)
        target_company_info = target_subs.get("companyInfo") or {}
        target_sic = target_company_info.get("sic") or target_subs.get("sic")
        target_sic = str(target_sic) if target_sic is not None else None
        if not target_sic:
            return 0

        entries = self.fetch_company_tickers()
        total = len(entries) if isinstance(entries, list) else 0
        count = 0
        processed = 0
        for idx, e in enumerate(entries, start=1):
            if max_fetch is not None and processed >= max_fetch:
                break
            processed += 1
            if idx % 200 == 0:
                logger.info("Refreshing peers: scanned %d/%d tickers, matched %d so far", idx, total, count)
            t = e.get("ticker")
            other_cik = e.get("cik_str")
            if not t or not other_cik:
                continue
            t_up = t.upper()
            if t_up == ticker.upper():
                continue
            try:
                subs = self.get_company_submissions(other_cik, write_cache=False)
            except Exception:
                subs = {}
            company_info = subs.get("companyInfo") or {}
            other_sic = company_info.get("sic") or subs.get("sic")
            other_sic = str(other_sic) if other_sic is not None else None
            if other_sic and other_sic == target_sic:
                upsert_peer(self.db_path, t_up, other_cik, other_sic, company_info.get("title"))
                count += 1
        return count
