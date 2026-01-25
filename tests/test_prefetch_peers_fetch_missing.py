import json

from pathlib import Path

from sigmak.prefetch_peers import prefetch_from_cache
from sigmak.filings_db import get_peer


class DummySvc:
    def __init__(self, entries, fetched):
        self._entries = entries
        self._fetched = fetched

    def fetch_company_tickers(self):
        return self._entries

    def get_company_submissions(self, cik, write_cache=True):
        return self._fetched[str(cik)]


def test_prefetch_reads_existing_and_fetches_missing(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    # existing cached submission
    sample_existing = {
        "cik": "0000000123",
        "tickers": ["TST"],
        "companyInfo": {"companyName": "Test Corp", "sic": "9999", "title": "Test Corp"},
    }
    fp_existing = cache / "submissions_0000000123.json"
    fp_existing.write_text(json.dumps(sample_existing), encoding="utf-8")

    # simulate fetched submission for another company (no file present)
    fetched = {
        "0000000456": {
            "cik": "0000000456",
            "tickers": ["MSS"],
            "companyInfo": {"companyName": "Missing Co", "sic": "9999", "title": "Missing Co"},
        }
    }

    # company tickers list contains both companies
    entries = [
        {"ticker": "TST", "cik_str": "0000000123"},
        {"ticker": "MSS", "cik_str": "0000000456"},
    ]

    # inject dummy PeerDiscoveryService into module
    import sigmak.prefetch_peers as pp

    pp._svc_for_test = DummySvc(entries, fetched)

    db_file = tmp_path / "test_db.sqlite"
    n = prefetch_from_cache(str(cache), str(db_file), fetch_missing=True)
    assert n == 2

    peer1 = get_peer(str(db_file), "TST")
    peer2 = get_peer(str(db_file), "MSS")
    assert peer1 is not None and peer1["ticker"] == "TST"
    assert peer2 is not None and peer2["ticker"] == "MSS"
