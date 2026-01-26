import importlib.util
from types import SimpleNamespace
from pathlib import Path

import pytest

from sigmak.downloads.tenk_downloader import FilingRecord


def load_module():
    spec = importlib.util.spec_from_file_location(
        "download_peers_and_target", Path("scripts/download_peers_and_target.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_select_peers_strict_sic_chooses_by_marketcap_and_availability(monkeypatch):
    mod = load_module()

    # Mock target row: market_cap 100
    target = {
        "ticker": "TGT",
        "sic": "9999",
        "market_cap": 100,
        "latest_10k_date": "2024-03-01",
    }

    # Candidates with varying market caps and availability
    candidates = [
        {"ticker": "A", "market_cap": 90, "latest_10k_date": "2024-02-01", "latest_10k_date": "2024-02-01", "latest_10k_date": "2024-02-01", "latest_10k_date": "2024-02-01", "latest_10k_date": "2024-02-01", "latest_10k_date": "2024-02-01"},
        {"ticker": "B", "market_cap": 150, "latest_10k_date": "2023-12-01"},
        {"ticker": "C", "market_cap": 102, "latest_10k_date": "2024-01-01"},
        {"ticker": "D", "market_cap": 50, "latest_10k_date": "2022-06-01"},
    ]

    # get_peer and get_peers_by_sic are imported symbols in the module; patch them
    monkeypatch.setattr(mod, "get_peer", lambda db, t: target)
    monkeypatch.setattr(mod, "get_peers_by_sic", lambda db, s, limit=500: candidates)

    # Create a dummy PeerDiscoveryService (not used here but required arg)
    svc = SimpleNamespace()

    selected = mod.select_peers_strict_sic(svc, db_path="/tmp/fake.db", target_ticker="TGT", year=2024, max_peers=2, require_filing_year=False)

    # Expect the two closest market caps to 100: C (102) and A (90)
    assert selected == ["C", "A"]


def test_download_for_ticker_downloads_and_skips(monkeypatch, tmp_path):
    mod = load_module()

    # Create a fake FilingRecord for the year
    f = FilingRecord(
        cik="0000000001",
        ticker="TST",
        accession="0001-01-000001",
        filing_type="10-K",
        filing_date="2024-02-02",
        sec_url="https://example.com/doc1.htm",
        source="company_submissions",
        raw_metadata={}
    )

    # Monkeypatch resolver and submissions fetcher
    monkeypatch.setattr(mod, "resolve_ticker_to_cik", lambda t: "0000000001")
    monkeypatch.setattr(mod, "fetch_company_submissions", lambda cik, form_type="10-K": [f])

    # Dummy downloader that records calls
    calls = {}

    class DummyDB:
        def insert_filing(self, filing):
            return "filing-id-1"

        def get_downloads_for_filing(self, filing_id):
            return []

    class DummyDownloader:
        def __init__(self):
            self.db = DummyDB()

        def download_filing(self, filing, filing_id):
            calls[filing.ticker] = filing.filing_date
            return True

    downloader = DummyDownloader()

    ticker, status = mod.download_for_ticker(downloader, "TST", 2024, force_refresh=False)
    assert ticker == "TST"
    assert status in ("downloaded", "fallback_recent")
    assert calls.get("TST") == "2024-02-02"
