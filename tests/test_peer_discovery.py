import json
from pathlib import Path

import requests

from sigmak.peer_discovery import PeerDiscoveryService


class DummyResponse:
    def __init__(self, data):
        self._data = data
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


def test_peer_discovery_caches_and_finds_peers(tmp_path, monkeypatch):
    # Prepare fake company_tickers mapping with two companies sharing the same SIC
    tickers = [
        {"cik_str": "0000320193", "ticker": "AAPL", "title": "Apple Inc."},
        {"cik_str": "0000789019", "ticker": "MSFT", "title": "Microsoft Corp."},
    ]

    submissions_map = {
        "0000320193": {"companyInfo": {"sic": "3571", "title": "Apple Inc."}},
        "0000789019": {"companyInfo": {"sic": "3571", "title": "Microsoft Corp."}},
    }

    def fake_get(url, headers=None, timeout=None):
        if url.endswith("company_tickers.json"):
            return DummyResponse(tickers)
        if "submissions" in url:
            for cik, data in submissions_map.items():
                if cik in url:
                    return DummyResponse(data)
        raise AssertionError(f"Unexpected URL in test: {url}")

    monkeypatch.setattr(requests, "get", fake_get)

    cache_dir = tmp_path / "peer_cache"
    service = PeerDiscoveryService(cache_dir=cache_dir, user_agent="test-agent")

    # ticker -> CIK
    cik = service.ticker_to_cik("AAPL")
    assert cik == "0000320193"

    # SIC extraction and caching of submissions
    sic = service.get_company_sic(cik)
    assert sic == "3571"

    # The cache files should have been written
    assert (cache_dir / "company_tickers.json").exists()
    assert (cache_dir / "submissions_0000320193.json").exists()

    # Finding peers: MSFT should be returned since it has the same SIC in our fake data
    peers = service.find_peers_for_ticker("AAPL")
    assert "MSFT" in peers
