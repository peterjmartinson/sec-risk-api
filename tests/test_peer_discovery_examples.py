import json

from pathlib import Path

from sigmak.prefetch_peers import prefetch_from_cache
from sigmak.peer_discovery import PeerDiscoveryService


def test_peer_discovery_examples(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()

    # company_tickers.json: list format
    tickers = [
        {"ticker": "NVDA", "cik_str": "0000000123"},
        {"ticker": "AMD", "cik_str": "0000000456"},
        {"ticker": "INTC", "cik_str": "0000000789"},
    ]
    (cache / "company_tickers.json").write_text(json.dumps(tickers), encoding="utf-8")

    # submissions JSONs all share same SIC so they are peers
    base = {
        "companyInfo": {
            "companyName": "Example Co",
            "title": "Example Co",
            "sic": "9999",
            "sicDescription": "Example Industry",
        },
        "tickers": ["TST"],
    }

    s1 = dict(base)
    s1.update({"cik": "0000000123", "tickers": ["NVDA"]})
    s2 = dict(base)
    s2.update({"cik": "0000000456", "tickers": ["AMD"]})
    s3 = dict(base)
    s3.update({"cik": "0000000789", "tickers": ["INTC"]})

    (cache / "submissions_0000000123.json").write_text(json.dumps(s1), encoding="utf-8")
    (cache / "submissions_0000000456.json").write_text(json.dumps(s2), encoding="utf-8")
    (cache / "submissions_0000000789.json").write_text(json.dumps(s3), encoding="utf-8")

    db_file = tmp_path / "peers.db"

    # populate DB from cache
    n = prefetch_from_cache(str(cache), str(db_file))
    assert n == 3

    svc = PeerDiscoveryService(cache_dir=cache, db_path=str(db_file))

    peers = svc.find_peers_for_ticker("NVDA", top_n=10)
    # NVDA should see AMD and INTC as peers (order not important)
    assert "AMD" in peers and "INTC" in peers

    # Acceptance: expected peers list
    expected = {"AMD", "INTC"}
    assert set(peers) >= expected
