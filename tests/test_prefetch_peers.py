import json
import tempfile
from pathlib import Path

from sigmak.prefetch_peers import prefetch_from_cache
from sigmak.filings_db import get_peer


def test_prefetch_upserts_peer_from_sample_cache(tmp_path):
    # prepare a fake cache directory with one submissions file
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    sample = {
        "cik": "0000000123",
        "tickers": ["TST"],
        "companyInfo": {
            "companyName": "Test Corp",
            "title": "Test Corp",
            "sic": "9999",
            "sicDescription": "Testing Services",
            "stateOfIncorporation": "DE",
            "fiscalYearEnd": "1231",
            "businessAddress": {"phone": "555-0100", "website": "https://example.com"}
        },
        "filings": {
            "recent": {"form": ["10-K", "8-K"], "filingDate": ["2025-01-01", "2025-06-01"]}
        }
    }
    fp = cache_dir / "submissions_0000000123.json"
    fp.write_text(json.dumps(sample), encoding="utf-8")

    # use a temporary DB file
    db_file = tmp_path / "test_db.sqlite"

    n = prefetch_from_cache(str(cache_dir), str(db_file))
    assert n == 1

    peer = get_peer(str(db_file), "TST")
    assert peer is not None
    assert peer["ticker"] == "TST"
    assert peer["cik"] == "0000000123"
    assert peer["sic"] == "9999"
    # recent filing date populated
    assert peer["market_cap"] is None or isinstance(peer["market_cap"], (int, float))
