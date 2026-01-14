"""
Unit tests for filings_db and YoY report integration.

Tests verify:
1. Deterministic selection of latest filing when duplicates exist
2. Missing identifier fallback and CSV logging
3. Report generation with database-sourced identifiers
"""
import os
import tempfile
from pathlib import Path

import pytest

from sigmak.filings_db import insert_filing, get_latest_filing, get_identifiers, MISSING_TOKEN


def test_get_latest_filing_selects_most_recent_date(tmp_path):
    db_file = tmp_path / "filings.db"
    # Insert two filings for same ticker/year with different dates
    insert_filing(str(db_file), "TEST", 2023, "ACC-OLD", "000111", "/old", "2023-01-01")
    insert_filing(str(db_file), "TEST", 2023, "ACC-NEW", "000111", "/new", "2023-10-01")

    latest = get_latest_filing(str(db_file), "TEST", 2023)
    assert latest is not None
    assert latest.accession == "ACC-NEW"
    assert latest.sec_url == "/new"


def test_missing_identifiers_return_fallback_and_log(tmp_path):
    """Test that missing DB rows use fallback token and log to CSV."""
    db_file = tmp_path / "empty.db"
    # Do not insert any rows
    missing_log = tmp_path / "missing.csv"

    ids = get_identifiers(str(db_file), "NOFIL", 2022, missing_log_path=str(missing_log))
    assert ids["accession"] == MISSING_TOKEN
    assert ids["cik"] == MISSING_TOKEN
    assert ids["sec_url"] == MISSING_TOKEN

    # Ensure the missing log was created and contains a row for NOFIL
    content = missing_log.read_text()
    assert "NOFIL" in content
    assert "2022" in content


def test_load_filing_provenance_uses_db_first(tmp_path):
    """Test that load_filing_provenance retrieves from SQLite before falling back to JSON."""
    import sys
    import importlib.util
    
    # Dynamically load the report module to avoid heavy imports
    script_path = Path(__file__).parent.parent / "scripts" / "generate_yoy_report.py"
    spec = importlib.util.spec_from_file_location("report_module", script_path)
    if not spec or not spec.loader:
        pytest.skip("Cannot load generate_yoy_report.py for testing")
    
    report_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = report_module
    spec.loader.exec_module(report_module)
    
    db_file = tmp_path / "filings.db"
    
    # Insert a filing row
    insert_filing(str(db_file), "FOO", 2025, "ACC-12345", "000999", 
                  "/Archives/edgar/data/000999/ACC-12345.htm", "2025-02-15")
    
    # Call load_filing_provenance with the test DB path
    prov = report_module.load_filing_provenance("FOO", 2025, filings_db_path=str(db_file))
    
    assert prov["accession"] == "ACC-12345"
    assert prov["cik"] == "000999"
    assert "/Archives/edgar/data/000999/ACC-12345.htm" in prov["sec_url"]

