"""Small, clean SQLite interface for filing provenance (filings_index).

Provides simple helpers used by the YoY report generator to fetch
accession, CIK, and SEC URL for a given ticker and filing year.

Design goals:
- Minimal dependency surface (stdlib sqlite3)
- Deterministic selection: when multiple filings exist for a ticker/year,
  return the row with the latest `filing_date`.
- Return clear fallback token when identifiers are missing and optionally
  record the missing identifiers to a reconciliation CSV.
"""
from __future__ import annotations

import csv
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional


DEFAULT_DB_PATH = os.environ.get("SIGMAK_FILINGS_DB", "./database/filings_index.db")
MISSING_TOKEN = "MISSING_IDENTIFIERS"


@dataclass
class FilingIdentifiers:
    accession: str
    cik: str
    sec_url: str
    filing_date: Optional[str] = None


def _ensure_db(db_path: str = DEFAULT_DB_PATH) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS filings_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                filing_year INTEGER NOT NULL,
                accession TEXT,
                cik TEXT,
                sec_url TEXT,
                filing_date TEXT
            )
            """
        )
        conn.commit()


def insert_filing(
    db_path: str,
    ticker: str,
    filing_year: int,
    accession: Optional[str],
    cik: Optional[str],
    sec_url: Optional[str],
    filing_date: Optional[str] = None,
) -> None:
    """Insert a filing row (used by downloader/tests)."""
    _ensure_db(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO filings_index (ticker, filing_year, accession, cik, sec_url, filing_date)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (ticker.upper(), filing_year, accession, cik, sec_url, filing_date),
        )
        conn.commit()


def get_latest_filing(db_path: str, ticker: str, filing_year: int) -> Optional[FilingIdentifiers]:
    """Return the filing identifiers for the latest filing_date for ticker/year.

    If multiple rows exist, the row with the latest ISO-8601 filing_date is chosen.
    Returns None if no row exists.
    """
    if not Path(db_path).exists():
        return None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT accession, cik, sec_url, filing_date
            FROM filings_index
            WHERE ticker = ? AND filing_year = ?
            ORDER BY
                CASE WHEN filing_date IS NULL THEN 0 ELSE 1 END DESC,
                filing_date DESC
            LIMIT 1
            """,
            (ticker.upper(), filing_year),
        )
        row = cursor.fetchone()
        if not row:
            return None

        return FilingIdentifiers(
            accession=row["accession"] if row["accession"] is not None else "",
            cik=row["cik"] if row["cik"] is not None else "",
            sec_url=row["sec_url"] if row["sec_url"] is not None else "",
            filing_date=row["filing_date"],
        )


def get_identifiers(
    db_path: Optional[str],
    ticker: str,
    filing_year: int,
    missing_log_path: Optional[str] = "output/missing_identifiers.csv",
) -> Dict[str, str]:
    """Get accession, cik, sec_url for a filing.

    Policy:
    - If a DB row exists with non-empty accession/cik/sec_url, return those values.
    - If any identifier is missing or DB has no row, return the fallback token
      `MISSING_IDENTIFIERS` for the missing fields and record an audit row in
      `missing_log_path` if provided.
    """
    db_path = db_path or DEFAULT_DB_PATH
    ident = get_latest_filing(db_path, ticker, filing_year)

    result = {
        "accession": MISSING_TOKEN,
        "cik": MISSING_TOKEN,
        "sec_url": MISSING_TOKEN,
    }

    missing_reasons = []

    if ident:
        if ident.accession:
            result["accession"] = ident.accession
        else:
            missing_reasons.append("accession")

        if ident.cik:
            result["cik"] = ident.cik
        else:
            missing_reasons.append("cik")

        if ident.sec_url:
            result["sec_url"] = ident.sec_url
        else:
            missing_reasons.append("sec_url")
    else:
        missing_reasons.extend(["accession", "cik", "sec_url"])

    if missing_reasons and missing_log_path:
        Path(missing_log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(missing_log_path, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            # If file is new, write header
            if csvfile.tell() == 0:
                writer.writerow(["timestamp", "ticker", "filing_year", "missing_fields"]) 
            writer.writerow([datetime.utcnow().isoformat(), ticker.upper(), filing_year, ";".join(missing_reasons)])

    return result
