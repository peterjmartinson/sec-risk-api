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
import time


DEFAULT_DB_PATH = os.environ.get("SIGMAK_FILINGS_DB", "./database/sec_filings.db")
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

        # Detect whether the filings_index table contains a filing_year column.
        cursor.execute("PRAGMA table_info('filings_index')")
        cols = [r[1] for r in cursor.fetchall()]
        has_filing_year = "filing_year" in cols

        if has_filing_year:
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
        else:
            # Fallback for older/newer schema which stores filing_date but not filing_year.
            # Use the year portion of filing_date (ISO YYYY-MM-DD) to match the requested year.
            cursor.execute(
                """
                SELECT accession, cik, sec_url, filing_date
                FROM filings_index
                WHERE ticker = ? AND substr(filing_date,1,4) = ?
                ORDER BY
                    CASE WHEN filing_date IS NULL THEN 0 ELSE 1 END DESC,
                    filing_date DESC
                LIMIT 1
                """,
                (ticker.upper(), str(filing_year)),
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


def ensure_peers_table(db_path: str = DEFAULT_DB_PATH) -> None:
    """Create the peers table used for peer discovery indexing."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        # If peers table doesn't exist, create with NOT NULL last_updated defaulting to now
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='peers'")
        if not cursor.fetchone():
            cursor.execute(
                """
                CREATE TABLE peers (
                    ticker TEXT PRIMARY KEY,
                    cik TEXT,
                    sic TEXT,
                    industry TEXT,
                    market_cap REAL,
                    last_updated INTEGER NOT NULL DEFAULT (strftime('%s','now'))
                )
                """
            )
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_peers_sic ON peers(sic)")
            conn.commit()
            return

        # If table exists, ensure last_updated is NOT NULL. If it's nullable or missing,
        # perform a migration to a new table with the NOT NULL constraint and copy data.
        cursor.execute("PRAGMA table_info('peers')")
        cols = cursor.fetchall()
        col_names = [c[1] for c in cols]
        last_updated_info = None
        for c in cols:
            if c[1] == "last_updated":
                last_updated_info = c
                break

        needs_migration = False
        if last_updated_info is None:
            needs_migration = True
        else:
            # PRAGMA table_info returns notnull flag at index 3
            notnull_flag = last_updated_info[3]
            if notnull_flag == 0:
                needs_migration = True

        if not needs_migration:
            # table exists and last_updated is already NOT NULL
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_peers_sic ON peers(sic)")
            conn.commit()
            return

        # perform migration: create new table, copy rows, and replace
        cursor.execute(
            """
            CREATE TABLE peers_new (
                ticker TEXT PRIMARY KEY,
                cik TEXT,
                sic TEXT,
                industry TEXT,
                market_cap REAL,
                last_updated INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            )
            """
        )

        # copy existing data, setting last_updated to existing value or now()
        cursor.execute(
            """
            INSERT INTO peers_new(ticker,cik,sic,industry,market_cap,last_updated)
            SELECT
                ticker,cik,sic,industry,market_cap, COALESCE(last_updated, strftime('%s','now'))
            FROM peers
            """
        )

        cursor.execute("DROP TABLE peers")
        cursor.execute("ALTER TABLE peers_new RENAME TO peers")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_peers_sic ON peers(sic)")
        conn.commit()


def upsert_peer(
    db_path: str,
    ticker: str,
    cik: Optional[str],
    sic: Optional[str],
    industry: Optional[str] = None,
    market_cap: Optional[float] = None,
    last_updated: Optional[int] = None,
) -> None:
    """Insert or update a peer row."""
    ensure_peers_table(db_path)
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        ts = int(time.time())
        if last_updated is None:
            last_updated = ts

        cursor.execute(
            """
            INSERT INTO peers(ticker,cik,sic,industry,market_cap,last_updated)
            VALUES(?,?,?,?,?,?)
            ON CONFLICT(ticker) DO UPDATE SET
                cik=excluded.cik,
                sic=excluded.sic,
                industry=excluded.industry,
                market_cap=excluded.market_cap,
                last_updated=excluded.last_updated
            """,
            (ticker.upper(), cik, sic, industry, market_cap, last_updated),
        )
        conn.commit()


def get_peers_by_sic(db_path: str, sic: str, limit: Optional[int] = None) -> List[Dict[str, Optional[str]]]:
    """Return list of peer rows matching SIC ordered by market_cap desc (NULLs last)."""
    if not Path(db_path).exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query = "SELECT ticker,cik,sic,industry,market_cap FROM peers WHERE sic = ? ORDER BY market_cap DESC NULLS LAST"
        params = [sic]
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [ {"ticker": r["ticker"], "cik": r["cik"], "sic": r["sic"], "industry": r["industry"], "market_cap": r["market_cap"]} for r in rows ]
