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
from typing import List, Any


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
                    name TEXT,
                    sic TEXT,
                    sic_description TEXT,
                    industry TEXT,
                    state_of_incorporation TEXT,
                    fiscal_year_end TEXT,
                    website TEXT,
                    phone TEXT,
                    owner_org TEXT,
                    entity_type TEXT,
                    category TEXT,
                    market_cap REAL,
                    recent_filing_count INTEGER,
                    latest_filing_date TEXT,
                    latest_10k_date TEXT,
                    submissions_cached_at INTEGER,
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

        # Determine if migration is needed. Migration is required when:
        # - any of the expected columns are missing from the existing table, or
        # - the `last_updated` column is missing or nullable.
        existing_cols = set(col_names)
        expected_cols = {
            "ticker",
            "cik",
            "name",
            "sic",
            "sic_description",
            "industry",
            "state_of_incorporation",
            "fiscal_year_end",
            "website",
            "phone",
            "owner_org",
            "entity_type",
            "category",
            "market_cap",
            "recent_filing_count",
            "latest_filing_date",
            "latest_10k_date",
            "submissions_cached_at",
            "last_updated",
        }

        needs_migration = False
        if not expected_cols.issubset(existing_cols):
            needs_migration = True
        else:
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
                name TEXT,
                sic TEXT,
                sic_description TEXT,
                industry TEXT,
                state_of_incorporation TEXT,
                fiscal_year_end TEXT,
                website TEXT,
                phone TEXT,
                owner_org TEXT,
                entity_type TEXT,
                category TEXT,
                market_cap REAL,
                recent_filing_count INTEGER,
                latest_filing_date TEXT,
                latest_10k_date TEXT,
                submissions_cached_at INTEGER,
                last_updated INTEGER NOT NULL DEFAULT (strftime('%s','now'))
            )
            """
        )

        # Build a SELECT that maps existing columns into the new schema. For any
        # expected column not present in the old table, select NULL as that column.
        select_items = []
        for col in [
            "ticker",
            "cik",
            "name",
            "sic",
            "sic_description",
            "industry",
            "state_of_incorporation",
            "fiscal_year_end",
            "website",
            "phone",
            "owner_org",
            "entity_type",
            "category",
            "market_cap",
            "recent_filing_count",
            "latest_filing_date",
            "latest_10k_date",
            "submissions_cached_at",
        ]:
            if col in existing_cols:
                select_items.append(col)
            else:
                select_items.append(f"NULL AS {col}")

        # Handle last_updated specially: if it exists, coalesce to now(), else use now()
        if "last_updated" in existing_cols:
            last_updated_expr = "COALESCE(last_updated, strftime('%s','now'))"
        else:
            last_updated_expr = "strftime('%s','now')"

        select_sql = ",".join(select_items) + "," + last_updated_expr

        insert_sql = f"INSERT INTO peers_new(\n                ticker,cik,name,sic,sic_description,industry,state_of_incorporation,\n                fiscal_year_end,website,phone,owner_org,entity_type,category,market_cap,\n                recent_filing_count,latest_filing_date,latest_10k_date,submissions_cached_at,last_updated)\n            SELECT {select_sql} FROM peers"

        cursor.execute(insert_sql)

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
    name: Optional[str] = None,
    sic_description: Optional[str] = None,
    state_of_incorporation: Optional[str] = None,
    fiscal_year_end: Optional[str] = None,
    website: Optional[str] = None,
    phone: Optional[str] = None,
    owner_org: Optional[str] = None,
    entity_type: Optional[str] = None,
    category: Optional[str] = None,
    recent_filing_count: Optional[int] = None,
    latest_filing_date: Optional[str] = None,
    latest_10k_date: Optional[str] = None,
    submissions_cached_at: Optional[int] = None,
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
            INSERT INTO peers(
                ticker,cik,name,sic,sic_description,industry,state_of_incorporation,
                fiscal_year_end,website,phone,owner_org,entity_type,category,market_cap,
                recent_filing_count,latest_filing_date,latest_10k_date,submissions_cached_at,last_updated)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(ticker) DO UPDATE SET
                cik=COALESCE(excluded.cik, cik),
                name=COALESCE(excluded.name, name),
                sic=COALESCE(excluded.sic, sic),
                sic_description=COALESCE(excluded.sic_description, sic_description),
                industry=COALESCE(excluded.industry, industry),
                state_of_incorporation=COALESCE(excluded.state_of_incorporation, state_of_incorporation),
                fiscal_year_end=COALESCE(excluded.fiscal_year_end, fiscal_year_end),
                website=COALESCE(excluded.website, website),
                phone=COALESCE(excluded.phone, phone),
                owner_org=COALESCE(excluded.owner_org, owner_org),
                entity_type=COALESCE(excluded.entity_type, entity_type),
                category=COALESCE(excluded.category, category),
                market_cap=COALESCE(excluded.market_cap, market_cap),
                recent_filing_count=COALESCE(excluded.recent_filing_count, recent_filing_count),
                latest_filing_date=COALESCE(excluded.latest_filing_date, latest_filing_date),
                latest_10k_date=COALESCE(excluded.latest_10k_date, latest_10k_date),
                submissions_cached_at=COALESCE(excluded.submissions_cached_at, submissions_cached_at),
                last_updated=excluded.last_updated
            """,
            (
                ticker.upper(),
                cik,
                name,
                sic,
                sic_description,
                industry,
                state_of_incorporation,
                fiscal_year_end,
                website,
                phone,
                owner_org,
                entity_type,
                category,
                market_cap,
                recent_filing_count,
                latest_filing_date,
                latest_10k_date,
                submissions_cached_at,
                last_updated,
            ),
        )
        conn.commit()


def get_peers_by_sic(db_path: str, sic: str, limit: Optional[int] = None) -> List[Dict[str, Optional[str]]]:
    """Return list of peer rows matching SIC ordered by market_cap desc (NULLs last)."""
    if not Path(db_path).exists():
        return []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query = "SELECT ticker,cik,name,sic,sic_description,industry,state_of_incorporation,fiscal_year_end,website,phone,market_cap,recent_filing_count,latest_filing_date,latest_10k_date,submissions_cached_at FROM peers WHERE sic = ? ORDER BY market_cap DESC NULLS LAST"
        params = [sic]
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        out = []
        for r in rows:
            out.append(
                {
                    "ticker": r["ticker"],
                    "cik": r["cik"],
                    "name": r["name"],
                    "sic": r["sic"],
                    "sic_description": r["sic_description"],
                    "industry": r["industry"],
                    "state_of_incorporation": r["state_of_incorporation"],
                    "fiscal_year_end": r["fiscal_year_end"],
                    "website": r["website"],
                    "phone": r["phone"],
                    "market_cap": r["market_cap"],
                    "recent_filing_count": r["recent_filing_count"],
                    "latest_filing_date": r["latest_filing_date"],
                    "latest_10k_date": r["latest_10k_date"],
                    "submissions_cached_at": r["submissions_cached_at"],
                }
            )
        return out


def get_peer(db_path: str, ticker: str) -> Optional[Dict[str, Any]]:
    """Return a single peer row by ticker or None."""
    if not Path(db_path).exists():
        return None
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT ticker,cik,name,sic,sic_description,industry,state_of_incorporation,fiscal_year_end,website,phone,market_cap,recent_filing_count,latest_filing_date,latest_10k_date,submissions_cached_at,last_updated FROM peers WHERE ticker = ?", (ticker.upper(),))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "ticker": row["ticker"],
            "cik": row["cik"],
            "name": row["name"],
            "sic": row["sic"],
            "sic_description": row["sic_description"],
            "industry": row["industry"],
            "state_of_incorporation": row["state_of_incorporation"],
            "fiscal_year_end": row["fiscal_year_end"],
            "website": row["website"],
            "phone": row["phone"],
            "market_cap": row["market_cap"],
            "recent_filing_count": row["recent_filing_count"],
            "latest_filing_date": row["latest_filing_date"],
            "latest_10k_date": row["latest_10k_date"],
            "submissions_cached_at": row["submissions_cached_at"],
            "last_updated": row["last_updated"],
        }


def get_all_peer_tickers(db_path: str) -> List[str]:
    """Return all tickers present in peers table."""
    if not Path(db_path).exists():
        return []
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("SELECT ticker FROM peers")
        return [r[0] for r in cur.fetchall()]


def populate_market_cap(db_path: str, tickers: Optional[List[str]] = None, delay: float = 1.0) -> int:
    """Populate `market_cap` for peers using `yfinance`.

    - If `tickers` is None, populate for all peers in the DB.
    - Returns number of rows updated.
    - Requires `yfinance` to be installed; raises informative ImportError otherwise.
    """
    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("yfinance is required to populate market cap. Install with `pip install yfinance`.") from exc

    if tickers is None:
        tickers = get_all_peer_tickers(db_path)

    updated = 0
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            info = tk.info
            mcap = info.get("marketCap")
        except Exception:
            mcap = None

        peer = get_peer(db_path, t)
        if not peer:
            continue
        # Only update if we found a numeric market cap
        if mcap is not None:
            upsert_peer(db_path, t, peer.get("cik"), peer.get("sic"), peer.get("industry"), float(mcap), int(time.time()))
            updated += 1
        time.sleep(delay)

    return updated
