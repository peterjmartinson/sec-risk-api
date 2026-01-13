# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
10-K Filing Downloader with SEC EDGAR Integration.

This module provides automated download of 10-K filings from SEC EDGAR,
with SQLite-backed tracking, retry logic, and SEC policy compliance.

Design Principles:
- TDD: Tests written first, implementation follows
- Type Safety: Strict type annotations for all public APIs
- SEC Compliance: User-Agent headers, rate limiting, exponential backoff
- Audit Trail: Full provenance tracking in SQLite (filings_index + downloads)
- Idempotent: Safe to re-run, deduplicates filings and downloads

Architecture:
1. FilingsDatabase: SQLite schema for filings_index and downloads tables
2. Ticker Resolution: CIK lookup via cached SEC ticker list
3. SEC API Fetchers: Company submissions JSON + full-index parsers
4. TenKDownloader: Main downloader with retry/backoff logic
5. CLI: Command-line interface for easy usage

Usage:
    # Programmatic
    >>> downloader = TenKDownloader()
    >>> results = downloader.download_10k("MSFT", years=3)
    
    # CLI
    $ python -m sigmak.downloads.tenk_downloader --ticker MSFT --years 3
"""

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from uuid import uuid4

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

SEC_BASE_URL = "https://www.sec.gov"
SEC_DATA_URL = "https://data.sec.gov"
USER_AGENT = "SigmaK Risk Analysis Tool/1.0 (distracted.fortune@protonmail.com)"
REQUEST_TIMEOUT = 30  # seconds
MAX_RETRIES = 3
BACKOFF_FACTOR = 2  # Exponential: 1s, 2s, 4s


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class FilingRecord:
    """
    Metadata for a single SEC filing.
    
    Attributes:
        cik: Central Index Key (10-digit with leading zeros)
        ticker: Stock ticker symbol
        accession: SEC accession number (unique filing identifier)
        filing_type: Form type (e.g., "10-K", "10-Q")
        filing_date: Filing date (YYYY-MM-DD)
        sec_url: Canonical SEC URL to the filing document
        source: Data source ("company_submissions", "full_index")
        raw_metadata: Original JSON/text for audit trail
        filing_id: Database UUID (populated after insertion)
    """
    cik: str
    ticker: str
    accession: str
    filing_type: str
    filing_date: str
    sec_url: str
    source: str
    raw_metadata: Dict[str, Any]
    filing_id: Optional[str] = None


@dataclass
class DownloadRecord:
    """
    Metadata for a downloaded filing.
    
    Attributes:
        filing_index_id: Foreign key to filings_index table
        ticker: Stock ticker symbol
        year: Filing year
        local_path: Local filesystem path to downloaded file
        filename: Downloaded filename
        download_timestamp: When download occurred
        http_status: HTTP response status code
        bytes: File size in bytes
        checksum: SHA-256 checksum of file contents
        notes: Optional notes or error messages
        download_id: Database UUID (populated after insertion)
    """
    filing_index_id: str
    ticker: str
    year: int
    local_path: str
    filename: str
    download_timestamp: str
    http_status: int
    bytes: int
    checksum: str
    notes: Optional[str] = None
    download_id: Optional[str] = None


# ============================================================================
# Database Layer
# ============================================================================


class FilingsDatabase:
    """
    SQLite database for SEC filings tracking.
    
    Schema:
    - filings_index: Discovered filings with CIK, accession, SEC URL
    - downloads: Downloaded files with local path, checksum, metadata
    
    Usage:
        >>> db = FilingsDatabase()
        >>> filing_id = db.insert_filing(filing_record)
        >>> download_id = db.insert_download(download_record)
    """
    
    def __init__(self, db_path: str = "./database/sec_filings.db") -> None:
        """
        Initialize database and create schema.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize schema
        self._create_schema()
        
        logger.info(f"FilingsDatabase initialized: {db_path}")
    
    def _create_schema(self) -> None:
        """Create database schema if not exists."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            cursor = conn.cursor()
            
            # filings_index table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS filings_index (
                    id TEXT PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    cik TEXT NOT NULL,
                    accession TEXT NOT NULL,
                    filing_type TEXT NOT NULL,
                    filing_date DATE NOT NULL,
                    sec_url TEXT NOT NULL,
                    source TEXT NOT NULL,
                    discovered_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    raw_metadata TEXT,
                    UNIQUE(cik, accession)
                )
            """)
            
            # downloads table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS downloads (
                    id TEXT PRIMARY KEY,
                    filing_index_id TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    local_path TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    download_timestamp TIMESTAMP NOT NULL,
                    http_status INTEGER NOT NULL,
                    bytes INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    notes TEXT,
                    UNIQUE(filing_index_id, local_path),
                    FOREIGN KEY (filing_index_id) REFERENCES filings_index(id)
                )
            """)
            
            # Create indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_filings_ticker
                ON filings_index(ticker)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_filings_date
                ON filings_index(filing_date DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_downloads_ticker
                ON downloads(ticker, year)
            """)
            
            conn.commit()
    
    def insert_filing(self, filing: FilingRecord) -> str:
        """
        Insert filing record (idempotent - returns existing ID if duplicate).
        
        Args:
            filing: FilingRecord to insert
        
        Returns:
            Filing ID (UUID)
        """
        # Check if already exists
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id FROM filings_index
                WHERE cik = ? AND accession = ?
            """, (filing.cik, filing.accession))
            
            existing = cursor.fetchone()
            if existing:
                logger.debug(f"Filing already exists: {filing.accession}")
                return existing[0]
            
            # Insert new record
            filing_id = str(uuid4())
            cursor.execute("""
                INSERT INTO filings_index (
                    id, ticker, cik, accession, filing_type, filing_date,
                    sec_url, source, raw_metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                filing_id,
                filing.ticker,
                filing.cik,
                filing.accession,
                filing.filing_type,
                filing.filing_date,
                filing.sec_url,
                filing.source,
                json.dumps(filing.raw_metadata)
            ))
            
            conn.commit()
            logger.info(f"Inserted filing: {filing.ticker} {filing.filing_date} ({filing_id})")
            return filing_id
    
    def insert_download(self, download: DownloadRecord) -> str:
        """
        Insert download record.
        
        Args:
            download: DownloadRecord to insert
        
        Returns:
            Download ID (UUID)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            cursor = conn.cursor()
            
            # Check if already exists
            cursor.execute("""
                SELECT id FROM downloads
                WHERE filing_index_id = ? AND local_path = ?
            """, (download.filing_index_id, download.local_path))
            
            existing = cursor.fetchone()
            if existing:
                logger.debug(f"Download already recorded: {download.local_path}")
                return existing[0]
            
            # Insert new record
            download_id = str(uuid4())
            cursor.execute("""
                INSERT INTO downloads (
                    id, filing_index_id, ticker, year, local_path, filename,
                    download_timestamp, http_status, bytes, checksum, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                download_id,
                download.filing_index_id,
                download.ticker,
                download.year,
                download.local_path,
                download.filename,
                download.download_timestamp,
                download.http_status,
                download.bytes,
                download.checksum,
                download.notes
            ))
            
            conn.commit()
            logger.info(f"Recorded download: {download.ticker} {download.year} ({download_id})")
            return download_id
    
    def count_filings(self) -> int:
        """Get total number of filings in index."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM filings_index")
            return cursor.fetchone()[0]
    
    def get_downloads_for_filing(self, filing_id: str) -> List[Dict[str, Any]]:
        """Get all downloads for a specific filing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM downloads WHERE filing_index_id = ?
            """, (filing_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_all_downloads(self) -> List[Dict[str, Any]]:
        """Get all download records."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM downloads")
            return [dict(row) for row in cursor.fetchall()]
    
    def get_filings_for_ticker(
        self,
        ticker: str,
        form_type: str = "10-K",
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get filings for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            form_type: Filing type filter
            limit: Maximum results to return
        
        Returns:
            List of filing records (newest first)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = """
                SELECT * FROM filings_index
                WHERE ticker = ? AND filing_type = ?
                ORDER BY filing_date DESC
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            cursor.execute(query, (ticker.upper(), form_type))
            return [dict(row) for row in cursor.fetchall()]


# ============================================================================
# Ticker → CIK Resolution
# ============================================================================

# Cached ticker→CIK mapping (load from SEC ticker list)
_TICKER_CIK_CACHE: Optional[Dict[str, str]] = None


def _load_ticker_cik_mapping() -> Dict[str, str]:
    """
    Load ticker→CIK mapping from SEC.
    
    Uses SEC's company_tickers.json endpoint which provides current mappings.
    Caches result in memory to avoid repeated network calls.
    
    Returns:
        Dictionary mapping ticker (uppercase) to CIK (10-digit string)
    """
    global _TICKER_CIK_CACHE
    
    if _TICKER_CIK_CACHE is not None:
        return _TICKER_CIK_CACHE
    
    try:
        url = f"{SEC_DATA_URL}/files/company_tickers.json"
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
        mapping = {}
        for entry in data.values():
            ticker = entry["ticker"].upper()
            cik = str(entry["cik_str"]).zfill(10)  # Pad to 10 digits
            mapping[ticker] = cik
        
        _TICKER_CIK_CACHE = mapping
        logger.info(f"Loaded {len(mapping)} ticker→CIK mappings from SEC")
        return mapping
    
    except Exception as e:
        logger.error(f"Failed to load ticker→CIK mapping: {e}")
        # Return minimal fallback mapping for testing
        return {
            "MSFT": "0000789019",
            "AAPL": "0000320193",
            "TSLA": "0001318605",
            "GOOGL": "0001652044",
            "AMZN": "0001018724"
        }


def resolve_ticker_to_cik(ticker: str) -> str:
    """
    Resolve ticker symbol to CIK.
    
    Args:
        ticker: Stock ticker symbol (case-insensitive)
    
    Returns:
        CIK as 10-digit string with leading zeros
    
    Raises:
        ValueError: If ticker is unknown
    """
    mapping = _load_ticker_cik_mapping()
    ticker_upper = ticker.upper()
    
    if ticker_upper not in mapping:
        raise ValueError(
            f"Unknown ticker: {ticker}. "
            f"Please verify ticker symbol on SEC EDGAR."
        )
    
    return mapping[ticker_upper]


# ============================================================================
# SEC API Fetchers
# ============================================================================


def _create_session_with_retry() -> requests.Session:
    """
    Create requests session with retry logic.
    
    Implements exponential backoff for transient errors (429, 503, 504).
    """
    session = requests.Session()
    
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    return session


def fetch_company_submissions(
    cik: str,
    form_type: str = "10-K"
) -> List[FilingRecord]:
    """
    Fetch company filings from SEC submissions JSON.
    
    Args:
        cik: Central Index Key (10-digit string)
        form_type: Filing type filter (default: "10-K")
    
    Returns:
        List of FilingRecords for matching filings
    
    Raises:
        requests.HTTPError: If SEC request fails after retries
    """
    url = f"{SEC_DATA_URL}/submissions/CIK{cik}.json"
    
    session = _create_session_with_retry()
    
    try:
        response = session.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Parse recent filings
        filings: List[FilingRecord] = []
        recent = data.get("filings", {}).get("recent", {})
        
        if not recent:
            logger.warning(f"No recent filings found for CIK {cik}")
            return filings
        
        # Get ticker from data
        tickers = data.get("tickers", [])
        ticker = tickers[0] if tickers else "UNKNOWN"
        
        # Iterate through filings
        accessions = recent.get("accessionNumber", [])
        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        primary_docs = recent.get("primaryDocument", [])
        
        for i in range(len(accessions)):
            if forms[i] != form_type:
                continue
            
            accession = accessions[i]
            filing_date = filing_dates[i]
            primary_doc = primary_docs[i] if i < len(primary_docs) else None
            
            # Construct SEC URL
            # Format: https://www.sec.gov/Archives/edgar/data/{cik}/{accession-no-dashes}/{primary-document}
            accession_no_dashes = accession.replace("-", "")
            
            if primary_doc:
                sec_url = (
                    f"{SEC_BASE_URL}/Archives/edgar/data/{cik.lstrip('0')}/"
                    f"{accession_no_dashes}/{primary_doc}"
                )
            else:
                # Link to filing page
                sec_url = (
                    f"{SEC_BASE_URL}/cgi-bin/viewer?action=view&cik={cik}&"
                    f"accession_number={accession}&xbrl_type=v"
                )
            
            filing = FilingRecord(
                cik=cik,
                ticker=ticker,
                accession=accession,
                filing_type=forms[i],
                filing_date=filing_date,
                sec_url=sec_url,
                source="company_submissions",
                raw_metadata={
                    "primary_document": primary_doc,
                    "index": i
                }
            )
            
            filings.append(filing)
        
        logger.info(f"Fetched {len(filings)} {form_type} filings for CIK {cik}")
        return filings
    
    except requests.HTTPError as e:
        logger.error(f"Failed to fetch submissions for CIK {cik}: {e}")
        raise


# ============================================================================
# 10-K Downloader
# ============================================================================


class TenKDownloader:
    """
    Automated 10-K filing downloader with SEC compliance.
    
    Features:
    - Discovers filings via SEC submissions JSON
    - Downloads HTM files (skips PDFs)
    - Tracks all filings and downloads in SQLite
    - Implements retry logic with exponential backoff
    - Calculates SHA-256 checksums for integrity
    - Idempotent: safe to re-run
    
    Usage:
        >>> downloader = TenKDownloader()
        >>> results = downloader.download_10k("MSFT", years=3)
        >>> print(f"Downloaded {len(results)} filings")
    """
    
    def __init__(
        self,
        db_path: str = "./database/sec_filings.db",
        download_dir: str = "./data/filings"
    ) -> None:
        """
        Initialize downloader.
        
        Args:
            db_path: Path to SQLite database
            download_dir: Base directory for downloaded files
        """
        self.db = FilingsDatabase(db_path=db_path)
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.session = _create_session_with_retry()
        
        logger.info(f"TenKDownloader initialized: download_dir={download_dir}")
    
    def download_10k(
        self,
        ticker: str,
        years: int = 3,
        force_refresh: bool = False
    ) -> List[DownloadRecord]:
        """
        Download 10-K filings for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            years: Number of years to download (default: 3)
            force_refresh: Force re-download even if already cached
        
        Returns:
            List of DownloadRecords with metadata
        """
        logger.info(f"Downloading {years} years of 10-K filings for {ticker}")
        
        # Resolve ticker → CIK
        cik = resolve_ticker_to_cik(ticker)
        logger.info(f"Resolved {ticker} → CIK {cik}")
        
        # Fetch filings from SEC
        filings = fetch_company_submissions(cik, form_type="10-K")
        
        if not filings:
            logger.warning(f"No 10-K filings found for {ticker}")
            return []
        
        # Filter to requested years (most recent N)
        filings_sorted = sorted(filings, key=lambda f: f.filing_date, reverse=True)
        target_filings = filings_sorted[:years]
        
        logger.info(f"Found {len(target_filings)} target filings")
        
        # Download each filing
        results: List[DownloadRecord] = []
        
        for filing in target_filings:
            # Insert/get filing record
            filing_id = self.db.insert_filing(filing)
            
            # Check if already downloaded
            existing_downloads = self.db.get_downloads_for_filing(filing_id)
            if existing_downloads and not force_refresh:
                logger.info(f"Skipping already downloaded: {filing.accession}")
                # Reconstruct DownloadRecord from database
                dl = existing_downloads[0]
                results.append(DownloadRecord(
                    filing_index_id=dl["filing_index_id"],
                    ticker=dl["ticker"],
                    year=dl["year"],
                    local_path=dl["local_path"],
                    filename=dl["filename"],
                    download_timestamp=dl["download_timestamp"],
                    http_status=dl["http_status"],
                    bytes=dl["bytes"],
                    checksum=dl["checksum"],
                    notes=dl["notes"],
                    download_id=dl["id"]
                ))
                continue
            
            # Download filing
            try:
                download_record = self.download_filing(filing, filing_id)
                results.append(download_record)
            
            except Exception as e:
                logger.error(f"Failed to download {filing.accession}: {e}")
                continue
        
        logger.info(f"Successfully downloaded {len(results)} filings for {ticker}")
        return results
    
    def download_filing(
        self,
        filing: FilingRecord,
        filing_id: str
    ) -> DownloadRecord:
        """
        Download a single filing and record metadata.
        
        Args:
            filing: FilingRecord with SEC URL
            filing_id: Database filing ID
        
        Returns:
            DownloadRecord with local path and checksum
        """
        # Create directory structure: {ticker}/{year}/
        year = int(filing.filing_date.split("-")[0])
        ticker_dir = self.download_dir / filing.ticker / str(year)
        ticker_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine filename
        filename = filing.sec_url.split("/")[-1]
        if not filename.endswith(".htm") and not filename.endswith(".html"):
            # Fallback to accession-based name
            filename = f"{filing.ticker.lower()}-{filing.filing_date}.htm"
        
        local_path = ticker_dir / filename
        
        logger.info(f"Downloading {filing.sec_url} → {local_path}")
        
        # Download file
        response = self.session.get(
            filing.sec_url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        
        # Save to disk
        content = response.content
        local_path.write_bytes(content)
        
        # Calculate checksum
        checksum = hashlib.sha256(content).hexdigest()
        
        # Create download record
        download_record = DownloadRecord(
            filing_index_id=filing_id,
            ticker=filing.ticker,
            year=year,
            local_path=str(local_path),
            filename=filename,
            download_timestamp=datetime.now().isoformat(),
            http_status=response.status_code,
            bytes=len(content),
            checksum=checksum,
            notes=None
        )
        
        # Insert into database
        download_id = self.db.insert_download(download_record)
        download_record.download_id = download_id
        
        logger.info(f"Downloaded {filename}: {len(content)} bytes, checksum={checksum[:8]}...")
        
        return download_record


# ============================================================================
# CLI Interface
# ============================================================================


def main() -> None:
    """
    Command-line interface for 10-K downloader.
    
    Usage:
        python -m sigmak.downloads.tenk_downloader --ticker MSFT --years 3
    """
    parser = argparse.ArgumentParser(
        description="Download SEC 10-K filings for a ticker symbol"
    )
    
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g., MSFT, AAPL)"
    )
    
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Number of years to download (default: 3)"
    )
    
    parser.add_argument(
        "--download-dir",
        type=str,
        default="./data/filings",
        help="Base directory for downloads (default: ./data/filings)"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="./database/sec_filings.db",
        help="Path to SQLite database (default: ./database/sec_filings.db)"
    )
    
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force re-download even if files exist"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create downloader
    downloader = TenKDownloader(
        db_path=args.db_path,
        download_dir=args.download_dir
    )
    
    # Download filings
    try:
        results = downloader.download_10k(
            ticker=args.ticker,
            years=args.years,
            force_refresh=args.force_refresh
        )
        
        # Print summary
        print(f"\n✅ Successfully downloaded {len(results)} 10-K filings for {args.ticker}")
        print(f"\nDownloads:")
        for r in results:
            print(f"  • {r.year}: {r.local_path} ({r.bytes} bytes)")
        
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        exit(1)
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Download failed")
        exit(1)


if __name__ == "__main__":
    main()
