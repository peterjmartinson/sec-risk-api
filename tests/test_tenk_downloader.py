# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Unit tests for 10-K downloader system.

Tests follow TDD principles: written before implementation.
Each test asserts exactly one behavior (Single Responsibility Principle).
"""

import json
import sqlite3
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests

from sigmak.downloads.tenk_downloader import (
    TenKDownloader,
    resolve_ticker_to_cik,
    fetch_company_submissions,
    FilingsDatabase,
    DownloadRecord,
    FilingRecord,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_db() -> Path:
    """Create a temporary SQLite database."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = Path(f.name)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def filings_db(temp_db: Path) -> FilingsDatabase:
    """Create a FilingsDatabase instance with temp database."""
    return FilingsDatabase(db_path=str(temp_db))


@pytest.fixture
def temp_download_dir() -> Path:
    """Create a temporary download directory."""
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_company_submissions() -> Dict[str, Any]:
    """Mock SEC company submissions JSON response."""
    return {
        "cik": "0000789019",
        "entityType": "operating",
        "name": "MICROSOFT CORP",
        "tickers": ["MSFT"],
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0000950170-24-028362",
                    "0000950170-23-032074",
                    "0000950170-22-015598"
                ],
                "filingDate": [
                    "2024-07-30",
                    "2023-07-27",
                    "2022-07-28"
                ],
                "form": [
                    "10-K",
                    "10-K",
                    "10-K"
                ],
                "primaryDocument": [
                    "msft-20240630.htm",
                    "msft-20230630.htm",
                    "msft-20220630.htm"
                ],
                "primaryDocDescription": [
                    "10-K",
                    "10-K",
                    "10-K"
                ]
            }
        }
    }


@pytest.fixture
def mock_htm_content() -> bytes:
    """Mock HTM filing content."""
    return b"""<!DOCTYPE html>
<html>
<head><title>10-K Filing</title></head>
<body>
<h1>Item 1A. Risk Factors</h1>
<p>Operational risks may affect our business.</p>
</body>
</html>"""


# ============================================================================
# Test 1: Ticker → CIK Resolution
# ============================================================================


class TestTickerResolution:
    """Test ticker to CIK resolution."""
    
    def test_resolve_ticker_to_cik_returns_expected_cik(self) -> None:
        """
        Given a known ticker symbol,
        When resolve_ticker_to_cik is called,
        Then it returns the correct CIK string with padding.
        """
        # Known mapping: MSFT -> CIK 0000789019
        cik = resolve_ticker_to_cik("MSFT")
        
        assert cik == "0000789019"
        assert isinstance(cik, str)
        assert len(cik) == 10  # SEC CIK is 10 digits with leading zeros
    
    def test_resolve_ticker_case_insensitive(self) -> None:
        """
        Given a lowercase ticker,
        When resolve_ticker_to_cik is called,
        Then it resolves correctly (case-insensitive).
        """
        cik_upper = resolve_ticker_to_cik("AAPL")
        cik_lower = resolve_ticker_to_cik("aapl")
        
        assert cik_upper == cik_lower
        assert cik_upper == "0000320193"
    
    def test_resolve_unknown_ticker_raises_error(self) -> None:
        """
        Given an unknown ticker,
        When resolve_ticker_to_cik is called,
        Then it raises ValueError with clear message.
        """
        with pytest.raises(ValueError, match="Unknown ticker.*INVALID"):
            resolve_ticker_to_cik("INVALID123")


# ============================================================================
# Test 2: Company Submissions Fetching
# ============================================================================


class TestCompanySubmissionsFetcher:
    """Test SEC company submissions JSON fetcher."""
    
    def test_fetch_company_submissions_parses_accession_and_sec_url(
        self,
        mock_company_submissions: Dict[str, Any]
    ) -> None:
        """
        Given a mocked SEC submissions JSON response,
        When fetch_company_submissions is called,
        Then it parses and returns FilingRecords with accession, filing_type, filing_date, and sec_url.
        """
        # Setup mock response
        with patch('sigmak.downloads.tenk_downloader._create_session_with_retry') as mock_session_factory:
            mock_session = Mock()
            mock_session_factory.return_value = mock_session
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_company_submissions
            mock_session.get.return_value = mock_response
            
            # Fetch submissions
            filings = fetch_company_submissions("0000789019")
            
            # Assertions
            assert len(filings) >= 3
            
            # Check first 10-K filing
            first_filing = filings[0]
            assert isinstance(first_filing, FilingRecord)
            assert first_filing.accession.startswith("0000950170")
            assert first_filing.filing_type == "10-K"
            assert first_filing.filing_date  # Has a filing date
            assert first_filing.cik == "0000789019"
            assert ".htm" in first_filing.sec_url
            
            # Verify session.get was called
            assert mock_session.get.called
    
    @patch('requests.get')
    def test_fetch_company_submissions_filters_by_form_type(
        self,
        mock_get: Mock,
        mock_company_submissions: Dict[str, Any]
    ) -> None:
        """
        Given submissions with multiple form types,
        When fetch_company_submissions is called with form_type filter,
        Then only matching forms are returned.
        """
        # Setup mock
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_company_submissions
        mock_get.return_value = mock_response
        
        # Fetch only 10-K filings
        filings = fetch_company_submissions("0000789019", form_type="10-K")
        
        # All returned filings should be 10-K
        assert all(f.filing_type == "10-K" for f in filings)


# ============================================================================
# Test 3: Database Schema and Constraints
# ============================================================================


class TestFilingsDatabase:
    """Test SQLite database schema and operations."""
    
    def test_filing_index_inserts_and_enforces_unique(
        self,
        filings_db: FilingsDatabase
    ) -> None:
        """
        Given a filing record,
        When inserted twice with same CIK and accession,
        Then second insert does not create duplicate (UNIQUE constraint enforced).
        """
        filing = FilingRecord(
            cik="0000789019",
            ticker="MSFT",
            accession="0000950170-24-028362",
            filing_type="10-K",
            filing_date="2024-07-30",
            sec_url="https://www.sec.gov/Archives/edgar/data/789019/000095017024028362/msft-20240630.htm",
            source="company_submissions",
            raw_metadata={"test": "data"}
        )
        
        # First insert should succeed
        filing_id_1 = filings_db.insert_filing(filing)
        assert filing_id_1 is not None
        
        # Second insert with same CIK+accession should return existing ID
        filing_id_2 = filings_db.insert_filing(filing)
        assert filing_id_1 == filing_id_2
        
        # Verify only one record exists
        count = filings_db.count_filings()
        assert count == 1
    
    def test_database_creates_required_tables(
        self,
        filings_db: FilingsDatabase
    ) -> None:
        """
        Given a new database,
        When FilingsDatabase is initialized,
        Then filings_index and downloads tables are created.
        """
        with sqlite3.connect(filings_db.db_path) as conn:
            cursor = conn.cursor()
            
            # Check filings_index table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='filings_index'
            """)
            assert cursor.fetchone() is not None
            
            # Check downloads table exists
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='downloads'
            """)
            assert cursor.fetchone() is not None
    
    def test_database_enforces_foreign_key_constraint(
        self,
        filings_db: FilingsDatabase
    ) -> None:
        """
        Given downloads table with foreign key to filings_index,
        When attempting to insert download with invalid filing_index_id,
        Then insertion fails (foreign key constraint).
        """
        # Try to insert download with non-existent filing_index_id
        with pytest.raises(sqlite3.IntegrityError):
            with sqlite3.connect(filings_db.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = ON")
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO downloads (
                        id, filing_index_id, ticker, year, local_path, 
                        filename, download_timestamp, http_status, bytes, checksum
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    "test-id",
                    "invalid-filing-id",  # Does not exist
                    "MSFT",
                    2024,
                    "/tmp/test.htm",
                    "test.htm",
                    datetime.now().isoformat(),
                    200,
                    1024,
                    "abc123"
                ))
                conn.commit()


# ============================================================================
# Test 4: File Download and Recording
# ============================================================================


class TestFileDownload:
    """Test HTM file download and metadata recording."""
    
    def test_download_10k_downloads_file_and_records_in_downloads_table(
        self,
        filings_db: FilingsDatabase,
        temp_download_dir: Path,
        mock_htm_content: bytes
    ) -> None:
        """
        Given a mocked HTTP response with HTM content,
        When download_10k downloads a filing,
        Then file is saved to disk with correct checksum and downloads row is created.
        """
        # Create downloader
        downloader = TenKDownloader(
            db_path=filings_db.db_path,
            download_dir=str(temp_download_dir)
        )
        
        # Insert a filing record first
        filing = FilingRecord(
            cik="0000789019",
            ticker="MSFT",
            accession="0000950170-24-028362",
            filing_type="10-K",
            filing_date="2024-07-30",
            sec_url="https://www.sec.gov/test/msft.htm",  # Mock URL
            source="company_submissions",
            raw_metadata={}
        )
        filing_id = filings_db.insert_filing(filing)
        
        # Mock the session's get method
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = mock_htm_content
        mock_response.headers = {"Content-Length": str(len(mock_htm_content))}
        mock_response.raise_for_status.return_value = None
        
        downloader.session.get = Mock(return_value=mock_response)
        
        # Download the filing
        download_record = downloader.download_filing(filing, filing_id)
        
        # Verify file was saved
        assert download_record.local_path
        local_file = Path(download_record.local_path)
        assert local_file.exists()
        assert local_file.read_bytes() == mock_htm_content
        
        # Verify download record in database
        assert download_record.http_status == 200
        assert download_record.bytes == len(mock_htm_content)
        assert download_record.checksum  # SHA-256 checksum present
        assert len(download_record.checksum) == 64  # SHA-256 is 64 hex chars
        
        # Verify in downloads table
        downloads = filings_db.get_downloads_for_filing(filing_id)
        assert len(downloads) == 1
        assert downloads[0]["checksum"] == download_record.checksum


# ============================================================================
# Test 5: CLI Interface
# ============================================================================


class TestCLI:
    """Test command-line interface."""
    
    @patch('sigmak.downloads.tenk_downloader.TenKDownloader.download_10k')
    def test_cli_default_years_is_three(
        self,
        mock_download: Mock
    ) -> None:
        """
        Given CLI invoked with only ticker argument,
        When download_10k CLI is executed,
        Then default years parameter is 3.
        """
        from sigmak.downloads.tenk_downloader import main
        
        # Mock sys.argv
        with patch('sys.argv', ['tenk_downloader.py', '--ticker', 'MSFT']):
            try:
                main()
            except SystemExit:
                pass  # CLI may exit after completion
        
        # Verify download_10k was called with years=3
        mock_download.assert_called()
        call_kwargs = mock_download.call_args[1] if mock_download.call_args[1] else {}
        
        # Either positional arg or kwarg
        if len(mock_download.call_args[0]) >= 2:
            assert mock_download.call_args[0][1] == 3  # years as second positional
        else:
            assert call_kwargs.get('years', 3) == 3


# ============================================================================
# Test 6: Rate Limiting and Retry Logic
# ============================================================================


class TestRetryLogic:
    """Test rate limiting and retry on transient errors."""
    
    def test_rate_limiting_and_retry_on_transient_errors(
        self,
        mock_company_submissions: Dict[str, Any]
    ) -> None:
        """
        Given SEC API integration,
        When fetch_company_submissions is called,
        Then it uses a session with retry logic configured (urllib3 handles retries).
        """
        # Verify that session is created with retry logic
        from sigmak.downloads.tenk_downloader import _create_session_with_retry
        
        # Create a real session to verify retry configuration
        session = _create_session_with_retry()
        
        # Verify the session has retry adapters mounted
        assert session.adapters is not None
        
        # Check that http:// and https:// adapters have retry configured
        http_adapter = session.get_adapter("http://")
        https_adapter = session.get_adapter("https://")
        
        # urllib3 HTTPAdapter with Retry will have max_retries attribute
        assert hasattr(http_adapter, 'max_retries')
        assert hasattr(https_adapter, 'max_retries')
        
        # Verify max_retries is configured (should be a Retry object)
        from urllib3.util.retry import Retry
        assert isinstance(http_adapter.max_retries, Retry)
        assert isinstance(https_adapter.max_retries, Retry)
        
        # Verify retry settings
        retry_config = http_adapter.max_retries
        assert retry_config.total == 3  # MAX_RETRIES constant
        assert 429 in retry_config.status_forcelist  # Rate limiting
        assert 503 in retry_config.status_forcelist  # Service unavailable
    
    def test_retry_exhaustion_raises_error(self) -> None:
        """
        Given SEC API consistently failing,
        When retries are exhausted,
        Then fetch_company_submissions raises exception.
        """
        with patch('sigmak.downloads.tenk_downloader._create_session_with_retry') as mock_session_factory:
            mock_session = Mock()
            mock_session_factory.return_value = mock_session
            
            # All calls fail
            def fail(*args, **kwargs):
                raise requests.HTTPError("503 Server Error")
            
            mock_session.get.side_effect = fail
            
            # Should raise after max retries
            with pytest.raises(requests.HTTPError):
                fetch_company_submissions("0000789019")


# ============================================================================
# Integration-Style Tests
# ============================================================================


class TestTenKDownloaderIntegration:
    """Integration tests for complete download workflow."""
    
    def test_download_10k_end_to_end(
        self,
        filings_db: FilingsDatabase,
        temp_download_dir: Path,
        mock_htm_content: bytes
    ) -> None:
        """
        Given a ticker and years,
        When download_10k is called,
        Then it discovers filings, downloads HTM files, and records everything.
        """
        with patch('sigmak.downloads.tenk_downloader.fetch_company_submissions') as mock_fetch:
            # Mock filing discovery
            mock_fetch.return_value = [
                FilingRecord(
                    cik="0000789019",
                    ticker="MSFT",
                    accession="0000950170-24-028362",
                    filing_type="10-K",
                    filing_date="2024-07-30",
                    sec_url="https://www.sec.gov/test/msft.htm",
                    source="company_submissions",
                    raw_metadata={}
                )
            ]
            
            # Create downloader
            downloader = TenKDownloader(
                db_path=filings_db.db_path,
                download_dir=str(temp_download_dir)
            )
            
            # Mock the session's get method
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.content = mock_htm_content
            mock_response.headers = {"Content-Length": str(len(mock_htm_content))}
            mock_response.raise_for_status.return_value = None
            
            downloader.session.get = Mock(return_value=mock_response)
            
            # Execute download
            results = downloader.download_10k("MSFT", years=1)
            
            # Verify results
            assert len(results) == 1
            assert results[0].ticker == "MSFT"
            assert results[0].http_status == 200
            assert Path(results[0].local_path).exists()
            
            # Verify database records
            assert filings_db.count_filings() == 1
            downloads = filings_db.get_all_downloads()
            assert len(downloads) == 1
