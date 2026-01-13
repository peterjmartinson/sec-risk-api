"""Downloads subpackage for sigmak.

This makes `src/sigmak/downloads` an importable package so modules like
`sigmak.downloads.tenk_downloader` are discoverable and can be committed.
"""

__all__ = []
# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
SEC Filing Downloads Package.

Provides automated download of 10-K filings with SQLite-backed tracking.
"""

from sigmak.downloads.tenk_downloader import (
    TenKDownloader,
    FilingsDatabase,
    FilingRecord,
    DownloadRecord,
    resolve_ticker_to_cik,
    fetch_company_submissions,
)

__all__ = [
    "TenKDownloader",
    "FilingsDatabase",
    "FilingRecord",
    "DownloadRecord",
    "resolve_ticker_to_cik",
    "fetch_company_submissions",
]
