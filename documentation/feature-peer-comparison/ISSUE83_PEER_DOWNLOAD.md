Automate Peer 10-K Filing Download Pipeline #83

Objective
Add a script that, given a target ticker/year (e.g., "NVDA", 2024) and a list of peer tickers (see peer-discovery service), fetches the latest 10-K HTML filings for the target and all peers into data/filings/{ticker}/{year} using the existing downloader logic.

Key Requirements
Provide a CLI utility (scripts/download_peers_and_target.py) that:
Accepts a target ticker and year (and optionally, a list of peer tickers; if not supplied, uses output from Issue #peer-discovery-sec-sic).
Downloads the most recent 10-K HTML filing for each company (target + peers) if not already present, using the existing TenKDownloader logic and database for tracking.
Handles missing filings, years where a company isn't public, and logs clear warnings (but continues processing others).
Prints a summary table of what was downloaded for each ticker (or why it was skipped).
Ensure idempotency: does NOT re-download existing files unless --force-refresh is specified.
Add end-to-end and unit tests: mock download logic, verify correct filesystem organization and DB updates.
Acceptance Criteria
Stand-alone script callable from command-line with minimal args for interactive use
Handles 404/missing filings gracefully (clear output, not halting the batch)
Output logs/files make it easy to troubleshoot missing or problematic filings