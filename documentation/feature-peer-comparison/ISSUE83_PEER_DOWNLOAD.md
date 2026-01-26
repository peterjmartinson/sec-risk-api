Automate Peer 10-K Filing Download Pipeline #83

Objective
Add a script that, given a target ticker/year (e.g., "NVDA", 2024) and a list of peer tickers (see peer-discovery service), fetches the latest 10-K HTML filings for the target and all peers into `data/filings/{ticker}/{year}` using the existing downloader logic.

Key Requirements
- Identify a subset of peers for comparison with the target company.
- Provide a CLI utility (`scripts/download_peers_and_target.py`) that:
	- Accepts a target ticker and year (and optionally, a list of peer tickers; if not supplied, uses output from Issue #peer-discovery-sec-sic).
	- Downloads the most recent 10-K HTML filing for each company (target + peers) if not already present, using the existing `TenKDownloader` logic and database for tracking.
	- Handles missing filings, years where a company isn't public, and logs clear warnings (but continues processing others).
	- Prints a summary table of what was downloaded for each ticker (or why it was skipped).
	- Ensures idempotency: does NOT re-download existing files unless `--force-refresh` is specified.
	- Adds end-to-end and unit tests: mock download logic, verify correct filesystem organization and DB updates.

Acceptance Criteria
- Stand-alone script callable from command-line with minimal args for interactive use
- Handles 404/missing filings gracefully (clear output, not halting the batch)
- Output logs/files make it easy to troubleshoot missing or problematic filings

Selection policy (strict SIC match)

- Primary criterion: Exact 4-digit SIC code match with the target.
- Select up to 6 peers. If more than 6 companies share the SIC, apply tie-breakers in this order:
	1. Filing availability for the requested year (prefer companies with an available 10-K for the year).
	2. Market-cap proximity: choose companies closest in market-cap percentile to the target (use stored `market_cap` in the `peers` DB).
	3. Liquidity filter: prefer companies with average daily volume above a configurable threshold (optional).
	4. Geography / exchange: prefer same primary country (US) when possible.
	5. If still tied, pick by descending `recent_filing_date` (more recently updated filings first) or alphabetically by ticker.

Fallbacks and relaxations

- If fewer than 6 peers are found with exact SIC and filing availability, relax the filing-availability requirement first (include peers without the specific year filing but with other recent filings), then expand to adjacent SICs (e.g., same 2-digit industry group).
- Always exclude the target ticker itself.

Implementation notes

- `scripts/download_peers_and_target.py` will accept `--max-peers` (default 6) and `--require-filing-year` (default true) to control strictness.
- Use `src/sigmak/peer_discovery.py` and the `peers` SQLite table as the authoritative source for candidate peers.
- Ensure idempotency and logging for each ticker processed; return a summary (downloaded / skipped / missing).
