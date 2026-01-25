Batch Analyze Target & Peer Filings, Generate Markdown Peer Comparison Report #82

Objective
Implement a Python script that, given a target ticker and year, analyzes risk factors for:

the target company (e.g., NVDA 2024)
its industry peers (any subset with available downloaded filings)
and generates a Markdown report benchmarking the target's risk profile vs. peers.
Key Requirements
CLI utility (scripts/generate_peer_comparison_report.py) that:
Accepts target ticker and year (and optionally peer tickers)
Loads filings for target and peers from the previous download step
Runs the full analysis pipeline (IntegrationPipeline) for each filing
Computes peer comparison stats: severity/novelty percentiles, category divergences, unique/shared risks.
Generates a Markdown report (in output/) showing how the target company compares to its industry peers (risk profile, risk factor distribution, stand-out risks, etc.).
Modular design: independent of downloading/discovery logic (can load any files present)
Logs/prints summary of what was analyzed and any missing filings
Tests for logic using small synthetic/mock filings in tests/data/
Acceptance Criteria
Stand-alone workflow: Run script after discovery + download, get a report with peer benchmarking
Handles subset analysis when some peers have no filing
Clean, actionable, easy-to-copy Markdown output suitable for investors or technical review