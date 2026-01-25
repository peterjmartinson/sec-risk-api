Add Automated Industry Peer Discovery via SEC SIC Codes #84

Objective
Implement a utility that, given a target company's ticker (e.g., "NVDA"), programmatically discovers industry peers using official SEC SIC codes and company metadata, caching results for performance and minimal SEC API impact.

Key Requirements
Provide a Python module/class (e.g., PeerDiscoveryService):
Accepts a ticker and fetches CIK, SIC, and SIC description from official SEC data feeds (see company_tickers.json and company filings endpoints).
Discovers up to N peer tickers with the same SIC code (not including the target).
Includes caching of metadata in ./data/cache/ to avoid repeated lookups.
Prints/logs summary of peer companies with name, ticker, and SIC info.
Add tests to demonstrate retrieval for e.g. NVDA, AMD, INTC.
CLI demo in scripts/demo_peer_discovery.py that prints: Target, Industry, Peers.
Acceptance Criteria
Robust for tickers with/without immediate peers (handles empty results, logs helpful errors)
Cache avoids repeated SEC requests
Clear peer list results validated against manual SEC search
Code well-documented (docstrings, CLI usage)
Unit tests in tests/ verifying CIK/SIC lookup and peer list