## [2026-01-25] Peer Discovery: Prefetch/backfill and DB-first discovery

### Status: COMPLETE ✓

### Summary
Added richer `peers` schema and a one-time prefetch/backfill utility to populate the table from cached `submissions_*.json` files. Modified peer discovery to query the `peers` table (DB-first) to avoid running thousands of live SEC requests during discovery.

### What I changed
- `src/sigmak/filings_db.py`: extended `peers` schema (name, sic_description, state_of_incorporation, fiscal_year_end, website, phone, recent_filing_count, latest_filing_date, latest_10k_date, submissions_cached_at, etc.) and migration logic; `upsert_peer` extended to accept and preserve these fields.
- `src/sigmak/prefetch_peers.py`: new module to scan cached `submissions_*.json` files and upsert richer peer metadata into the DB.
- `src/sigmak/peer_discovery.py`: `find_peers_for_ticker` now queries the `peers` table first and will not trigger large-scale live SEC fetching; use `prefetch_peers` or `refresh_peers_for_ticker` for deliberate backfills.
- `tests/test_prefetch_peers.py`: unit test for the prefetch flow.

### How to use
1. Run a one-time prefetch/backfill from your local cache:
```
PYTHONPATH=src python -m sigmak.prefetch_peers --cache-dir data/peer_discovery --db database/sec_filings.db
```
2. After prefetch, `PeerDiscoveryService.find_peers_for_ticker()` will return peers from the DB without performing live SEC calls.

### Notes
- The prefetch is idempotent and safe to re-run; missing market caps are intentionally left NULL and can be populated with `sigmak.filings_db.populate_market_cap()`.
