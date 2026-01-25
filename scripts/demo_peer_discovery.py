#!/usr/bin/env python3
"""Simple demo for PeerDiscoveryService.

Usage: python scripts/demo_peer_discovery.py AAPL
"""
import sys
import argparse
from sigmak.peer_discovery import PeerDiscoveryService


def parse_args(argv):
    p = argparse.ArgumentParser(description="Demo peer discovery")
    p.add_argument("ticker", help="Ticker symbol (e.g., AAPL)")
    p.add_argument("--refresh-db", action="store_true", help="Refresh peers in the filings DB for the target's industry")
    p.add_argument("--max-fetch", type=int, default=0, help="Limit number of companies to scan when refreshing (0 = no limit)")
    p.add_argument("--top", type=int, default=10, help="Number of peers to show")
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv[1:])
    ticker = args.ticker.upper()
    svc = PeerDiscoveryService()
    cik = svc.ticker_to_cik(ticker)
    print("Target:", ticker, "=> CIK", cik)
    if not cik:
        return 0
    sic = svc.get_company_sic(cik)
    print("Industry (SIC):", sic)

    if args.refresh_db:
        max_fetch = args.max_fetch if args.max_fetch > 0 else None
        inserted = svc.refresh_peers_for_ticker(ticker, max_fetch=max_fetch)
        print(f"Refreshed peers for SIC {sic}: inserted/updated {inserted} rows")

    peers = svc.find_peers_for_ticker(ticker, top_n=args.top)
    print("Peers:", ", ".join(peers) if peers else "(none found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
