#!/usr/bin/env python3
"""Populate market_cap for peers in `database/sec_filings.db`.

Usage:
  python scripts/populate_peer_marketcap.py --tickers AAPL,MSFT
  python scripts/populate_peer_marketcap.py --all
"""
import argparse
import sys

from sigmak.filings_db import populate_market_cap, DEFAULT_DB_PATH


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Populate market_cap for peers using yfinance")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--tickers", help="Comma-separated tickers to update")
    group.add_argument("--all", action="store_true", help="Update all peers in DB")
    p.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between requests")
    p.add_argument("--db-path", default=DEFAULT_DB_PATH, help="Path to filings DB")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tickers = None
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    try:
        updated = populate_market_cap(args.db_path, tickers=tickers, delay=args.delay)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print(f"Updated market_cap for {updated} peers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
