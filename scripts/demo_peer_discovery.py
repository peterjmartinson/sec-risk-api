#!/usr/bin/env python3
"""Simple demo for PeerDiscoveryService.

Usage: python scripts/demo_peer_discovery.py AAPL
"""
import sys
import argparse
import logging
import threading
import time
from typing import Any, Callable

from sigmak.peer_discovery import PeerDiscoveryService


# Configure basic logging for the demo script
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _run_with_heartbeat(func: Callable[..., Any], *args: Any, heartbeat: str = "working", interval: float = 5.0) -> Any:
    """Run `func(*args)` while printing a heartbeat every `interval` seconds.

    This provides simple feedback when long network/API calls are in progress.
    """
    stop = threading.Event()

    def _hb():
        i = 0
        while not stop.is_set():
            i += 1
            logger.info("%s... (%ds)", heartbeat, int(i * interval))
            stop.wait(interval)

    t = threading.Thread(target=_hb, daemon=True)
    t.start()
    try:
        result = func(*args)
    finally:
        stop.set()
        t.join()
    return result


def parse_args(argv):
    p = argparse.ArgumentParser(description="Demo peer discovery")
    p.add_argument("ticker", help="Ticker symbol (e.g., AAPL)")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    p.add_argument("--refresh-db", action="store_true", help="Refresh peers in the filings DB for the target's industry")
    p.add_argument("--max-fetch", type=int, default=0, help="Limit number of companies to scan when refreshing (0 = no limit)")
    p.add_argument("--top", type=int, default=10, help="Number of peers to show")
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv[1:])
    ticker = args.ticker.upper()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    svc = PeerDiscoveryService()
    cik = svc.ticker_to_cik(ticker)
    logger.info("Target: %s => CIK %s", ticker, cik)
    if not cik:
        return 0
    sic = svc.get_company_sic(cik)
    logger.info("Industry (SIC): %s", sic)

    if args.refresh_db:
        max_fetch = args.max_fetch if args.max_fetch > 0 else None
        logger.info("Refreshing peers for %s (max_fetch=%s)", ticker, max_fetch)
        try:
            inserted = _run_with_heartbeat(svc.refresh_peers_for_ticker, ticker, max_fetch, heartbeat="refreshing peers", interval=5.0)
            logger.info("Refreshed peers for SIC %s: inserted/updated %s rows", sic, inserted)
        except Exception:
            logger.exception("Error while refreshing peers for %s", ticker)

    try:
        peers = _run_with_heartbeat(svc.find_peers_for_ticker, ticker, args.top, heartbeat="finding peers", interval=4.0)
        if peers:
            logger.info("Peers: %s", ", ".join(peers))
        else:
            logger.info("Peers: (none found)")
    except Exception:
        logger.exception("Error while finding peers for %s", ticker)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
