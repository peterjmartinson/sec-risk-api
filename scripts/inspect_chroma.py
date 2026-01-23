#!/usr/bin/env python3
"""Inspect persisted ChromaDB contents.

Single-responsibility functions; CLI flags let you choose which collections
to sample and whether to inspect the raw SQLite file.

Usage: python scripts/inspect_chroma.py --dir chroma_db --max-sample 5
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import traceback
from typing import Any, Dict, List, Optional


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Inspect ChromaDB persistence directory")
    parser.add_argument("--dir", default="chroma_db", help="Chroma persist directory")
    parser.add_argument("--collections", default=None, help="Comma-separated collection names to inspect (default: all)")
    parser.add_argument("--max-sample", type=int, default=5, help="Max sample rows per collection")
    parser.add_argument("--show-docs", action="store_true", help="Include document text samples in output")
    parser.add_argument("--show-metadata", action="store_true", help="Include metadata samples in output")
    parser.add_argument("--inspect-sqlite", action="store_true", help="Also dump sqlite schema from chroma.sqlite3")
    return parser.parse_args(argv)


def import_chromadb() -> Any:
    """Try to import chromadb and return module. Raise informative error on failure."""
    try:
        import chromadb  # type: ignore

        return chromadb
    except Exception as exc:  # pragma: no cover - environment dependent
        raise ImportError("chromadb is not installed or failed to import") from exc


def create_client(persist_dir: str):
    """Create and return a chromadb client pointed at `persist_dir`.

    This handles multiple chromadb versions by trying a Settings-based
    constructor first, then falling back.
    """
    chromadb = import_chromadb()
    # Try Settings API first (newer chromadb)
    try:
        from chromadb.config import Settings  # type: ignore

        client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_dir))
        return client
    except Exception:
        # Fallback: older chromadb style
        try:
            client = chromadb.Client(persist_directory=persist_dir)
            return client
        except Exception:
            # Last-ditch: default client (may open in-memory)
            try:
                client = chromadb.Client()
                return client
            except Exception as exc:  # pragma: no cover - environment dependent
                traceback.print_exc()
                raise RuntimeError("Failed to create chromadb client") from exc


def list_collection_names(client: Any) -> List[str]:
    """Return a list of collection names in the Chroma instance."""
    try:
        cols = client.list_collections()
    except Exception:
        # Older API might expose .list_collections() differently
        try:
            cols = client.get_collections()
        except Exception:
            return []

    names: List[str] = []
    for c in cols:
        # `c` may be a dict or an object
        if isinstance(c, dict):
            name = c.get("name") or c.get("id")
        else:
            name = getattr(c, "name", None) or getattr(c, "id", None)
        if name:
            names.append(name)
    return names


def sample_collection(client: Any, name: str, limit: int = 5, show_docs: bool = False, show_meta: bool = False) -> Dict[str, Any]:
    """Return a small sample from a named collection.

    The returned dict contains counts and sampled ids/documents/metadatas.
    """
    out: Dict[str, Any] = {"name": name, "count": 0, "ids": [], "documents": [], "metadatas": []}
    try:
        col = client.get_collection(name=name)
        # Try modern API
        try:
            sample = col.get(offset=0, limit=limit, include=["metadatas", "documents", "ids", "distances"])
        except Exception:
            # Fallback to simpler include
            try:
                sample = col.get(offset=0, limit=limit, include=["metadatas", "documents", "ids"])
            except Exception:
                sample = {"ids": [], "documents": [], "metadatas": []}

        ids = sample.get("ids", [])
        docs = sample.get("documents", [])
        metas = sample.get("metadatas", [])

        out["count"] = len(ids)
        out["ids"] = ids
        if show_docs:
            out["documents"] = docs
        if show_meta:
            out["metadatas"] = metas
    except Exception:
        out["error"] = traceback.format_exc()
    return out


def inspect_sqlite_schema(sqlite_path: str) -> Dict[str, Any]:
    """Return sqlite master table listing for quick offline inspection."""
    out: Dict[str, Any] = {"sqlite_path": sqlite_path, "tables": []}
    try:
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        cur.execute("SELECT name, type, sql FROM sqlite_master WHERE type IN ('table','index','view')")
        rows = cur.fetchall()
        out["tables"] = [{"name": r[0], "type": r[1], "sql": r[2]} for r in rows]
        conn.close()
    except Exception:
        out["error"] = traceback.format_exc()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    """Orchestrate inspection and print JSON summary to stdout."""
    args = parse_args(argv)
    try:
        client = create_client(args.dir)
    except Exception as exc:
        print("Failed to create chromadb client:", exc, file=sys.stderr)
        return 2

    available = list_collection_names(client)
    requested: List[str]
    if args.collections:
        requested = [c.strip() for c in args.collections.split(",") if c.strip()]
    else:
        requested = available

    result: Dict[str, Any] = {"persist_dir": args.dir, "collections": {}, "available_collections": available}
    for name in requested:
        if name not in available:
            result["collections"][name] = {"error": "collection not found"}
            continue
        sample = sample_collection(client, name, limit=args.max_sample, show_docs=args.show_docs, show_meta=args.show_metadata)
        result["collections"][name] = sample

    if args.inspect_sqlite:
        sqlite_path = f"{args.dir.rstrip('/')}/chroma.sqlite3"
        result["sqlite"] = inspect_sqlite_schema(sqlite_path)

    # Pretty-print JSON to stdout so the caller can redirect or parse
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
