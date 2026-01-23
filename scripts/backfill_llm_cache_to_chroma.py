#!/usr/bin/env python3
# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Backfill script to populate llm_risk_classification collection from output/*.json files.

This script reads existing LLM classification results from output/*.json files
and ensures they are properly stored in both SQLite and ChromaDB.

Usage:
    # Dry run (preview changes without writing)
    python scripts/backfill_llm_cache_to_chroma.py --dry-run
    
    # Write to database
    python scripts/backfill_llm_cache_to_chroma.py --write
    
    # Custom paths
    python scripts/backfill_llm_cache_to_chroma.py --write \\
        --output-dir ./custom_output \\
        --db-path ./database/custom.db
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from sigmak.config import get_settings
from sigmak.drift_detection import ClassificationSource, DriftDetectionSystem
from sigmak.embeddings import EmbeddingEngine
from sigmak.llm_classifier import LLMClassificationResult
from sigmak.risk_taxonomy import RiskCategory, validate_category

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class BackfillStats:
    """Statistics for backfill operation."""
    
    def __init__(self) -> None:
        self.files_processed = 0
        self.entries_found = 0
        self.entries_inserted = 0
        self.entries_skipped = 0
        self.errors = 0
    
    def __str__(self) -> str:
        return (
            f"Files processed: {self.files_processed}\n"
            f"Entries found: {self.entries_found}\n"
            f"Entries inserted: {self.entries_inserted}\n"
            f"Entries skipped: {self.entries_skipped}\n"
            f"Errors: {self.errors}"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Backfill LLM classifications from output/*.json to ChromaDB"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to database"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write changes to database"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory containing output JSON files (default: ./output)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to SQLite database (default: from config.yaml)"
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default=None,
        help="Path to ChromaDB persistence directory (default: from config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Must specify either --dry-run or --write
    if not args.dry_run and not args.write:
        parser.error("Must specify either --dry-run or --write")
    
    return args


def load_output_json(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load and parse an output JSON file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        List of risk entries (may be empty if file format doesn't match)
    """
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Check for risks key
            if "risks" in data:
                return data["risks"]
            # Single entry
            return [data]
        
        logger.warning(f"Unexpected JSON format in {file_path}")
        return []
    
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return []


def extract_llm_result(entry: Dict[str, Any]) -> LLMClassificationResult:
    """
    Extract LLM classification result from JSON entry.
    
    Args:
        entry: JSON risk entry
    
    Returns:
        LLMClassificationResult
    
    Raises:
        ValueError: If required fields are missing
    """
    # Validate required fields
    if "category" not in entry:
        raise ValueError("Missing 'category' field")
    
    category = validate_category(entry["category"])
    confidence = float(entry.get("confidence", 0.0))
    evidence = entry.get("llm_evidence", entry.get("evidence", ""))
    rationale = entry.get("llm_rationale", entry.get("rationale", ""))
    model_version = entry.get("model_version", "unknown")
    prompt_version = entry.get("prompt_version", "1")  # Default to v1 for old data
    
    # Parse timestamp
    timestamp_str = entry.get("timestamp", datetime.now().isoformat())
    try:
        timestamp = datetime.fromisoformat(timestamp_str)
    except ValueError:
        timestamp = datetime.now()
    
    return LLMClassificationResult(
        category=category,
        confidence=confidence,
        evidence=evidence,
        rationale=rationale,
        model_version=model_version,
        prompt_version=prompt_version,
        timestamp=timestamp,
        response_time_ms=float(entry.get("response_time_ms", 0.0)),
        input_tokens=int(entry.get("input_tokens", 0)),
        output_tokens=int(entry.get("output_tokens", 0))
    )


def backfill_entry(
    entry: Dict[str, Any],
    drift_system: DriftDetectionSystem,
    embedding_engine: EmbeddingEngine,
    dry_run: bool = True
) -> bool:
    """
    Backfill a single entry into the database.
    
    Args:
        entry: JSON risk entry
        drift_system: DriftDetectionSystem instance
        embedding_engine: EmbeddingEngine instance
        dry_run: If True, don't actually write to database
    
    Returns:
        True if entry was inserted, False if skipped
    """
    # Extract text
    risk_text = entry.get("risk_text", entry.get("text", ""))
    if not risk_text or not risk_text.strip():
        logger.warning("Empty risk_text, skipping entry")
        return False
    
    # Check if LLM classification exists
    classification_method = entry.get("classification_method", "")
    if classification_method != "llm":
        logger.debug(f"Skipping non-LLM entry (method={classification_method})")
        return False
    
    try:
        # Extract LLM result
        llm_result = extract_llm_result(entry)
        
        # Generate embedding
        embeddings = embedding_engine.encode([risk_text])
        embedding = embeddings[0].tolist()
        
        if dry_run:
            logger.info(
                f"[DRY RUN] Would insert: category={llm_result.category.value}, "
                f"confidence={llm_result.confidence:.2f}, "
                f"prompt_version={llm_result.prompt_version}"
            )
            return True
        
        # Insert into database
        record_id, chroma_id = drift_system.insert_classification(
            text=risk_text,
            embedding=embedding,
            llm_result=llm_result,
            source=ClassificationSource.LLM,
            allow_duplicates=False  # Skip duplicates
        )
        
        logger.info(
            f"Inserted: record_id={record_id}, category={llm_result.category.value}"
        )
        return True
    
    except Exception as e:
        logger.error(f"Error processing entry: {e}")
        return False


def main() -> int:
    """Main backfill routine."""
    args = parse_args()
    
    # Initialize settings
    settings = get_settings()
    
    # Use custom paths if provided
    db_path = args.db_path or str(settings.database.sqlite_path)
    chroma_path = args.chroma_path or str(settings.chroma.persist_directory)
    
    logger.info("=" * 60)
    logger.info("LLM Classification Backfill")
    logger.info("=" * 60)
    logger.info(f"Mode: {'DRY RUN' if args.dry_run else 'WRITE'}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Database: {db_path}")
    logger.info(f"ChromaDB: {chroma_path}")
    logger.info("=" * 60)
    
    # Initialize components
    drift_system = DriftDetectionSystem(
        db_path=db_path,
        chroma_path=chroma_path,
        embedding_model=settings.chroma.embedding_model
    )
    
    embedding_engine = EmbeddingEngine(
        model_name=settings.chroma.embedding_model
    )
    
    # Find all JSON files in output directory
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.error(f"Output directory not found: {output_dir}")
        return 1
    
    json_files = list(output_dir.glob("results_*.json"))
    if not json_files:
        logger.warning(f"No results_*.json files found in {output_dir}")
        return 0
    
    logger.info(f"Found {len(json_files)} JSON files")
    
    # Process files
    stats = BackfillStats()
    
    for json_file in sorted(json_files):
        logger.info(f"Processing: {json_file.name}")
        stats.files_processed += 1
        
        entries = load_output_json(json_file)
        stats.entries_found += len(entries)
        
        for entry in entries:
            try:
                inserted = backfill_entry(
                    entry=entry,
                    drift_system=drift_system,
                    embedding_engine=embedding_engine,
                    dry_run=args.dry_run
                )
                
                if inserted:
                    stats.entries_inserted += 1
                else:
                    stats.entries_skipped += 1
            
            except Exception as e:
                logger.error(f"Failed to process entry: {e}")
                stats.errors += 1
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Backfill Summary")
    logger.info("=" * 60)
    logger.info(str(stats))
    logger.info("=" * 60)
    
    if args.dry_run:
        logger.info("DRY RUN complete. Run with --write to persist changes.")
    else:
        logger.info("Backfill complete.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
