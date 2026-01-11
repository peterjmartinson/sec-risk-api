# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
SQLite persistence layer for LLM classification results.

This module provides storage for LLM responses along with their embeddings,
enabling future lookups to reduce reliance on the LLM for repeated queries.

Design Principles:
- Store LLM responses with full provenance (model version, timestamp, tokens)
- Enable fast lookup by text hash to avoid duplicate LLM calls
- Track embeddings for future similarity search
- Maintain audit trail of all LLM classifications
"""

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from sigmak.risk_taxonomy import RiskCategory, validate_category

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================


class StorageError(Exception):
    """Base exception for storage errors."""
    pass


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class LLMStorageRecord:
    """
    Record of an LLM classification result with full provenance.
    
    Attributes:
        text: The risk paragraph text that was classified
        embedding: Vector embedding of the text (384-dim for all-MiniLM-L6-v2)
        category: The classified risk category
        confidence: LLM's confidence score [0.0, 1.0]
        evidence: Quoted text from source that justifies classification
        rationale: LLM's explanation for the classification
        model_version: Gemini model version used
        timestamp: When classification was performed
        response_time_ms: API response time in milliseconds
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
        record_id: Database record ID (populated after insertion)
    """
    text: str
    embedding: List[float]
    category: RiskCategory
    confidence: float
    evidence: str
    rationale: str
    model_version: str
    timestamp: datetime
    response_time_ms: float
    input_tokens: int
    output_tokens: int
    record_id: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate confidence is in valid range."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"Confidence must be in [0.0, 1.0], got {self.confidence}"
            )


# ============================================================================
# SQLite Storage
# ============================================================================


class LLMStorage:
    """
    SQLite storage for LLM classification results.
    
    Usage:
        >>> storage = LLMStorage(db_path="./llm_cache.db")
        >>> record = LLMStorageRecord(...)
        >>> record_id = storage.insert(record)
        >>> results = storage.query_by_text("Some risk text")
    """
    
    # Schema version for migrations
    SCHEMA_VERSION = 1
    
    def __init__(self, db_path: str = "./llm_cache.db") -> None:
        """
        Initialize LLM storage with SQLite database.
        
        Args:
            db_path: Path to SQLite database file
        
        Raises:
            StorageError: If database initialization fails
        """
        self.db_path = db_path
        
        try:
            # Ensure parent directory exists
            db_file = Path(db_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize database
            self._create_schema()
            
            logger.info(f"LLMStorage initialized: db_path={db_path}")
        
        except Exception as e:
            raise StorageError(f"Failed to initialize storage: {e}")
    
    def _create_schema(self) -> None:
        """Create database schema if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Create main table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS llm_classifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    response_time_ms REAL NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index on text_hash for fast lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_text_hash
                ON llm_classifications (text_hash)
            """)
            
            # Create index on category for aggregation queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_category
                ON llm_classifications (category)
            """)
            
            # Create index on timestamp for recency queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON llm_classifications (timestamp DESC)
            """)
            
            conn.commit()
    
    def _compute_text_hash(self, text: str) -> str:
        """
        Compute SHA-256 hash of text for fast lookup.
        
        Args:
            text: Text to hash
        
        Returns:
            Hexadecimal hash string
        """
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def insert(self, record: LLMStorageRecord) -> int:
        """
        Insert a classification record into storage.
        
        Args:
            record: LLM classification record to store
        
        Returns:
            Database record ID
        
        Raises:
            StorageError: If insertion fails
        """
        try:
            text_hash = self._compute_text_hash(record.text)
            embedding_json = json.dumps(record.embedding)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO llm_classifications (
                        text, text_hash, embedding_json, category, confidence,
                        evidence, rationale, model_version, timestamp,
                        response_time_ms, input_tokens, output_tokens
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.text,
                    text_hash,
                    embedding_json,
                    record.category.value,
                    record.confidence,
                    record.evidence,
                    record.rationale,
                    record.model_version,
                    record.timestamp.isoformat(),
                    record.response_time_ms,
                    record.input_tokens,
                    record.output_tokens
                ))
                
                record_id = cursor.lastrowid
                conn.commit()
                
                logger.info(
                    f"Inserted LLM classification: id={record_id}, "
                    f"category={record.category.value}, confidence={record.confidence:.2f}"
                )
                
                # lastrowid should always be an int after successful insert
                assert isinstance(record_id, int), "Failed to get record ID after insert"
                return record_id
        
        except Exception as e:
            raise StorageError(f"Failed to insert record: {e}")
    
    def query_by_text(self, text: str) -> List[LLMStorageRecord]:
        """
        Query records by exact text match.
        
        Args:
            text: Text to search for
        
        Returns:
            List of matching records (may be multiple if text was classified multiple times)
        """
        text_hash = self._compute_text_hash(text)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM llm_classifications
                WHERE text_hash = ?
                ORDER BY timestamp DESC
            """, (text_hash,))
            
            rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]
    
    def query_by_category(self, category: RiskCategory) -> List[LLMStorageRecord]:
        """
        Query all records for a specific risk category.
        
        Args:
            category: Risk category to filter by
        
        Returns:
            List of matching records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM llm_classifications
                WHERE category = ?
                ORDER BY timestamp DESC
            """, (category.value,))
            
            rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]
    
    def query_by_confidence(
        self,
        min_confidence: float,
        max_confidence: float = 1.0
    ) -> List[LLMStorageRecord]:
        """
        Query records by confidence threshold.
        
        Args:
            min_confidence: Minimum confidence score (inclusive)
            max_confidence: Maximum confidence score (inclusive)
        
        Returns:
            List of matching records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM llm_classifications
                WHERE confidence >= ? AND confidence <= ?
                ORDER BY confidence DESC, timestamp DESC
            """, (min_confidence, max_confidence))
            
            rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]
    
    def query_recent(self, limit: int = 10) -> List[LLMStorageRecord]:
        """
        Query most recent classification records.
        
        Args:
            limit: Maximum number of records to return
        
        Returns:
            List of most recent records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM llm_classifications
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [self._row_to_record(row) for row in rows]
    
    def get_total_count(self) -> int:
        """
        Get total number of stored classifications.
        
        Returns:
            Total count of records
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM llm_classifications")
            result = cursor.fetchone()
            if result is None:
                return 0
            return int(result[0])
    
    def get_category_counts(self) -> Dict[str, int]:
        """
        Get count of classifications per category.
        
        Returns:
            Dictionary mapping category names to counts
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT category, COUNT(*) as count
                FROM llm_classifications
                GROUP BY category
                ORDER BY count DESC
            """)
            
            return {row[0]: row[1] for row in cursor.fetchall()}
    
    def _row_to_record(self, row: sqlite3.Row) -> LLMStorageRecord:
        """
        Convert database row to LLMStorageRecord.
        
        Args:
            row: SQLite row object
        
        Returns:
            LLMStorageRecord instance
        """
        return LLMStorageRecord(
            record_id=row['id'],
            text=row['text'],
            embedding=json.loads(row['embedding_json']),
            category=validate_category(row['category']),
            confidence=row['confidence'],
            evidence=row['evidence'],
            rationale=row['rationale'],
            model_version=row['model_version'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            response_time_ms=row['response_time_ms'],
            input_tokens=row['input_tokens'],
            output_tokens=row['output_tokens']
        )
