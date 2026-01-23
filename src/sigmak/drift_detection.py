# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Drift detection system for risk classification quality monitoring.

This module implements a hybrid classification storage system with SQLite + ChromaDB,
periodic review jobs to detect classification drift, and embedding versioning for
model upgrades.

Design Principles:
- Dual storage: SQLite for provenance, ChromaDB for semantic search
- Periodic review: Sample low-confidence and old classifications
- Drift metrics: Agreement rate between original and re-classification
- Archive versioning: Preserve old embeddings when model changes
- Full audit trail: Every classification tracked with source and timestamp

Architecture:
1. DriftDetectionSystem: Core storage with SQLite + ChromaDB integration
2. DriftReviewJob: Periodic job (APScheduler) to detect drift
3. DriftMetrics: Statistics on classification agreement/drift
4. ClassificationSource: Enum tracking classification origin (LLM vs vector)
"""

import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from uuid import uuid4

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection

from sigmak.llm_classifier import GeminiClassifier, LLMClassificationResult
from sigmak.risk_taxonomy import RiskCategory, validate_category

logger = logging.getLogger(__name__)


# ============================================================================
# Enums and Constants
# ============================================================================


class ClassificationSource(Enum):
    """Source of a risk classification."""
    LLM = "llm"  # Gemini LLM classification
    VECTOR = "vector_search"  # ChromaDB similarity search
    MANUAL = "manual"  # Human-provided label
    

# Drift thresholds
WARNING_THRESHOLD = 0.85  # Below 85% agreement triggers warning
CRITICAL_THRESHOLD = 0.75  # Below 75% triggers manual review


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class DriftMetrics:
    """
    Metrics from periodic drift review.
    
    Attributes:
        total_reviewed: Number of classifications reviewed
        agreements: Count where new classification matches old
        disagreements: Count where classifications differ
        agreement_rate: Fraction of agreements [0.0, 1.0]
        avg_confidence_change: Mean change in confidence scores
        timestamp: When review was performed
    """
    total_reviewed: int
    agreements: int
    disagreements: int
    agreement_rate: float
    avg_confidence_change: float
    timestamp: datetime
    
    def requires_manual_review(self, critical_threshold: float = CRITICAL_THRESHOLD) -> bool:
        """Check if agreement rate triggers manual review."""
        return self.agreement_rate < critical_threshold
    
    def is_warning(self, warning_threshold: float = WARNING_THRESHOLD) -> bool:
        """Check if agreement rate triggers warning."""
        return self.agreement_rate < warning_threshold


# ============================================================================
# Drift Detection System
# ============================================================================


class DriftDetectionSystem:
    """
    Hybrid classification storage with drift detection.
    
    Features:
    - Dual storage: SQLite (provenance) + ChromaDB (semantic search)
    - Cross-referencing via chroma_id in SQLite
    - Periodic sampling for drift review
    - Embedding versioning and archival
    - Duplicate detection via text hashing
    
    Usage:
        >>> system = DriftDetectionSystem(db_path="./database/risk_classifications.db")
        >>> record_id, chroma_id = system.insert_classification(...)
        >>> results = system.similarity_search(query_embedding, n_results=5)
    """
    
    SCHEMA_VERSION = 2  # Incremented for drift detection schema
    
    def __init__(
        self,
        db_path: str = "./database/risk_classifications.db",
        chroma_path: str = "./database",
        embedding_model: str = "all-MiniLM-L6-v2"
    ) -> None:
        """
        Initialize drift detection system.
        
        Args:
            db_path: Path to SQLite database
            chroma_path: Path to ChromaDB persistent storage
            embedding_model: Name/version of embedding model
        """
        self.db_path = db_path
        self.chroma_path = chroma_path
        self.embedding_model = embedding_model
        
        # Ensure directories exist
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        
        # Initialize databases
        self._create_schema()
        self._initialize_chroma()
        
        logger.info(f"DriftDetectionSystem initialized: db={db_path}, chroma={chroma_path}")
    
    def _create_schema(self) -> None:
        """Create SQLite schema with drift detection fields."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main classifications table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_classifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    chroma_id TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    category TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    response_time_ms REAL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    last_reviewed_at TEXT,
                    review_count INTEGER DEFAULT 0,
                    archive_version INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Embedding archives table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embedding_archives (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    classification_id INTEGER NOT NULL,
                    embedding_json TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    archived_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (classification_id) REFERENCES risk_classifications(id)
                )
            """)
            
            # Drift metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS drift_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_reviewed INTEGER NOT NULL,
                    agreements INTEGER NOT NULL,
                    disagreements INTEGER NOT NULL,
                    agreement_rate REAL NOT NULL,
                    avg_confidence_change REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indices
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_text_hash
                ON risk_classifications (text_hash)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_chroma_id
                ON risk_classifications (chroma_id)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_confidence
                ON risk_classifications (confidence)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON risk_classifications (timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_source
                ON risk_classifications (source)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_last_reviewed
                ON risk_classifications (last_reviewed_at)
            """)
            
            conn.commit()
    
    def _initialize_chroma(self) -> None:
        """Initialize ChromaDB client and collection."""
        self.chroma_client: ClientAPI = chromadb.PersistentClient(path=self.chroma_path)
        self.collection: Collection = self.chroma_client.get_or_create_collection(
            name="risk_classifications",
            metadata={"hnsw:space": "cosine"}
        )
    
    def _compute_text_hash(self, text: str) -> str:
        """Compute SHA-256 hash of text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def insert_classification(
        self,
        text: str,
        embedding: List[float],
        llm_result: LLMClassificationResult,
        source: ClassificationSource,
        allow_duplicates: bool = True
    ) -> Tuple[int, str]:
        """
        Insert classification into both SQLite and ChromaDB.
        
        Args:
            text: Risk paragraph text
            embedding: 384-dimensional vector
            llm_result: LLM classification result with provenance
            source: Classification source (LLM, vector, manual)
            allow_duplicates: If False, return existing record for duplicate text
        
        Returns:
            Tuple of (sqlite_record_id, chroma_document_id)
        """
        text_hash = self._compute_text_hash(text)
        
        # Check for existing record
        if not allow_duplicates:
            existing = self._get_by_text_hash(text_hash)
            if existing:
                logger.info(f"Duplicate detected: returning existing record {existing['id']}")
                return existing['id'], existing['chroma_id']
        
        # Generate ChromaDB ID
        chroma_id = str(uuid4())
        
        # Insert into SQLite
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO risk_classifications (
                    text, text_hash, chroma_id, embedding_json, category, confidence,
                    evidence, rationale, model_version, prompt_version, embedding_model, source,
                    timestamp, response_time_ms, input_tokens, output_tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                text,
                text_hash,
                chroma_id,
                json.dumps(embedding),
                llm_result.category.value,
                llm_result.confidence,
                llm_result.evidence,
                llm_result.rationale,
                llm_result.model_version,
                llm_result.prompt_version,
                self.embedding_model,
                source.value,
                llm_result.timestamp.isoformat(),
                llm_result.response_time_ms,
                llm_result.input_tokens,
                llm_result.output_tokens
            ))
            
            record_id = cursor.lastrowid
            conn.commit()
        
        # Insert into ChromaDB
        self.collection.add(
            ids=[chroma_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[{
                "record_id": record_id,
                "category": llm_result.category.value,
                "confidence": llm_result.confidence,
                "source": source.value,
                "prompt_version": llm_result.prompt_version,
                "timestamp": llm_result.timestamp.isoformat(),
                "origin_text": text[:500]  # Store first 500 chars for provenance
            }]
        )
        
        logger.info(
            f"Inserted classification: id={record_id}, chroma_id={chroma_id}, "
            f"category={llm_result.category.value}, source={source.value}"
        )
        
        assert isinstance(record_id, int)
        return record_id, chroma_id
    
    def similarity_search(
        self,
        query_embedding: List[float],
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar classifications using ChromaDB.
        
        Args:
            query_embedding: Query vector
            n_results: Number of results to return
        
        Returns:
            List of records with full SQLite provenance
        """
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Enrich with SQLite data
        enriched_results: List[Dict[str, Any]] = []
        
        if results['ids'] and len(results['ids'][0]) > 0:
            for i, chroma_id in enumerate(results['ids'][0]):
                record = self._get_by_chroma_id(chroma_id)
                if record:
                    # Add similarity score (1 - distance for cosine)
                    record['similarity_score'] = 1.0 - results['distances'][0][i]
                    enriched_results.append(record)
        
        return enriched_results
    
    def get_record_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Get classification record by SQLite ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM risk_classifications WHERE id = ?
            """, (record_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def _get_by_chroma_id(self, chroma_id: str) -> Optional[Dict[str, Any]]:
        """Get record by ChromaDB ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM risk_classifications WHERE chroma_id = ?
            """, (chroma_id,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def _get_by_text_hash(self, text_hash: str) -> Optional[Dict[str, Any]]:
        """Get most recent record by text hash."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM risk_classifications
                WHERE text_hash = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (text_hash,))
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def sample_for_review(
        self,
        min_confidence: float = 0.0,
        max_confidence: float = 0.75,
        sample_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Sample low-confidence classifications for review.
        
        Args:
            min_confidence: Minimum confidence threshold
            max_confidence: Maximum confidence threshold
            sample_size: Number of records to sample
        
        Returns:
            List of sampled records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM risk_classifications
                WHERE confidence >= ? AND confidence <= ?
                ORDER BY RANDOM()
                LIMIT ?
            """, (min_confidence, max_confidence, sample_size))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def sample_old_records(
        self,
        before_date: datetime,
        sample_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Sample old classifications for drift detection.
        
        Args:
            before_date: Sample records before this date
            sample_size: Number of records to sample
        
        Returns:
            List of sampled old records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM risk_classifications
                WHERE timestamp < ?
                ORDER BY RANDOM()
                LIMIT ?
            """, (before_date.isoformat(), sample_size))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def update_review_metadata(self, record_id: int) -> None:
        """Update review timestamp and count."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE risk_classifications
                SET last_reviewed_at = ?, review_count = review_count + 1
                WHERE id = ?
            """, (datetime.now().isoformat(), record_id))
            
            conn.commit()
    
    def log_drift_metrics(self, metrics: DriftMetrics) -> None:
        """Store drift metrics in database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO drift_metrics (
                    total_reviewed, agreements, disagreements,
                    agreement_rate, avg_confidence_change, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metrics.total_reviewed,
                metrics.agreements,
                metrics.disagreements,
                metrics.agreement_rate,
                metrics.avg_confidence_change,
                metrics.timestamp.isoformat()
            ))
            
            conn.commit()
        
        logger.info(
            f"Drift metrics logged: agreement_rate={metrics.agreement_rate:.2%}, "
            f"reviewed={metrics.total_reviewed}"
        )
    
    def get_recent_drift_metrics(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent drift review results."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM drift_metrics
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def archive_and_update_embedding(
        self,
        record_id: int,
        new_embedding: List[float],
        new_model_version: str
    ) -> None:
        """
        Archive old embedding and update to new version.
        
        Args:
            record_id: SQLite record ID
            new_embedding: New embedding vector
            new_model_version: New embedding model version
        """
        # Get current record
        record = self.get_record_by_id(record_id)
        if not record:
            raise ValueError(f"Record {record_id} not found")
        
        # Archive old embedding
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO embedding_archives (
                    classification_id, embedding_json, embedding_model
                ) VALUES (?, ?, ?)
            """, (
                record_id,
                record['embedding_json'],
                record['embedding_model']
            ))
            
            # Update with new embedding
            cursor.execute("""
                UPDATE risk_classifications
                SET embedding_json = ?,
                    embedding_model = ?,
                    archive_version = archive_version + 1
                WHERE id = ?
            """, (
                json.dumps(new_embedding),
                new_model_version,
                record_id
            ))
            
            conn.commit()
        
        # Update ChromaDB
        self.collection.update(
            ids=[record['chroma_id']],
            embeddings=[new_embedding]
        )
        
        logger.info(f"Archived and updated embedding for record {record_id}")
    
    def get_embedding_archives(self, record_id: int) -> List[Dict[str, Any]]:
        """Get archived embeddings for a record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM embedding_archives
                WHERE classification_id = ?
                ORDER BY archived_at DESC
            """, (record_id,))
            
            archives = []
            for row in cursor.fetchall():
                archive = dict(row)
                archive['embedding'] = json.loads(archive['embedding_json'])
                del archive['embedding_json']
                archives.append(archive)
            
            return archives
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics on embedding model versions."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Total records
            cursor.execute("SELECT COUNT(*) FROM risk_classifications")
            total_records = cursor.fetchone()[0]
            
            # Current model distribution
            cursor.execute("""
                SELECT embedding_model, COUNT(*) as count
                FROM risk_classifications
                GROUP BY embedding_model
            """)
            model_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Archived versions
            cursor.execute("""
                SELECT COUNT(DISTINCT classification_id) as archived_count
                FROM embedding_archives
            """)
            archived_count = cursor.fetchone()[0]
            
            return {
                "total_records": total_records,
                "current_model_version": self.embedding_model,
                "model_distribution": model_counts,
                "archived_versions": archived_count
            }


# ============================================================================
# Drift Review Job
# ============================================================================


class DriftReviewJob:
    """
    Periodic job to detect classification drift.
    
    Process:
    1. Sample low-confidence classifications
    2. Sample old classifications
    3. Re-classify with current LLM
    4. Compare with original classification
    5. Calculate agreement rate
    6. Log metrics and trigger alerts
    
    Usage:
        >>> job = DriftReviewJob(drift_system)
        >>> metrics = job.run_review(sample_size=50)
        >>> if metrics.requires_manual_review():
        ...     send_alert(metrics)
    """
    
    def __init__(
        self,
        drift_system: DriftDetectionSystem,
        llm_classifier: Optional[GeminiClassifier] = None
    ) -> None:
        """
        Initialize drift review job.
        
        Args:
            drift_system: Drift detection system instance
            llm_classifier: Optional Gemini classifier (creates if None)
        """
        self.drift_system = drift_system
        self.llm_classifier = llm_classifier or GeminiClassifier()
    
    def run_review(
        self,
        sample_size: int = 20,
        low_conf_ratio: float = 0.6,
        old_days: int = 90
    ) -> DriftMetrics:
        """
        Run periodic drift review.
        
        Args:
            sample_size: Total number of records to review
            low_conf_ratio: Fraction of samples from low-confidence pool
            old_days: Sample records older than this many days
        
        Returns:
            DriftMetrics with agreement statistics
        """
        logger.info(f"Starting drift review: sample_size={sample_size}")
        
        # Sample low-confidence records
        low_conf_size = int(sample_size * low_conf_ratio)
        low_conf_records = self.drift_system.sample_for_review(
            max_confidence=0.75,
            sample_size=low_conf_size
        )
        
        # Sample old records
        old_size = sample_size - len(low_conf_records)
        cutoff_date = datetime.now() - timedelta(days=old_days)
        old_records = self.drift_system.sample_old_records(
            before_date=cutoff_date,
            sample_size=old_size
        )
        
        # Combine samples
        sample_records = low_conf_records + old_records
        
        # Re-classify and compare
        agreements = 0
        disagreements = 0
        confidence_changes: List[float] = []
        
        for record in sample_records:
            try:
                # Re-classify with LLM
                new_result = self.llm_classifier.classify(record['text'])
                
                # Compare categories
                old_category = validate_category(record['category'])
                if new_result.category == old_category:
                    agreements += 1
                else:
                    disagreements += 1
                    logger.warning(
                        f"Drift detected: record {record['id']} changed from "
                        f"{old_category.value} to {new_result.category.value}"
                    )
                
                # Track confidence change
                confidence_change = new_result.confidence - record['confidence']
                confidence_changes.append(confidence_change)
                
                # Update review metadata
                self.drift_system.update_review_metadata(record['id'])
            
            except Exception as e:
                logger.error(f"Failed to review record {record['id']}: {e}")
                continue
        
        # Calculate metrics
        total_reviewed = agreements + disagreements
        agreement_rate = agreements / total_reviewed if total_reviewed > 0 else 0.0
        avg_confidence_change = sum(confidence_changes) / len(confidence_changes) if confidence_changes else 0.0
        
        metrics = DriftMetrics(
            total_reviewed=total_reviewed,
            agreements=agreements,
            disagreements=disagreements,
            agreement_rate=agreement_rate,
            avg_confidence_change=avg_confidence_change,
            timestamp=datetime.now()
        )
        
        # Log metrics
        self.drift_system.log_drift_metrics(metrics)
        
        # Check thresholds
        if metrics.requires_manual_review():
            logger.critical(
                f"MANUAL REVIEW REQUIRED: Agreement rate {metrics.agreement_rate:.1%} "
                f"below critical threshold {CRITICAL_THRESHOLD:.1%}"
            )
        elif metrics.is_warning():
            logger.warning(
                f"Drift warning: Agreement rate {metrics.agreement_rate:.1%} "
                f"below warning threshold {WARNING_THRESHOLD:.1%}"
            )
        else:
            logger.info(
                f"Drift review passed: Agreement rate {metrics.agreement_rate:.1%}"
            )
        
        return metrics
