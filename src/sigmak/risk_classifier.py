# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Risk classification with LLM fallback integration.

This module provides threshold-based risk categorization that routes to Gemini LLM
when vector search cannot confidently classify a risk paragraph.

Design Principles:
- Use vector search (ChromaDB) as primary classification method
- Fall back to LLM when similarity scores are below high threshold
- Use LLM for confirmation when scores are uncertain
- Cache all LLM responses for future lookups
- Track full provenance for audit trail
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, List

from sigmak.embeddings import EmbeddingEngine
from sigmak.indexing_pipeline import IndexingPipeline
from sigmak.llm_classifier import GeminiClassifier, LLMClassificationResult
from sigmak.llm_storage import LLMStorage, LLMStorageRecord
from sigmak.risk_taxonomy import RiskCategory, validate_category

logger = logging.getLogger(__name__)


# ============================================================================
# Thresholds
# ============================================================================

# High confidence threshold - use vector search result directly
HIGH_THRESHOLD = 0.80

# Low confidence threshold - fall back to LLM
LOW_THRESHOLD = 0.64


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class RiskClassificationResult:
    """
    Result of risk classification with provenance.
    
    Attributes:
        text: The risk paragraph text
        category: Classified risk category
        confidence: Confidence score [0.0, 1.0]
        method: Classification method used ("vector_search" or "llm")
        similarity_score: Vector similarity score (1 - distance)
        llm_result: LLM classification result (if LLM was used)
        cached: Whether result was retrieved from cache
        timestamp: When classification was performed
    """
    text: str
    category: RiskCategory
    confidence: float
    method: str  # "vector_search" or "llm"
    similarity_score: float
    llm_result: Optional[LLMClassificationResult]
    cached: bool
    timestamp: datetime


# ============================================================================
# Risk Classifier with LLM Fallback
# ============================================================================


class RiskClassifierWithLLM:
    """
    Risk classifier that combines vector search with LLM fallback.
    
    Classification Strategy:
    1. Check cache for previous LLM classification
    2. Perform vector search in ChromaDB
    3. If similarity >= HIGH_THRESHOLD: use vector result
    4. If similarity < LOW_THRESHOLD: fall back to LLM
    5. If LOW_THRESHOLD <= similarity < HIGH_THRESHOLD: use LLM for confirmation
    6. Store LLM results in cache for future lookups
    
    Usage:
        >>> classifier = RiskClassifierWithLLM()
        >>> result = classifier.classify("Supply chain disruptions...")
        >>> print(f"Category: {result.category}, Method: {result.method}")
    """
    
    def __init__(
        self,
        indexing_pipeline: Optional[IndexingPipeline] = None,
        llm_classifier: Optional[GeminiClassifier] = None,
        llm_storage: Optional[LLMStorage] = None,
        embedding_engine: Optional[EmbeddingEngine] = None,
        high_threshold: float = HIGH_THRESHOLD,
        low_threshold: float = LOW_THRESHOLD
    ) -> None:
        """
        Initialize risk classifier with LLM fallback.
        
        Args:
            indexing_pipeline: Pipeline for vector search (will create if None)
            llm_classifier: Gemini classifier (will create if None)
            llm_storage: SQLite storage for LLM results (will create if None)
            embedding_engine: Embedding engine (will create if None)
            high_threshold: Similarity threshold for direct use (default: 0.80)
            low_threshold: Similarity threshold for LLM fallback (default: 0.64)
        """
        self.pipeline = indexing_pipeline or IndexingPipeline()
        self.llm_classifier = llm_classifier or GeminiClassifier()
        self.llm_storage = llm_storage or LLMStorage()
        self.embedding_engine = embedding_engine or EmbeddingEngine()
        
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        
        logger.info(
            f"RiskClassifierWithLLM initialized: "
            f"high_threshold={high_threshold}, low_threshold={low_threshold}"
        )
    
    def classify(
        self,
        text: str,
        use_cache: bool = True
    ) -> RiskClassificationResult:
        """
        Classify a risk paragraph with LLM fallback.
        
        Args:
            text: Risk paragraph text to classify
            use_cache: Whether to check cache for previous LLM classifications
        
        Returns:
            RiskClassificationResult with category and provenance
        """
        timestamp = datetime.now()
        
        # Step 1: Check cache for previous LLM classification
        if use_cache:
            cached_results = self.llm_storage.query_by_text(text)
            if cached_results:
                # Use most recent cached result
                cached = cached_results[0]
                logger.info(f"Found cached LLM result for text (hash: {text[:50]}...)")
                
                # Convert similarity from distance (lower = better) to score (higher = better)
                # For cached results, we use confidence as a proxy for similarity
                similarity_score = cached.confidence
                
                return RiskClassificationResult(
                    text=text,
                    category=cached.category,
                    confidence=cached.confidence,
                    method="llm",
                    similarity_score=similarity_score,
                    llm_result=None,  # Already cached
                    cached=True,
                    timestamp=timestamp
                )
        
        # Step 2: Search unified collection for classified chunks
        # Generate embedding for the query text
        query_embedding = self.embedding_engine.encode([text])[0]
        query_embedding_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding
        
        # Query ChromaDB directly for classified chunks only
        # Note: Using $ne "" instead of $ne None because ChromaDB doesn't support None in filters
        try:
            search_results = self.pipeline.collection.query(
                query_embeddings=[query_embedding_list],
                n_results=5,  # Get top 5 to find best classified match
                where={"category": {"$ne": ""}}  # Only classified chunks (non-empty category)
            )
            
            classified_matches = []
            if search_results['ids'] and len(search_results['ids'][0]) > 0:
                for i in range(len(search_results['ids'][0])):
                    distance = search_results['distances'][0][i]
                    similarity_score = 1.0 - (distance / 2.0)  # Convert to [0, 1]
                    
                    classified_matches.append({
                        "metadata": search_results['metadatas'][0][i],
                        "distance": distance,
                        "similarity_score": similarity_score,
                        "text": search_results['documents'][0][i]
                    })
            
            if classified_matches:
                top_match = classified_matches[0]
                similarity_score = top_match["similarity_score"]
                
                logger.info(
                    f"Found classified chunk: similarity={similarity_score:.4f}, "
                    f"category={top_match['metadata'].get('category')}"
                )
                
                # Step 3: Apply threshold logic
                if similarity_score >= self.high_threshold:
                    # High confidence - use cached classification
                    category = self._extract_category_from_metadata(top_match["metadata"])
                    
                    logger.info(
                        f"High confidence ({similarity_score:.2f} >= {self.high_threshold}) - "
                        f"using cached classification"
                    )
                    
                    return RiskClassificationResult(
                        text=text,
                        category=category,
                        confidence=similarity_score,
                        method="vector_search",
                        similarity_score=similarity_score,
                        llm_result=None,
                        cached=True,
                        timestamp=timestamp
                    )
                else:
                    logger.info(
                        f"Low/uncertain confidence ({similarity_score:.2f} < {self.high_threshold}) - "
                        f"falling back to LLM"
                    )
            else:
                logger.info("No classified chunks found - falling back to LLM")
        
        except Exception as e:
            logger.warning(f"Error searching classified chunks: {e} - falling back to LLM")
        
        # Step 4: Fall back to LLM
        return self._classify_with_llm(text, 0.0, timestamp)
    
    def _classify_with_llm(
        self,
        text: str,
        vector_similarity: float,
        timestamp: datetime,
        ticker: Optional[str] = None,
        filing_year: Optional[int] = None,
        chunk_index: Optional[int] = None
    ) -> RiskClassificationResult:
        """
        Classify using Gemini LLM and store result.
        
        Args:
            text: Risk paragraph text
            vector_similarity: Similarity score from vector search
            timestamp: Classification timestamp
            ticker: Optional ticker (if known, will update unified collection)
            filing_year: Optional filing year (if known)
            chunk_index: Optional chunk index (if known)
        
        Returns:
            RiskClassificationResult
        """
        # Call LLM
        llm_result = self.llm_classifier.classify(text)
        
        # Generate embedding for storage
        embedding_result = self.embedding_engine.encode([text])[0]
        # Handle both numpy arrays and lists (for testing)
        embedding = embedding_result.tolist() if hasattr(embedding_result, 'tolist') else embedding_result
        
        # Store in SQLite cache (legacy)
        storage_record = LLMStorageRecord(
            text=text,
            embedding=embedding,
            category=llm_result.category,
            confidence=llm_result.confidence,
            evidence=llm_result.evidence,
            rationale=llm_result.rationale,
            model_version=llm_result.model_version,
            timestamp=llm_result.timestamp,
            response_time_ms=llm_result.response_time_ms,
            input_tokens=llm_result.input_tokens,
            output_tokens=llm_result.output_tokens
        )
        
        self.llm_storage.insert(storage_record)
        
        # Update unified collection if we have chunk identifiers
        if ticker and filing_year is not None and chunk_index is not None:
            try:
                self.pipeline.update_chunk_classification(
                    ticker=ticker,
                    filing_year=filing_year,
                    chunk_index=chunk_index,
                    category=llm_result.category.value,
                    confidence=llm_result.confidence,
                    source="llm",
                    model_version=llm_result.model_version,
                    prompt_version=llm_result.prompt_version
                )
                logger.info(
                    f"Updated unified collection for {ticker}_{filing_year}_{chunk_index}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to update unified collection: {e} - "
                    f"classification still cached in SQLite"
                )
        
        logger.info(
            f"LLM classification stored: category={llm_result.category.value}, "
            f"confidence={llm_result.confidence:.2f}"
        )
        
        return RiskClassificationResult(
            text=text,
            category=llm_result.category,
            confidence=llm_result.confidence,
            method="llm",
            similarity_score=vector_similarity,
            llm_result=llm_result,
            cached=False,
            timestamp=timestamp
        )
    
    def _extract_category_from_metadata(self, metadata: Dict[str, Any]) -> RiskCategory:
        """
        Extract risk category from search result metadata.
        
        Args:
            metadata: Search result metadata
        
        Returns:
            RiskCategory enum
        
        Raises:
            ValueError: If metadata missing category or category invalid
        """
        if "category" not in metadata or metadata["category"] is None:
            raise ValueError("Metadata missing category field")
        
        return validate_category(metadata["category"])
    
    def classify_batch(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[RiskClassificationResult]:
        """
        Classify multiple risk paragraphs.
        
        Args:
            texts: List of risk paragraph texts
            use_cache: Whether to use cache for lookups
        
        Returns:
            List of RiskClassificationResult objects
        """
        return [self.classify(text, use_cache=use_cache) for text in texts]
