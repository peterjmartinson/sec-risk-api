# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-Encoder reranker for improving semantic search relevance.
    
    Unlike bi-encoders (which embed query and documents separately),
    cross-encoders process [query, document] pairs jointly, achieving
    higher accuracy at the cost of increased latency.
    
    Use Case: Rerank the top-K candidates from vector search to ensure
    the final top-N results are maximally relevant.
    
    Model: ms-marco-MiniLM-L-6-v2
    - Trained on Microsoft's MARCO dataset (passage ranking)
    - Optimized for query-document relevance scoring
    - Latency: ~20-40ms per pair on CPU
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        """
        Initialize the cross-encoder model.
        
        Args:
            model_name: HuggingFace model identifier for cross-encoder
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name
        logger.info("Cross-encoder model loaded successfully")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidate chunks using cross-encoder scoring.
        
        Process:
        1. Extract text from candidate results
        2. Score each [query, candidate_text] pair using cross-encoder
        3. Sort candidates by cross-encoder score (descending)
        4. Return top_k candidates with rerank_score added
        
        Args:
            query: The original search query
            candidates: List of candidate results from vector search
                       Each must have a "text" field
            top_k: Number of top results to return after reranking
        
        Returns:
            Reranked list of candidates (top_k items), each with added
            "rerank_score" field. Higher scores indicate better relevance.
        """
        if not candidates:
            logger.warning("No candidates to rerank")
            return []

        # Extract texts for scoring
        candidate_texts = [c["text"] for c in candidates]
        
        # Create query-document pairs
        pairs = [[query, text] for text in candidate_texts]
        
        # Score all pairs (cross-encoder returns raw logits)
        logger.info(f"Reranking {len(pairs)} candidates...")
        scores = self.model.predict(pairs)
        
        # Attach scores to candidates
        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        
        # Sort by rerank_score (descending)
        reranked = sorted(
            candidates,
            key=lambda x: x["rerank_score"],
            reverse=True
        )
        
        # Return top_k
        top_results = reranked[:top_k]
        
        logger.info(
            f"Reranking complete. Top score: {top_results[0]['rerank_score']:.4f}"
        )
        
        return top_results

    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.
        
        Useful for unit testing and debugging.
        
        Args:
            query: Search query
            document: Document text to score against query
        
        Returns:
            Relevance score (higher = more relevant)
        """
        score = self.model.predict([[query, document]])[0]
        return float(score)
