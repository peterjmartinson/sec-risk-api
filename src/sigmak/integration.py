"""
Integration pipeline for end-to-end risk analysis (Issue #22).

This module provides the "Walking Skeleton" that orchestrates the full
retrieval-scoring pipeline from raw SEC filing to structured risk analysis.

Flow:
1. Ingest HTML filing → Extract Item 1A
2. Chunk and embed text
3. Index into vector database
4. Retrieve risk chunks
5. Compute severity and novelty scores
6. Return structured, cited JSON output

Usage:
    >>> pipeline = IntegrationPipeline()
    >>> result = pipeline.analyze_filing(
    ...     html_path="data/sample_10k.html",
    ...     ticker="AAPL",
    ...     filing_year=2025
    ... )
    >>> print(result.to_json())
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import json
import re

from sigmak.indexing_pipeline import IndexingPipeline
from sigmak.scoring import RiskScorer, RiskScore
from sigmak.risk_classifier import RiskClassifierWithLLM
from sigmak.llm_classifier import GeminiClassifier
from sigmak.llm_storage import LLMStorage
from sigmak.drift_detection import DriftDetectionSystem, ClassificationSource

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class IntegrationError(Exception):
    """
    Raised when integration pipeline fails.
    
    Examples:
    - Missing HTML file
    - Invalid HTML structure
    - Missing Item 1A section
    - Invalid ticker/year
    """
    pass


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class RiskAnalysisResult:
    """
    Container for complete risk analysis results.
    
    Attributes:
        ticker: Stock symbol (e.g., "AAPL")
        filing_year: Filing year (e.g., 2025)
        risks: List of risk dictionaries with scores and citations
        metadata: Pipeline execution metadata (timing, counts, etc.)
    
    Example:
        >>> result = RiskAnalysisResult(
        ...     ticker="AAPL",
        ...     filing_year=2025,
        ...     risks=[{
        ...         "text": "Risk disclosure...",
        ...         "severity": {"value": 0.75, "explanation": "..."},
        ...         "novelty": {"value": 0.82, "explanation": "..."},
        ...         "source_citation": "Risk disclosure...",
        ...         "metadata": {"ticker": "AAPL", "filing_year": 2025}
        ...     }],
        ...     metadata={"chunks_indexed": 5, "total_latency_ms": 1234.5}
        ... )
    """
    ticker: str
    filing_year: int
    risks: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        return {
            "ticker": self.ticker,
            "filing_year": self.filing_year,
            "risks": self.risks,
            "metadata": self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert result to formatted JSON string.
        
        Args:
            indent: Number of spaces for indentation (default: 2)
        
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)


# ============================================================================
# Integration Pipeline
# ============================================================================

class IntegrationPipeline:
    """
    End-to-end pipeline for SEC risk analysis.
    
    Orchestrates:
    - Ingestion (HTML parsing)
    - Indexing (chunking + embedding + storage)
    - Retrieval (semantic search)
    - Scoring (severity + novelty)
    
    Usage:
        >>> pipeline = IntegrationPipeline(persist_path="./chroma_db")
        >>> result = pipeline.analyze_filing(
        ...     html_path="data/sample_10k.html",
        ...     ticker="AAPL",
        ...     filing_year=2025
        ... )
        >>> print(f"Analyzed {len(result.risks)} risks")
    """
    
    def __init__(
        self,
        persist_path: str = "./database",
        use_llm: bool = False,
        db_only_classification: bool = False,
        db_only_similarity_threshold: float = 0.8
    ) -> None:
        """
        Initialize integration pipeline.
        
        Args:
            persist_path: Path to vector database storage
            use_llm: Enable LLM-based risk classification (default: False)
        """
        self.persist_path = persist_path
        self.use_llm = use_llm
        # When True, classifier will only consult the vector DB and never call the LLM
        self.db_only_classification = db_only_classification
        # Similarity threshold (0-1) for accepting a DB match in DB-only mode
        self.db_only_similarity_threshold = db_only_similarity_threshold
        
        # Initialize sub-components
        self.indexing_pipeline = IndexingPipeline(persist_path=persist_path)
        self.scorer = RiskScorer()
        # Lazy-load DriftDetectionSystem for persisting LLM classifications
        self._drift_system: Optional[DriftDetectionSystem] = None
        
        # Lazy-load classifiers only when needed
        self._llm_classifier: Optional[RiskClassifierWithLLM] = None
        self._gemini_classifier: Optional[GeminiClassifier] = None
        
        logger.info(f"Initialized IntegrationPipeline with DB at {persist_path}, use_llm={use_llm}")
    
    @property
    def gemini_classifier(self) -> GeminiClassifier:
        """Lazy-load direct Gemini classifier (bypasses threshold logic)."""
        if self._gemini_classifier is None:
            logger.info("Initializing Gemini classifier...")
            self._gemini_classifier = GeminiClassifier()
        return self._gemini_classifier

    @property
    def drift_system(self) -> DriftDetectionSystem:
        """Lazily create a DriftDetectionSystem instance for persistence."""
        if self._drift_system is None:
            db_path = str(Path(self.persist_path) / "risk_classifications.db")
            chroma_path = str(self.persist_path)
            # Use default embedding model consistent with IndexingPipeline
            self._drift_system = DriftDetectionSystem(
                db_path=db_path,
                chroma_path=chroma_path,
                embedding_model="all-MiniLM-L6-v2",
            )
        return self._drift_system
    
    @property
    def llm_classifier(self) -> RiskClassifierWithLLM:
        """Lazy-load LLM classifier only when needed."""
        if self._llm_classifier is None:
            logger.info("Initializing LLM classifier...")
            llm = GeminiClassifier()
            storage = LLMStorage()
            self._llm_classifier = RiskClassifierWithLLM(
                indexing_pipeline=self.indexing_pipeline,
                llm_classifier=llm,
                llm_storage=storage
            )
        return self._llm_classifier
    
    # ========================================================================
    # Main Analysis Method
    # ========================================================================
    
    def analyze_filing(
        self,
        html_path: str,
        ticker: str,
        filing_year: int,
        retrieve_top_k: int = 10,
        rerank: bool = True
    ) -> RiskAnalysisResult:
        """
        Analyze a SEC filing end-to-end.
        
        Pipeline:
        1. Validate inputs
        2. Index filing into vector database
        3. Retrieve top-k risk chunks
        4. Score each chunk (severity + novelty)
        5. Return structured results
        
        Args:
            html_path: Path to HTML filing
            ticker: Stock symbol (e.g., "AAPL")
            filing_year: Filing year (e.g., 2025)
            retrieve_top_k: Number of top risks to analyze (default: 10)
        
        Returns:
            RiskAnalysisResult with scored risks and metadata
        
        Raises:
            IntegrationError: If validation fails or processing errors occur
        
        Example:
            >>> pipeline = IntegrationPipeline()
            >>> result = pipeline.analyze_filing(
            ...     html_path="data/apple_10k.html",
            ...     ticker="AAPL",
            ...     filing_year=2025
            ... )
            >>> print(f"Found {len(result.risks)} risks")
            >>> print(f"Avg severity: {sum(r['severity']['value'] for r in result.risks) / len(result.risks):.2f}")
        """
        import time
        start_time = time.time()
        
        # Step 1: Validate inputs
        self._validate_inputs(html_path, ticker, filing_year)
        
        # Step 2: Index filing
        try:
            logger.info(f"Indexing {ticker} {filing_year} from {html_path}")
            index_stats = self.indexing_pipeline.index_filing(
                html_path=html_path,
                ticker=ticker,
                filing_year=filing_year,
                item_type="Item 1A"
            )
            logger.info(f"Indexed {index_stats['chunks_indexed']} chunks")
        except FileNotFoundError as e:
            raise IntegrationError(f"HTML file not found at path: {html_path}") from e
        except ValueError as e:
            if "Item 1A" in str(e):
                raise IntegrationError(f"Item 1A section not found in filing") from e
            raise IntegrationError(f"Invalid HTML or parsing error: {e}") from e
        except Exception as e:
            raise IntegrationError(f"Indexing failed: {e}") from e
        
        # Step 3: Retrieve top-k risk chunks
        try:
            logger.info(f"Retrieving top {retrieve_top_k} risks for {ticker} (rerank={rerank})")
            search_results = self.indexing_pipeline.semantic_search(
                query="risk factors financial operational regulatory geopolitical",
                n_results=retrieve_top_k,
                where={"ticker": ticker, "filing_year": filing_year},
                rerank=rerank
            )
        except Exception as e:
            raise IntegrationError(f"Semantic search failed: {e}") from e
        
        # Step 4: Score each chunk
        risks = []
        for chunk in search_results:
            try:
                # Compute severity
                severity_score = self.scorer.calculate_severity(chunk)
                
                # Compute novelty (compare with historical filings)
                historical_chunks = self._get_historical_chunks(ticker, filing_year)
                novelty_score = self.scorer.calculate_novelty(chunk, historical_chunks)
                
                # Build risk dictionary
                risk_dict = {
                    "text": chunk["text"],
                    "source_citation": chunk["text"][:500],  # Truncate for readability
                    "severity": {
                        "value": severity_score.value,
                        "explanation": severity_score.explanation
                    },
                    "novelty": {
                        "value": novelty_score.value,
                        "explanation": novelty_score.explanation
                    },
                    "metadata": chunk["metadata"]
                }
                
                # Optional: Classify via DB-only or LLM depending on flags
                if self.db_only_classification:
                    # DB-only mode: never call the LLM, do a best-effort vector lookup
                    if not chunk.get("text") or len(chunk.get("text", "").strip()) < 20:
                        logger.warning("Skipping DB-only classification: chunk text too short or empty")
                        db_classification = {
                            'category': 'UNCATEGORIZED',
                            'confidence': 0.0,
                            'method': 'skipped',
                            'llm_result': None
                        }
                    else:
                        logger.info("Classifying risk using DB-only mode (no LLM call)")
                        db_classification = self._classify_risk_db_only(chunk=chunk)

                    risk_dict["category"] = db_classification["category"]
                    risk_dict["category_confidence"] = db_classification["confidence"]
                    risk_dict["classification_method"] = db_classification["method"]

                elif self.use_llm:
                    # LLM mode: call LLM and persist result as before
                    if not chunk.get("text") or len(chunk.get("text", "").strip()) < 50:
                        logger.warning("Skipping LLM classification: chunk text too short or empty")
                        llm_classification = {
                            'category': 'UNCATEGORIZED',
                            'confidence': 0.0,
                            'method': 'skipped',
                            'llm_result': None
                        }
                    else:
                        logger.info(f"Classifying risk with LLM...")
                        llm_classification = self._classify_risk_with_llm(
                            chunk=chunk,
                            vector_metadata=chunk.get("metadata")
                        )
                    risk_dict["category"] = llm_classification["category"]
                    risk_dict["category_confidence"] = llm_classification["confidence"]
                    risk_dict["classification_method"] = llm_classification["method"]
                    if llm_classification["llm_result"]:
                        risk_dict["llm_evidence"] = llm_classification["llm_result"]["evidence"]
                        risk_dict["llm_rationale"] = llm_classification["llm_result"]["rationale"]
                
                risks.append(risk_dict)
                
            except Exception as e:
                logger.warning(f"Failed to score chunk: {e}")
                continue
        
        # Step 5: Build result
        elapsed_ms = (time.time() - start_time) * 1000
        
        result = RiskAnalysisResult(
            ticker=ticker,
            filing_year=filing_year,
            risks=risks,
            metadata={
                "chunks_indexed": index_stats["chunks_indexed"],
                "risks_analyzed": len(risks),
                "total_latency_ms": elapsed_ms,
                "index_latency_ms": index_stats.get("embedding_latency_ms", 0),
                "retrieve_top_k": retrieve_top_k,
                "rerank": rerank
            }
        )
        
        logger.info(f"Analysis complete: {len(risks)} risks in {elapsed_ms:.2f}ms")
        return result
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _classify_risk_with_llm(
        self,
        chunk: Dict[str, Any],
        vector_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify risk with LLM (bypassing threshold logic).
        
        When use_llm=True, this calls the Gemini classifier directly
        to ensure LLM classification regardless of vector confidence.
        
        Args:
            chunk: Risk chunk dictionary with 'text' and 'metadata'
            vector_metadata: Optional metadata from vector search (for future use)
        
        Returns:
            Dictionary with 'category', 'confidence', 'method', 'llm_result'
        """
        try:
            # Call Gemini directly (bypasses vector threshold logic)
            result = self.gemini_classifier.classify(text=chunk["text"])

            # Persist the LLM classification to SQLite + ChromaDB
            try:
                # Generate embedding for the chunk
                embedding = self.indexing_pipeline.embeddings.encode([chunk["text"]])[0].tolist()
                record_id, chroma_id = self.drift_system.insert_classification(
                    text=chunk["text"],
                    embedding=embedding,
                    llm_result=result,
                    source=ClassificationSource.LLM,
                )
                logger.info("Classification embedded into vector storage")
            except Exception:
                logger.exception("Failed to persist LLM classification to vector storage")

            return {
                'category': result.category.value,
                'confidence': result.confidence,
                'method': 'llm',
                'llm_result': {
                    'evidence': result.evidence,
                    'rationale': result.rationale,
                    'model_version': result.model_version
                }
            }
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return {
                'category': 'UNCATEGORIZED',
                'confidence': 0.0,
                'method': 'error',
                'llm_result': None
            }

    def _classify_risk_db_only(
        self,
        chunk: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Best-effort classification using only existing database/vector-store entries.

        Returns a dict with 'category', 'confidence', 'method', 'llm_result'. No LLM
        call is made from this method.
        """
        try:
            # Generate embedding for the chunk
            embedding = self.indexing_pipeline.embeddings.encode([chunk["text"]])[0].tolist()

            # Query the DriftDetectionSystem for similar cached classifications
            results = self.drift_system.similarity_search(query_embedding=embedding, n_results=1)

            if not results:
                return {
                    'category': 'UNCATEGORIZED',
                    'confidence': 0.0,
                    'method': 'db_only_no_match',
                    'llm_result': None
                }

            top = results[0]
            similarity = top.get('similarity_score', 0.0)

            if similarity >= self.db_only_similarity_threshold:
                return {
                    'category': top.get('category', 'UNCATEGORIZED'),
                    'confidence': float(top.get('confidence', 0.0)),
                    'method': 'vector_db',
                    'llm_result': None
                }
            else:
                return {
                    'category': 'UNCATEGORIZED',
                    'confidence': float(top.get('confidence', 0.0)),
                    'method': 'db_only_low_similarity',
                    'llm_result': None
                }
        except Exception as e:
            logger.exception(f"DB-only classification failed: {e}")
            return {
                'category': 'UNCATEGORIZED',
                'confidence': 0.0,
                'method': 'error',
                'llm_result': None
            }
    
    def _validate_inputs(
        self,
        html_path: str,
        ticker: str,
        filing_year: int
    ) -> None:
        """
        Validate input parameters.
        
        Raises:
            IntegrationError: If validation fails
        """
        # Validate ticker
        if not ticker or not ticker.strip():
            raise IntegrationError("Ticker cannot be empty")
        
        # Allow alphanumeric + dot/hyphen (e.g., BRK.B, ABC-D)
        if not re.match(r'^[A-Z0-9.\-]+$', ticker):
            raise IntegrationError(
                f"Invalid ticker format: {ticker}. "
                "Must contain only uppercase letters, numbers, dots, and hyphens."
            )
        
        # Validate year
        if filing_year < 1990 or filing_year > 2030:
            raise IntegrationError(
                f"Invalid filing year: {filing_year}. Must be between 1990-2030."
            )
        
        # Validate HTML path exists
        html_file = Path(html_path)
        if not html_file.exists():
            raise IntegrationError(f"HTML file not found at path: {html_path}")
        
        if not html_file.is_file():
            raise IntegrationError(f"Path is not a file: {html_path}")
    
    def _get_historical_chunks(
        self,
        ticker: str,
        current_year: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical chunks for novelty comparison.
        
        Searches for chunks from prior years (up to 3 years back).
        
        Args:
            ticker: Stock symbol
            current_year: Current filing year
        
        Returns:
            List of historical chunks (empty if no history exists)
        """
        historical_chunks = []
        
        # Look back up to 3 years
        for year in range(current_year - 3, current_year):
            if year < 1990:
                continue
            
            try:
                # Search for historical risks from this year
                results = self.indexing_pipeline.semantic_search(
                    query="risk factors",
                    n_results=20,  # Get more historical context
                    where={"ticker": ticker, "filing_year": year},
                    rerank=True
                )
                historical_chunks.extend(results)
            except Exception as e:
                logger.debug(f"No historical data found for {ticker} {year}: {e}")
                continue
        
        logger.info(f"Found {len(historical_chunks)} historical chunks for {ticker}")
        return historical_chunks
