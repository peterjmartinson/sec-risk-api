"""
Celery tasks for asynchronous background processing (Issue #4.3).

This module defines all background tasks for the SigmaK API:
- Filing ingestion and indexing
- Risk analysis and scoring
- Batch processing

Configuration:
- Broker: Redis (default: redis://localhost:6379/0)
- Backend: Redis (for result storage)
- Task acknowledgment: Late (acks_late=True for crash recovery)
- Max retries: 3 with exponential backoff

Usage:
    # Start Celery worker
    celery -A sigmak.tasks worker --loglevel=info
    
    # Submit task from code
    from sigmak.tasks import analyze_filing_task
    result = analyze_filing_task.delay(
        ticker="AAPL",
        filing_year=2025,
        html_content="<html>...</html>"
    )
    
    # Check task status
    print(result.state)  # PENDING, PROGRESS, SUCCESS, FAILURE
    print(result.result)  # Task result when complete
"""

from celery import Celery, Task
from celery.exceptions import Retry
from typing import Dict, Any, Optional
import logging
import os
import tempfile
from pathlib import Path

from sigmak.integration import IntegrationPipeline, IntegrationError
from sigmak.indexing_pipeline import IndexingPipeline

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Celery Configuration
# ============================================================================

# Read Redis URL from environment (default to localhost)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Celery app
celery_app = Celery(
    "sigmak",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,  # Critical for crash recovery
    worker_prefetch_multiplier=1,  # Process one task at a time for reliability
    result_expires=3600,  # Results expire after 1 hour
)

# ============================================================================
# Task Base Class
# ============================================================================

class CallbackTask(Task):
    """
    Base task class with progress tracking.
    
    This class provides a common interface for updating task progress
    via Celery's update_state mechanism.
    """
    
    def update_progress(
        self,
        current: int,
        total: int,
        status: str = "Processing..."
    ) -> None:
        """
        Update task progress for status endpoint.
        
        Args:
            current: Current progress count
            total: Total items to process
            status: Human-readable status message
        """
        self.update_state(
            state="PROGRESS",
            meta={
                "current": current,
                "total": total,
                "status": status
            }
        )


# ============================================================================
# Task 1: Analyze Filing (Full Pipeline)
# ============================================================================

@celery_app.task(
    bind=True,
    base=CallbackTask,
    name="sigmak.analyze_filing",
    max_retries=3,
    default_retry_delay=60,  # Retry after 60 seconds
    acks_late=True
)
def analyze_filing_task(
    self: CallbackTask,
    ticker: str,
    filing_year: int,
    html_content: Optional[str] = None,
    html_path: Optional[str] = None,
    retrieve_top_k: int = 10
) -> Dict[str, Any]:
    """
    Background task for complete risk analysis pipeline.
    
    This task executes the full IntegrationPipeline:
    1. Ingest HTML filing
    2. Extract Item 1A risk factors
    3. Chunk and embed text
    4. Index into vector database
    5. Retrieve risk chunks
    6. Compute severity and novelty scores
    
    Args:
        self: Task instance (bound)
        ticker: Stock ticker symbol
        filing_year: Filing year
        html_content: Raw HTML content (optional)
        html_path: Path to HTML file (optional)
        retrieve_top_k: Number of top risks to analyze
    
    Returns:
        Dictionary with analysis results
    
    Raises:
        IntegrationError: If pipeline fails
        Retry: If recoverable error occurs
    """
    try:
        logger.info(f"Starting analysis for {ticker} ({filing_year})")
        
        # Update progress: Starting
        self.update_progress(0, 5, "Initializing pipeline...")
        
        # Initialize pipeline
        pipeline = IntegrationPipeline()
        
        self.update_progress(1, 5, "Ingesting HTML filing...")
        
        # Handle html_content vs html_path
        temp_file: Optional[Path] = None
        try:
            if html_content:
                # Write content to temporary file
                temp_file = Path(tempfile.mktemp(suffix=".html"))
                temp_file.write_text(html_content)
                target_path = str(temp_file)
            elif html_path:
                target_path = html_path
            else:
                raise IntegrationError("Either html_content or html_path required")
            
            self.update_progress(2, 5, "Extracting and chunking risk factors...")
            
            # Run analysis
            result = pipeline.analyze_filing(
                html_path=target_path,
                ticker=ticker,
                filing_year=filing_year,
                retrieve_top_k=retrieve_top_k
            )
            
            self.update_progress(4, 5, "Computing severity and novelty scores...")
            
            # Convert to dictionary
            result_dict = result.to_dict()
            
            self.update_progress(5, 5, "Analysis complete")
            
            logger.info(f"Analysis complete for {ticker} ({filing_year})")
            return result_dict
        
        finally:
            # Cleanup temporary file
            if temp_file and temp_file.exists():
                temp_file.unlink()
    
    except IntegrationError as e:
        logger.error(f"Integration error for {ticker}: {e}")
        # Non-recoverable error, do not retry
        raise
    
    except (ConnectionError, TimeoutError) as e:
        # Recoverable errors: retry with exponential backoff
        logger.warning(f"Recoverable error for {ticker}: {e}. Retrying...")
        retry_count = self.request.retries
        countdown = 60 * (2 ** retry_count)  # Exponential backoff
        raise self.retry(exc=e, countdown=countdown)
    
    except Exception as e:
        logger.error(f"Unexpected error for {ticker}: {e}")
        raise


# ============================================================================
# Task 2: Index Filing Only
# ============================================================================

@celery_app.task(
    bind=True,
    base=CallbackTask,
    name="sigmak.index_filing",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True
)
def index_filing_task(
    self: CallbackTask,
    ticker: str,
    filing_year: int,
    html_content: Optional[str] = None,
    html_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Background task for indexing filing into vector database.
    
    This task only performs ingestion and indexing (no scoring):
    1. Ingest HTML filing
    2. Extract Item 1A risk factors
    3. Chunk and embed text
    4. Index into vector database
    
    Args:
        self: Task instance (bound)
        ticker: Stock ticker symbol
        filing_year: Filing year
        html_content: Raw HTML content (optional)
        html_path: Path to HTML file (optional)
    
    Returns:
        Dictionary with indexing metadata (chunks_indexed, doc_ids, etc.)
    
    Raises:
        Exception: If indexing fails
        Retry: If recoverable error occurs
    """
    try:
        logger.info(f"Starting indexing for {ticker} ({filing_year})")
        
        self.update_progress(0, 4, "Initializing indexing pipeline...")
        
        # Initialize indexing pipeline
        pipeline = IndexingPipeline()
        
        self.update_progress(1, 4, "Ingesting HTML filing...")
        
        # Handle html_content vs html_path
        temp_file: Optional[Path] = None
        try:
            if html_content:
                temp_file = Path(tempfile.mktemp(suffix=".html"))
                temp_file.write_text(html_content)
                target_path = str(temp_file)
            elif html_path:
                target_path = html_path
            else:
                raise ValueError("Either html_content or html_path required")
            
            self.update_progress(2, 4, "Extracting, chunking, and indexing...")
            
            # Run indexing
            metadata = pipeline.index_filing(
                html_path=target_path,
                ticker=ticker,
                filing_year=filing_year,
                item_type="risk_factors"
            )
            
            self.update_progress(4, 4, "Indexing complete")
            
            logger.info(
                f"Indexing complete for {ticker} ({filing_year}): "
                f"{metadata['chunks_indexed']} chunks indexed"
            )
            
            return {
                "ticker": ticker,
                "filing_year": filing_year,
                "metadata": metadata
            }
        
        finally:
            if temp_file and temp_file.exists():
                temp_file.unlink()
    
    except (ConnectionError, TimeoutError) as e:
        logger.warning(f"Recoverable error for {ticker}: {e}. Retrying...")
        retry_count = self.request.retries
        countdown = 60 * (2 ** retry_count)
        raise self.retry(exc=e, countdown=countdown)
    
    except Exception as e:
        logger.error(f"Indexing error for {ticker}: {e}")
        raise


# ============================================================================
# Task 3: Batch Processing
# ============================================================================

@celery_app.task(
    bind=True,
    base=CallbackTask,
    name="sigmak.batch_analyze",
    max_retries=3,
    default_retry_delay=60,
    acks_late=True
)
def batch_analyze_task(
    self: CallbackTask,
    filings: list[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Background task for batch analysis of multiple filings.
    
    Args:
        self: Task instance (bound)
        filings: List of filing specifications, each with:
            - ticker: str
            - filing_year: int
            - html_path: str (or html_content: str)
    
    Returns:
        Dictionary with batch results and metadata
    """
    try:
        logger.info(f"Starting batch analysis for {len(filings)} filings")
        
        results = []
        total = len(filings)
        
        for idx, filing in enumerate(filings):
            self.update_progress(
                idx,
                total,
                f"Processing {filing['ticker']} ({filing['filing_year']})..."
            )
            
            # Submit individual analyze task
            result = analyze_filing_task.apply_async(
                kwargs=filing
            )
            
            results.append({
                "ticker": filing["ticker"],
                "filing_year": filing["filing_year"],
                "task_id": result.id
            })
        
        self.update_progress(total, total, "Batch submission complete")
        
        return {
            "total_filings": total,
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Batch analysis error: {e}")
        raise
