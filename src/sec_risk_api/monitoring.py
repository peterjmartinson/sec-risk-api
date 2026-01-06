"""
Monitoring and logging infrastructure (Issue #4.3).

Provides structured logging, metrics collection, health checks, and error tracking
for production deployment with financial-grade SLAs.

Features:
- JSON structured logging for log aggregation
- Prometheus-compatible metrics
- Health and readiness probes
- Error tracking and alerting
- Performance monitoring
- Graceful shutdown handling

Usage:
    from sec_risk_api.monitoring import log_api_request, MetricsCollector
    
    # Log API request
    log_api_request(
        endpoint="/analyze",
        method="POST",
        status_code=200,
        latency_ms=1234.5,
        user="client_123"
    )
    
    # Collect metrics
    metrics = MetricsCollector()
    metrics.increment('api_requests_total', labels={'endpoint': '/analyze'})
"""

import logging
import json
import time
import signal
import psutil
import os
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager
import threading

from sec_risk_api.config import get_config, Config

# Setup logger
logger = logging.getLogger(__name__)


# ============================================================================
# Structured Logging
# ============================================================================

class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    
    Outputs logs in JSON format for ingestion by log aggregators
    (e.g., Elasticsearch, CloudWatch, Datadog).
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add custom fields from extra
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'user'):
            log_data['user'] = record.user
        if hasattr(record, 'latency_ms'):
            log_data['latency_ms'] = record.latency_ms
        
        return json.dumps(log_data)


def setup_logging(log_level: str = 'INFO') -> None:
    """
    Configure structured logging.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # Add JSON handler
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    root_logger.addHandler(handler)


def log_api_request(
    endpoint: str,
    method: str,
    status_code: int,
    latency_ms: float,
    user: Optional[str] = None,
    request_id: Optional[str] = None
) -> None:
    """
    Log API request with structured data.
    
    Args:
        endpoint: API endpoint path
        method: HTTP method
        status_code: Response status code
        latency_ms: Request latency in milliseconds
        user: User identifier
        request_id: Unique request ID
    """
    logger.info(
        f"API request: {method} {endpoint} -> {status_code} ({latency_ms:.2f}ms)",
        extra={
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'latency_ms': latency_ms,
            'user': user,
            'request_id': request_id
        }
    )


def log_llm_usage(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_cost: float,
    latency_ms: float
) -> None:
    """
    Log LLM API usage for cost tracking.
    
    Args:
        model: LLM model name
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens
        total_cost: Estimated cost in USD
        latency_ms: API call latency
    """
    logger.info(
        f"LLM usage: {model} - {prompt_tokens} prompt + {completion_tokens} completion tokens",
        extra={
            'model': model,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': prompt_tokens + completion_tokens,
            'cost_usd': total_cost,
            'latency_ms': latency_ms
        }
    )


def log_error(
    error: Exception,
    endpoint: Optional[str] = None,
    user: Optional[str] = None,
    request_id: Optional[str] = None
) -> None:
    """
    Log error with full context.
    
    Args:
        error: Exception that occurred
        endpoint: API endpoint where error occurred
        user: User identifier
        request_id: Request ID for correlation
    """
    logger.error(
        f"Error: {str(error)}",
        exc_info=True,
        extra={
            'error_type': type(error).__name__,
            'endpoint': endpoint,
            'user': user,
            'request_id': request_id
        }
    )


def log_critical_error(error: Exception, severity: str = 'high') -> None:
    """
    Log critical error and trigger alert.
    
    Args:
        error: Critical exception
        severity: Alert severity (low, medium, high, critical)
    """
    logger.critical(
        f"CRITICAL: {str(error)}",
        exc_info=True,
        extra={'severity': severity}
    )
    
    # Trigger alert
    send_alert(
        message=f"Critical error: {str(error)}",
        severity=severity,
        error_type=type(error).__name__
    )


def send_alert(message: str, severity: str, **kwargs: Any) -> None:
    """
    Send alert to monitoring system.
    
    In production, integrate with PagerDuty, Slack, email, etc.
    For now, log at CRITICAL level.
    
    Args:
        message: Alert message
        severity: Alert severity
        **kwargs: Additional context
    """
    logger.critical(
        f"ALERT [{severity}]: {message}",
        extra={**kwargs, 'alert': True}
    )


def log_db_query(
    operation: str,
    latency_ms: float,
    result_count: int,
    collection: str
) -> None:
    """
    Log database query performance.
    
    Args:
        operation: Query operation (search, insert, update)
        latency_ms: Query latency
        result_count: Number of results
        collection: Collection/table name
    """
    logger.info(
        f"DB query: {operation} on {collection} - {result_count} results ({latency_ms:.2f}ms)",
        extra={
            'operation': operation,
            'latency_ms': latency_ms,
            'result_count': result_count,
            'collection': collection
        }
    )


# ============================================================================
# Metrics Collection
# ============================================================================

class MetricsCollector:
    """
    Simple in-memory metrics collector.
    
    For production, integrate with Prometheus, StatsD, or CloudWatch.
    This implementation provides the interface for testing.
    """
    
    def __init__(self) -> None:
        """Initialize metrics storage."""
        self._counters: Dict[str, int] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def increment(
        self,
        metric_name: str,
        value: int = 1,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a counter metric.
        
        Args:
            metric_name: Metric name
            value: Increment value
            labels: Metric labels (e.g., {'endpoint': '/analyze'})
        """
        with self._lock:
            key = self._make_key(metric_name, labels)
            self._counters[key] = self._counters.get(key, 0) + value
    
    def observe(
        self,
        metric_name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a histogram observation.
        
        Args:
            metric_name: Metric name
            value: Observed value
            labels: Metric labels
        """
        with self._lock:
            key = self._make_key(metric_name, labels)
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)
    
    def get_metric(
        self,
        metric_name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> int:
        """
        Get counter value.
        
        Args:
            metric_name: Metric name
            labels: Metric labels. If None, sums all labeled counters with this name.
                    If provided, sums all counters where labels are a subset match.
        
        Returns:
            Counter value
        """
        if labels is None:
            # Sum all counters with this metric name (any labels)
            total = 0
            for key, value in self._counters.items():
                if key.startswith(metric_name):
                    total += value
            return total
        
        # Sum all counters where provided labels match (subset match)
        total = 0
        for key, value in self._counters.items():
            if not key.startswith(metric_name):
                continue
            
            # Check if all provided labels match in the key
            match = all(f"{k}={v}" in key for k, v in labels.items())
            if match:
                total += value
        
        return total
    
    def get_histogram_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Get histogram statistics.
        
        Args:
            metric_name: Metric name
        
        Returns:
            Dictionary with p50, p95, p99 percentiles
        """
        with self._lock:
            values = []
            for key, hist_values in self._histograms.items():
                if key.startswith(metric_name):
                    values.extend(hist_values)
            
            if not values:
                return {'p50': 0, 'p95': 0, 'p99': 0}
            
            values.sort()
            return {
                'p50': self._percentile(values, 50),
                'p95': self._percentile(values, 95),
                'p99': self._percentile(values, 99)
            }
    
    def _make_key(self, metric_name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create metric key from name and labels."""
        if not labels:
            return metric_name
        label_str = ','.join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{metric_name}{{{label_str}}}"
    
    def _percentile(self, values: List[float], p: int) -> float:
        """Calculate percentile."""
        if not values:
            return 0.0
        k = (len(values) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(values):
            return values[-1]
        return values[f] + (k - f) * (values[c] - values[f])


# Global metrics collector
_metrics_collector = MetricsCollector()


@contextmanager
def track_operation(operation_name: str, labels: Optional[Dict[str, str]] = None):
    """
    Context manager to track operation latency.
    
    Usage:
        with track_operation('embedding_generation'):
            generate_embeddings(text)
    
    Args:
        operation_name: Operation name
        labels: Additional labels
    """
    start = time.time()
    try:
        yield
    finally:
        latency_seconds = time.time() - start
        _metrics_collector.observe(
            f'{operation_name}_duration_seconds',
            latency_seconds,
            labels=labels
        )


# ============================================================================
# Health Checks
# ============================================================================

def check_redis_connection() -> bool:
    """
    Check Redis connection.
    
    Returns:
        True if Redis is reachable, False otherwise
    """
    try:
        import redis
        config = get_config()
        client = redis.from_url(config.redis_url, socket_connect_timeout=2)
        client.ping()
        return True
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        return False


def check_chroma_connection() -> bool:
    """
    Check ChromaDB connection.
    
    Returns:
        True if Chroma is accessible, False otherwise
    """
    try:
        import chromadb
        config = get_config()
        client = chromadb.PersistentClient(path=config.chroma_persist_path)
        # Try to list collections
        client.list_collections()
        return True
    except Exception as e:
        logger.warning(f"ChromaDB health check failed: {e}")
        return False


def check_error_rate_threshold(
    total_requests: int,
    error_requests: int,
    threshold: float = 0.05
) -> None:
    """
    Check if error rate exceeds threshold and alert.
    
    Args:
        total_requests: Total request count
        error_requests: Failed request count
        threshold: Error rate threshold (0.0-1.0)
    """
    if total_requests == 0:
        return
    
    error_rate = error_requests / total_requests
    
    if error_rate > threshold:
        send_alert(
            message=f"Error rate {error_rate:.2%} exceeds threshold {threshold:.2%}",
            severity='high',
            total_requests=total_requests,
            error_requests=error_requests,
            error_rate=error_rate
        )


def categorize_exception(error: Exception) -> str:
    """
    Categorize exception type for monitoring.
    
    Args:
        error: Exception instance
    
    Returns:
        Category string (client_error, server_error, dependency_error)
    """
    if isinstance(error, (ValueError, TypeError, KeyError)):
        return 'client_error'
    elif isinstance(error, (ConnectionError, TimeoutError)):
        return 'dependency_error'
    elif isinstance(error, (MemoryError, OSError)):
        return 'server_error'
    else:
        return 'unknown_error'


# ============================================================================
# Graceful Shutdown
# ============================================================================

class GracefulShutdown:
    """
    Handles graceful shutdown for zero-downtime deployments.
    
    Tracks active requests and ensures they complete before shutdown.
    """
    
    def __init__(self, timeout: int = 30) -> None:
        """
        Initialize shutdown handler.
        
        Args:
            timeout: Max seconds to wait for requests to complete
        """
        self.timeout = timeout
        self._shutting_down = False
        self._active_requests: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def register_active_request(self, request_id: str) -> None:
        """Register an active request."""
        with self._lock:
            self._active_requests[request_id] = time.time()
    
    def complete_request(self, request_id: str) -> None:
        """Mark request as complete."""
        with self._lock:
            self._active_requests.pop(request_id, None)
    
    def initiate_shutdown(self) -> None:
        """Initiate graceful shutdown."""
        logger.info("Initiating graceful shutdown...")
        self._shutting_down = True
    
    def is_shutting_down(self) -> bool:
        """Check if shutdown is in progress."""
        return self._shutting_down
    
    def active_request_count(self) -> int:
        """Get count of active requests."""
        return len(self._active_requests)
    
    def can_shutdown(self) -> bool:
        """Check if safe to shutdown (no active requests)."""
        return self.active_request_count() == 0
    
    def wait_for_completion(self) -> bool:
        """
        Wait for active requests to complete.
        
        Returns:
            True if all requests completed, False if timeout
        """
        start = time.time()
        while not self.can_shutdown():
            if time.time() - start > self.timeout:
                logger.warning(
                    f"Shutdown timeout: {self.active_request_count()} requests still active"
                )
                return False
            time.sleep(0.5)
        return True


# Global shutdown handler
_shutdown_handler = GracefulShutdown()


def setup_signal_handlers(shutdown_callback: Callable[[], None]) -> None:
    """
    Setup signal handlers for graceful shutdown.
    
    Args:
        shutdown_callback: Function to call on SIGTERM/SIGINT
    """
    def signal_handler(signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}")
        _shutdown_handler.initiate_shutdown()
        shutdown_callback()
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


# ============================================================================
# Performance Monitoring
# ============================================================================

def get_memory_stats() -> Dict[str, float]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory metrics in MB
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident set size
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual memory size
        'peak_mb': getattr(memory_info, 'peak_wset', memory_info.rss) / 1024 / 1024
    }


# Initialize logging on module import
setup_logging(get_config().log_level)
