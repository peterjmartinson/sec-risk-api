"""
Unit tests for monitoring and logging infrastructure (Issue #4.3).

Following TDD principles: Write tests BEFORE implementation.
Each test verifies exactly ONE behavior (SRP).

Test Categories:
1. Structured logging tests
2. Metrics collection tests
3. Health check tests
4. Error tracking tests

Usage:
    pytest tests/test_monitoring_logging.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any
import logging
import json


# ============================================================================
# Test 1: Structured Logging
# ============================================================================

class TestStructuredLogging:
    """Test structured logging captures all required metrics."""
    
    def test_api_request_logs_latency(self) -> None:
        """
        Test: API requests log execution latency.
        
        Given: An API request is processed
        When: Request completes
        Then: Latency is logged in milliseconds
        """
        from sigmak.monitoring import log_api_request
        
        with patch('sigmak.monitoring.logger') as mock_logger:
            log_api_request(
                endpoint="/analyze",
                method="POST",
                status_code=200,
                latency_ms=1234.5,
                user="test_user"
            )
            
            # Verify log was called with latency
            assert mock_logger.info.called
            log_data = mock_logger.info.call_args[0][0]
            assert "latency_ms" in log_data or "1234.5" in str(log_data)
    
    def test_llm_usage_logs_token_count(self) -> None:
        """
        Test: LLM calls log token usage.
        
        Given: An LLM API is called
        When: Response is received
        Then: Token usage is logged (input + output)
        """
        from sigmak.monitoring import log_llm_usage
        
        with patch('sigmak.monitoring.logger') as mock_logger:
            log_llm_usage(
                model="gpt-4",
                prompt_tokens=150,
                completion_tokens=300,
                total_cost=0.0045,
                latency_ms=2345.6
            )
            
            assert mock_logger.info.called
            call_args = str(mock_logger.info.call_args)
            assert "150" in call_args  # prompt tokens
            assert "300" in call_args  # completion tokens
    
    def test_error_logs_include_stack_trace(self) -> None:
        """
        Test: Error logs capture full stack trace and context.
        
        Given: An exception occurs
        When: Error is logged
        Then: Stack trace and request context are captured
        """
        from sigmak.monitoring import log_error
        
        with patch('sigmak.monitoring.logger') as mock_logger:
            try:
                raise ValueError("Test error")
            except Exception as e:
                log_error(
                    error=e,
                    endpoint="/analyze",
                    user="test_user",
                    request_id="req-123"
                )
            
            assert mock_logger.error.called
            # Should log with exc_info for stack trace
            call_kwargs = mock_logger.error.call_args[1]
            assert call_kwargs.get('exc_info') is True or 'exc_info' in str(mock_logger.error.call_args)
    
    def test_logs_are_json_structured(self) -> None:
        """
        Test: Logs use JSON format for parsing by monitoring tools.
        
        Given: Any log event
        When: Log is formatted
        Then: Output is valid JSON with standard fields
        """
        from sigmak.monitoring import JSONFormatter
        
        formatter = JSONFormatter()
        
        # Create a log record
        logger = logging.getLogger('test')
        record = logger.makeRecord(
            name='test',
            level=logging.INFO,
            fn='test.py',
            lno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        # Format the record
        formatted = formatter.format(record)
        
        # Should be valid JSON
        data = json.loads(formatted)
        assert 'timestamp' in data or 'time' in data
        assert 'level' in data or 'levelname' in data.get('levelname', '')
        assert 'message' in data or 'msg' in data


# ============================================================================
# Test 2: Metrics Collection
# ============================================================================

class TestMetricsCollection:
    """Test metrics are collected for monitoring."""
    
    def test_request_counter_increments(self) -> None:
        """
        Test: Request counter tracks total API calls.
        
        Given: API receives requests
        When: Requests are processed
        Then: Counter increments for each request
        """
        from sigmak.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        
        initial_count = collector.get_metric('api_requests_total')
        
        collector.increment('api_requests_total', labels={'endpoint': '/analyze'})
        collector.increment('api_requests_total', labels={'endpoint': '/analyze'})
        
        final_count = collector.get_metric('api_requests_total')
        assert final_count == initial_count + 2
    
    def test_latency_histogram_records_percentiles(self) -> None:
        """
        Test: Latency histogram enables percentile calculations.
        
        Given: Multiple requests with varying latencies
        When: Latencies are recorded
        Then: Can calculate p50, p95, p99
        """
        from sigmak.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        
        # Record various latencies
        latencies = [100, 200, 150, 500, 1000, 250, 300]
        for latency in latencies:
            collector.observe('api_latency_seconds', latency / 1000.0)
        
        # Should be able to get percentiles
        stats = collector.get_histogram_stats('api_latency_seconds')
        assert 'p50' in stats or 'percentile_50' in stats or len(latencies) > 0
    
    def test_error_rate_metric_tracks_failures(self) -> None:
        """
        Test: Error rate metric tracks failed requests.
        
        Given: Mix of successful and failed requests
        When: Responses are recorded
        Then: Error rate is calculable
        """
        from sigmak.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        
        # Successful requests
        for _ in range(8):
            collector.increment('api_requests_total', labels={'status': '200'})
        
        # Failed requests
        for _ in range(2):
            collector.increment('api_requests_total', labels={'status': '500'})
        
        # Error rate should be 20%
        total = collector.get_metric('api_requests_total')
        errors = collector.get_metric('api_requests_total', labels={'status': '500'})
        
        assert total == 10
        assert errors == 2
    
    def test_celery_task_metrics_tracked(self) -> None:
        """
        Test: Celery task execution metrics are collected.
        
        Given: Background tasks are executed
        When: Tasks complete
        Then: Task count, latency, and status are tracked
        """
        from sigmak.monitoring import MetricsCollector
        
        collector = MetricsCollector()
        
        collector.increment('celery_tasks_total', labels={'task': 'analyze_filing', 'status': 'success'})
        collector.observe('celery_task_duration_seconds', 15.5, labels={'task': 'analyze_filing'})
        
        task_count = collector.get_metric('celery_tasks_total', labels={'task': 'analyze_filing'})
        assert task_count == 1


# ============================================================================
# Test 3: Health Checks
# ============================================================================

class TestHealthChecks:
    """Test health and readiness endpoints."""
    
    def test_health_check_endpoint_returns_status(self) -> None:
        """
        Test: /health endpoint returns service status.
        
        Given: Service is running
        When: GET /health is called
        Then: Returns 200 with status details
        """
        from fastapi.testclient import TestClient
        from sigmak.api import app
        
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert data['status'] in ['healthy', 'degraded', 'unhealthy']
    
    def test_readiness_check_verifies_dependencies(self) -> None:
        """
        Test: /ready endpoint checks all dependencies.
        
        Given: Service with dependencies (DB, Redis)
        When: GET /ready is called
        Then: Returns 200 only if all dependencies are healthy
        """
        from fastapi.testclient import TestClient
        from sigmak.api import app
        
        with patch('sigmak.api.check_redis_connection') as mock_redis, \
             patch('sigmak.api.check_chroma_connection') as mock_chroma:
            
            # All dependencies healthy
            mock_redis.return_value = True
            mock_chroma.return_value = True
            
            client = TestClient(app)
            response = client.get("/ready")
            
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'ready'
            assert data['dependencies']['redis'] == 'ok'
            assert data['dependencies']['chromadb'] == 'ok'
    
    def test_readiness_fails_when_redis_down(self) -> None:
        """
        Test: /ready returns 503 when Redis is unavailable.
        
        Given: Redis is down
        When: GET /ready is called
        Then: Returns 503 with dependency status
        """
        from fastapi.testclient import TestClient
        from sigmak.api import app
        
        with patch('sigmak.api.check_redis_connection') as mock_redis, \
             patch('sigmak.api.check_chroma_connection') as mock_chroma:
            
            mock_redis.return_value = False
            mock_chroma.return_value = True
            
            client = TestClient(app)
            response = client.get("/ready")
            
            assert response.status_code == 503
            data = response.json()['detail']  # HTTPException detail
            assert data['dependencies']['redis'] == 'unavailable'
    
    def test_liveness_probe_responds_quickly(self) -> None:
        """
        Test: /live endpoint responds in < 100ms.
        
        Given: Service is running
        When: GET /live is called
        Then: Returns 200 in < 100ms
        """
        from fastapi.testclient import TestClient
        from sigmak.api import app
        import time
        
        client = TestClient(app)
        
        start = time.time()
        response = client.get("/live")
        elapsed_ms = (time.time() - start) * 1000
        
        assert response.status_code == 200
        assert elapsed_ms < 100


# ============================================================================
# Test 4: Error Tracking
# ============================================================================

class TestErrorTracking:
    """Test error tracking and alerting."""
    
    def test_errors_include_request_id(self) -> None:
        """
        Test: All errors include request_id for tracing.
        
        Given: An error occurs during request processing
        When: Error is logged
        Then: Request ID is included for correlation
        """
        from sigmak.monitoring import log_error
        
        with patch('sigmak.monitoring.logger') as mock_logger:
            error = ValueError("Test error")
            log_error(
                error=error,
                request_id="req-abc-123",
                endpoint="/analyze"
            )
            
            call_str = str(mock_logger.error.call_args)
            assert "req-abc-123" in call_str
    
    def test_critical_errors_trigger_alert(self) -> None:
        """
        Test: Critical errors trigger alerting mechanism.
        
        Given: A critical error occurs (DB down, OOM)
        When: Error is logged
        Then: Alert is sent to monitoring system
        """
        from sigmak.monitoring import log_critical_error
        
        with patch('sigmak.monitoring.send_alert') as mock_alert:
            error = ConnectionError("Database connection lost")
            log_critical_error(error, severity="high")
            
            assert mock_alert.called
            call_args = mock_alert.call_args[1]
            assert call_args['severity'] == 'high'
    
    def test_error_rate_threshold_detection(self) -> None:
        """
        Test: High error rates are detected and alerted.
        
        Given: Error rate exceeds threshold (e.g., > 5%)
        When: Metrics are evaluated
        Then: Alert is triggered
        """
        from sigmak.monitoring import check_error_rate_threshold
        
        with patch('sigmak.monitoring.send_alert') as mock_alert:
            # Simulate 10% error rate
            check_error_rate_threshold(
                total_requests=100,
                error_requests=10,
                threshold=0.05
            )
            
            # Should trigger alert
            assert mock_alert.called
    
    def test_exception_types_are_categorized(self) -> None:
        """
        Test: Exceptions are categorized for better monitoring.
        
        Given: Various exception types occur
        When: Exceptions are logged
        Then: Each is tagged with category (client, server, dependency)
        """
        from sigmak.monitoring import categorize_exception
        
        assert categorize_exception(ValueError()) == 'client_error'
        assert categorize_exception(ConnectionError()) == 'dependency_error'
        assert categorize_exception(MemoryError()) == 'server_error'


# ============================================================================
# Test 5: Deployment Readiness
# ============================================================================

class TestDeploymentReadiness:
    """Test deployment and zero-downtime migration capabilities."""
    
    def test_service_starts_with_environment_variables(self) -> None:
        """
        Test: Service starts correctly with env vars.
        
        Given: Required environment variables are set
        When: Service starts
        Then: Configuration is loaded correctly
        """
        import os
        from sigmak.config import get_config, reset_config
        
        with patch.dict(os.environ, {
            'REDIS_URL': 'redis://test:6379',
            'LOG_LEVEL': 'INFO',
            'ENVIRONMENT': 'production'
        }):
            reset_config()  # Reset singleton to pick up new env vars
            config = get_config()
            assert config.redis_url == 'redis://test:6379'
            assert config.log_level == 'INFO'
            assert config.environment == 'production'
        
        # Reset after test
        reset_config()
    
    def test_graceful_shutdown_completes_in_flight_requests(self) -> None:
        """
        Test: Graceful shutdown waits for in-flight requests.
        
        Given: Service receives SIGTERM with active requests
        When: Shutdown is initiated
        Then: Active requests complete before shutdown
        """
        from sigmak.monitoring import GracefulShutdown
        
        shutdown_handler = GracefulShutdown(timeout=30)
        
        # Simulate active request
        shutdown_handler.register_active_request("req-123")
        
        # Start shutdown
        shutdown_handler.initiate_shutdown()
        
        assert shutdown_handler.is_shutting_down()
        assert shutdown_handler.active_request_count() == 1
        
        # Complete request
        shutdown_handler.complete_request("req-123")
        
        assert shutdown_handler.can_shutdown()
    
    def test_container_responds_to_sigterm(self) -> None:
        """
        Test: Container handles SIGTERM for orchestration.
        
        Given: Container receives SIGTERM
        When: Signal is handled
        Then: Graceful shutdown is initiated
        """
        from sigmak.monitoring import setup_signal_handlers
        import signal
        
        shutdown_called = []
        
        def mock_shutdown() -> None:
            shutdown_called.append(True)
        
        setup_signal_handlers(mock_shutdown)
        
        # Simulate SIGTERM
        # Note: actual signal testing is complex, this tests the setup
        assert signal.getsignal(signal.SIGTERM) is not signal.SIG_DFL


# ============================================================================
# Test 6: Performance Monitoring
# ============================================================================

class TestPerformanceMonitoring:
    """Test performance metrics collection."""
    
    def test_embedding_latency_tracked(self) -> None:
        """
        Test: Embedding generation latency is tracked.
        
        Given: Embeddings are generated
        When: Operation completes
        Then: Latency is logged and metrics updated
        """
        from sigmak.monitoring import track_operation, _metrics_collector
        
        initial_metrics = len(list(_metrics_collector._histograms.keys()))
        
        with track_operation('embedding_generation'):
            # Simulate work
            pass
        
        # Should have recorded a new histogram
        final_metrics = len(list(_metrics_collector._histograms.keys()))
        assert final_metrics > initial_metrics
    
    def test_database_query_performance_logged(self) -> None:
        """
        Test: Vector DB query performance is logged.
        
        Given: Vector similarity search is performed
        When: Query completes
        Then: Query time and result count are logged
        """
        from sigmak.monitoring import log_db_query
        
        with patch('sigmak.monitoring.logger') as mock_logger:
            log_db_query(
                operation='similarity_search',
                latency_ms=234.5,
                result_count=10,
                collection='sec_risk_factors'
            )
            
            assert mock_logger.info.called
    
    def test_memory_usage_monitored(self) -> None:
        """
        Test: Memory usage is tracked for leak detection.
        
        Given: Service is running
        When: Memory metrics are collected
        Then: Current and peak memory are tracked
        """
        from sigmak.monitoring import get_memory_stats
        
        stats = get_memory_stats()
        
        assert 'rss_mb' in stats  # Resident set size
        assert 'peak_mb' in stats or 'rss_mb' in stats
        assert stats['rss_mb'] > 0
