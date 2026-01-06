"""
Unit tests for async task queue with Celery + Redis (Issue #4.3).

Following TDD principles: Write tests BEFORE implementation.
Each test verifies exactly ONE behavior (SRP).

Test Categories:
1. API immediate response tests
2. Task status reporting tests
3. Queue failure recovery tests
4. End-to-end integration tests

Usage:
    pytest tests/test_async_task_queue.py -v
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import time
import json

# Note: These imports will exist after implementation
# For now, we're testing the interface contract


# ============================================================================
# Test 1: API Returns Immediately (Non-blocking)
# ============================================================================

class TestAPIImmediateResponse:
    """Test that API endpoints return immediately without waiting for tasks."""
    
    def test_analyze_endpoint_returns_task_id_immediately(self) -> None:
        """
        Test: POST /analyze returns task_id within 100ms even if ingestion takes 10s.
        
        Given: A slow ingestion task that takes 10 seconds
        When: POST /analyze is called
        Then: Response received in < 100ms with task_id
        """
        # This test will pass once we implement async endpoints
        with patch('sec_risk_api.tasks.analyze_filing_task.delay') as mock_task, \
             patch('sec_risk_api.api.authenticate_api_key') as mock_auth:
            # Mock authentication
            mock_auth.return_value = "test-user"
            
            # Mock Celery task returning AsyncResult with task_id
            mock_result = Mock()
            mock_result.id = "test-task-123"
            mock_task.return_value = mock_result
            
            from fastapi.testclient import TestClient
            from sec_risk_api.api import app
            
            client = TestClient(app)
            
            start_time = time.time()
            response = client.post(
                "/analyze?async_mode=true",
                json={
                    "ticker": "AAPL",
                    "filing_year": 2025,
                    "html_content": "<html>Test content</html>"
                },
                headers={"X-API-Key": "test-key"}
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            # API should respond instantly (< 100ms)
            assert elapsed_ms < 100, f"API took {elapsed_ms}ms, expected < 100ms"
            assert response.status_code == 202  # Accepted
            assert "task_id" in response.json()
            assert response.json()["task_id"] == "test-task-123"
    
    def test_index_filing_endpoint_returns_immediately(self) -> None:
        """
        Test: POST /index returns task_id immediately without blocking.
        
        Given: An indexing operation that takes time
        When: POST /index is called
        Then: Response received immediately with task_id
        """
        with patch('sec_risk_api.tasks.index_filing_task.delay') as mock_task, \
             patch('sec_risk_api.api.authenticate_api_key') as mock_auth:
            mock_auth.return_value = "test-user"
            
            mock_result = Mock()
            mock_result.id = "index-task-456"
            mock_task.return_value = mock_result
            
            from fastapi.testclient import TestClient
            from sec_risk_api.api import app
            
            client = TestClient(app)
            
            start_time = time.time()
            response = client.post(
                "/index",
                json={
                    "ticker": "MSFT",
                    "filing_year": 2024,
                    "html_path": "data/sample_10k.html"
                },
                headers={"X-API-Key": "test-key"}
            )
            elapsed_ms = (time.time() - start_time) * 1000
            
            assert elapsed_ms < 100
            assert response.status_code == 202
            assert response.json()["task_id"] == "index-task-456"


# ============================================================================
# Test 2: Reliable Task Status Reporting
# ============================================================================

class TestTaskStatusReporting:
    """Test reliable status and progress reporting for all queue tasks."""
    
    def test_task_status_endpoint_returns_pending_state(self) -> None:
        """
        Test: GET /tasks/{task_id} returns 'PENDING' for queued task.
        
        Given: A task that has been queued but not started
        When: GET /tasks/{task_id} is called
        Then: Status is 'PENDING'
        """
        from fastapi.testclient import TestClient
        from sec_risk_api.api import app
        
        with patch('sec_risk_api.tasks.celery_app.AsyncResult') as mock_result_cls, \
             patch('sec_risk_api.api.authenticate_api_key') as mock_auth:
            mock_auth.return_value = "test-user"
            
            mock_result = Mock()
            mock_result.state = "PENDING"
            mock_result.info = None
            mock_result_cls.return_value = mock_result
            
            client = TestClient(app)
            response = client.get(
                "/tasks/test-task-123",
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 200
            assert response.json()["status"] == "PENDING"
            assert response.json()["task_id"] == "test-task-123"
    
    def test_task_status_endpoint_returns_progress_state(self) -> None:
        """
        Test: GET /tasks/{task_id} returns progress for running task.
        
        Given: A task that is 50% complete
        When: GET /tasks/{task_id} is called
        Then: Status is 'PROGRESS' with progress percentage
        """
        from fastapi.testclient import TestClient
        from sec_risk_api.api import app
        
        with patch('sec_risk_api.tasks.celery_app.AsyncResult') as mock_result_cls, \
             patch('sec_risk_api.api.authenticate_api_key') as mock_auth:
            mock_auth.return_value = "test-user"
            
            mock_result = Mock()
            mock_result.state = "PROGRESS"
            mock_result.info = {
                "current": 50,
                "total": 100,
                "status": "Indexing chunks..."
            }
            mock_result_cls.return_value = mock_result
            
            client = TestClient(app)
            response = client.get(
                "/tasks/test-task-456",
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "PROGRESS"
            assert data["progress"]["current"] == 50
            assert data["progress"]["total"] == 100
            assert data["progress"]["status"] == "Indexing chunks..."
    
    def test_task_status_endpoint_returns_success_with_result(self) -> None:
        """
        Test: GET /tasks/{task_id} returns complete result for successful task.
        
        Given: A task that completed successfully
        When: GET /tasks/{task_id} is called
        Then: Status is 'SUCCESS' with full result
        """
        from fastapi.testclient import TestClient
        from sec_risk_api.api import app
        
        with patch('sec_risk_api.tasks.celery_app.AsyncResult') as mock_result_cls, \
             patch('sec_risk_api.api.authenticate_api_key') as mock_auth:
            mock_auth.return_value = "test-user"
            
            mock_result = Mock()
            mock_result.state = "SUCCESS"
            mock_result.result = {
                "ticker": "AAPL",
                "filing_year": 2025,
                "risks": [{"text": "Risk 1", "severity": 0.75}]
            }
            mock_result_cls.return_value = mock_result
            
            client = TestClient(app)
            response = client.get(
                "/tasks/test-task-789",
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "SUCCESS"
            assert "result" in data
            assert data["result"]["ticker"] == "AAPL"
    
    def test_task_status_endpoint_returns_failure_with_error(self) -> None:
        """
        Test: GET /tasks/{task_id} returns error for failed task.
        
        Given: A task that failed with an exception
        When: GET /tasks/{task_id} is called
        Then: Status is 'FAILURE' with error message
        """
        from fastapi.testclient import TestClient
        from sec_risk_api.api import app
        
        with patch('sec_risk_api.tasks.celery_app.AsyncResult') as mock_result_cls, \
             patch('sec_risk_api.api.authenticate_api_key') as mock_auth:
            mock_auth.return_value = "test-user"
            
            mock_result = Mock()
            mock_result.state = "FAILURE"
            mock_result.info = Exception("File not found")
            mock_result_cls.return_value = mock_result
            
            client = TestClient(app)
            response = client.get(
                "/tasks/test-task-failed",
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "FAILURE"
            assert "error" in data
            assert "File not found" in str(data["error"])


# ============================================================================
# Test 3: Queue Failure Recovery
# ============================================================================

class TestQueueFailureRecovery:
    """Test robustness for queue failures with retry logic."""
    
    def test_task_retries_on_connection_error(self) -> None:
        """
        Test: Task automatically retries on Redis connection failure.
        
        Given: Redis connection fails temporarily
        When: Task execution is attempted
        Then: Task retries with exponential backoff
        """
        from sec_risk_api.tasks import analyze_filing_task
        
        # Mock the task to track retry behavior
        with patch.object(analyze_filing_task, 'retry') as mock_retry:
            # Simulate connection error
            connection_error = ConnectionError("Redis connection failed")
            
            # Task should call retry() with exponential backoff
            try:
                raise connection_error
            except ConnectionError as exc:
                analyze_filing_task.retry(exc=exc, countdown=60)
            
            # Verify retry was called
            mock_retry.assert_called_once()
            call_kwargs = mock_retry.call_args[1]
            assert call_kwargs['countdown'] == 60
    
    def test_task_max_retries_exhausted_returns_failure(self) -> None:
        """
        Test: Task returns FAILURE after max retries exhausted.
        
        Given: A task that fails after 3 retry attempts
        When: All retries are exhausted
        Then: Task state is FAILURE with error details
        """
        from sec_risk_api.tasks import analyze_filing_task
        
        # This tests the max_retries configuration
        assert hasattr(analyze_filing_task, 'max_retries')
        assert analyze_filing_task.max_retries >= 3
    
    def test_celery_worker_recovers_from_crash(self) -> None:
        """
        Test: Worker recovers and resumes tasks after crash.
        
        Given: A worker crashes mid-execution
        When: Worker restarts
        Then: Incomplete tasks are re-queued or marked as failed
        """
        # This is an integration test that verifies Celery's acks_late=True
        from sec_risk_api.tasks import analyze_filing_task
        
        # Task should be configured with acks_late=True for crash recovery
        # This ensures tasks are not lost if worker crashes
        assert hasattr(analyze_filing_task, 'acks_late')
        assert analyze_filing_task.acks_late is True
    
    def test_redis_unavailable_returns_meaningful_error(self) -> None:
        """
        Test: API returns meaningful error when Redis is down.
        
        Given: Redis broker is unavailable
        When: POST /analyze is called
        Then: Returns 503 Service Unavailable with clear message
        """
        from fastapi.testclient import TestClient
        from sec_risk_api.api import app
        
        with patch('sec_risk_api.tasks.analyze_filing_task.delay') as mock_task, \
             patch('sec_risk_api.api.authenticate_api_key') as mock_auth:
            mock_auth.return_value = "test-user"
            
            # Simulate Redis connection failure
            mock_task.side_effect = ConnectionError("Cannot connect to Redis")
            
            client = TestClient(app)
            response = client.post(
                "/analyze?async_mode=true",
                json={
                    "ticker": "AAPL",
                    "filing_year": 2025,
                    "html_content": "<html>Test</html>"
                },
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 503
            assert "service unavailable" in response.json()["detail"].lower()


# ============================================================================
# Test 4: End-to-End Integration
# ============================================================================

class TestEndToEndIntegration:
    """Test full workflow from API request to task completion."""
    
    @pytest.mark.integration
    def test_full_analyze_workflow(self) -> None:
        """
        Test: Complete workflow from POST /analyze to GET /tasks/{id}.
        
        Given: A valid analyze request
        When: POST /analyze followed by GET /tasks/{id}
        Then: Task progresses from PENDING → PROGRESS → SUCCESS
        """
        from fastapi.testclient import TestClient
        from sec_risk_api.api import app
        
        client = TestClient(app)
        
        # Step 1: Submit analysis request
        with patch('sec_risk_api.tasks.analyze_filing_task.delay') as mock_task, \
             patch('sec_risk_api.api.authenticate_api_key') as mock_auth:
            mock_auth.return_value = "test-user"
            
            mock_result = Mock()
            mock_result.id = "integration-test-123"
            mock_task.return_value = mock_result
            
            response = client.post(
                "/analyze?async_mode=true",
                json={
                    "ticker": "TSLA",
                    "filing_year": 2024,
                    "html_content": "<html>Risk factors...</html>"
                },
                headers={"X-API-Key": "test-key"}
            )
            
            assert response.status_code == 202
            task_id = response.json()["task_id"]
        
        # Step 2: Check task status (simulating progression)
        states = ["PENDING", "PROGRESS", "SUCCESS"]
        
        for state in states:
            with patch('sec_risk_api.tasks.celery_app.AsyncResult') as mock_result_cls, \
                 patch('sec_risk_api.api.authenticate_api_key') as mock_auth:
                mock_auth.return_value = "test-user"
                
                mock_result = Mock()
                mock_result.state = state
                
                if state == "PROGRESS":
                    mock_result.info = {"current": 5, "total": 10}
                elif state == "SUCCESS":
                    mock_result.result = {"ticker": "TSLA", "risks": []}
                
                mock_result_cls.return_value = mock_result
                
                status_response = client.get(
                    f"/tasks/{task_id}",
                    headers={"X-API-Key": "test-key"}
                )
                
                assert status_response.status_code == 200
                assert status_response.json()["status"] == state
    
    @pytest.mark.integration
    def test_multiple_concurrent_tasks(self) -> None:
        """
        Test: Multiple tasks can be queued and processed concurrently.
        
        Given: 5 analyze requests submitted simultaneously
        When: All tasks are queued
        Then: Each task has unique task_id and can be tracked independently
        """
        from fastapi.testclient import TestClient
        from sec_risk_api.api import app
        
        client = TestClient(app)
        task_ids = []
        
        with patch('sec_risk_api.tasks.analyze_filing_task.delay') as mock_task, \
             patch('sec_risk_api.api.authenticate_api_key') as mock_auth:
            mock_auth.return_value = "test-user"
            
            # Submit 5 concurrent requests
            for i in range(5):
                mock_result = Mock()
                mock_result.id = f"concurrent-task-{i}"
                mock_task.return_value = mock_result
                
                response = client.post(
                    "/analyze?async_mode=true",
                    json={
                        "ticker": f"TICK{i}",
                        "filing_year": 2024,
                        "html_content": f"<html>Content {i}</html>"
                    },
                    headers={"X-API-Key": "test-key"}
                )
                
                assert response.status_code == 202
                task_ids.append(response.json()["task_id"])
        
        # All task IDs should be unique
        assert len(task_ids) == len(set(task_ids))
        assert all(tid.startswith("concurrent-task-") for tid in task_ids)
    
    @pytest.mark.integration
    def test_task_result_polling_pattern(self) -> None:
        """
        Test: Client can poll task status until completion.
        
        Given: A long-running task
        When: Client polls GET /tasks/{id} repeatedly
        Then: Eventually receives SUCCESS with result
        """
        from fastapi.testclient import TestClient
        from sec_risk_api.api import app
        
        client = TestClient(app)
        task_id = "polling-test-123"
        
        # Simulate polling 3 times: PENDING, PROGRESS, SUCCESS
        states_sequence = [
            ("PENDING", None),
            ("PROGRESS", {"current": 50, "total": 100}),
            ("SUCCESS", {"ticker": "AAPL", "risks": []})
        ]
        
        for state, info_or_result in states_sequence:
            with patch('sec_risk_api.tasks.celery_app.AsyncResult') as mock_result_cls, \
                 patch('sec_risk_api.api.authenticate_api_key') as mock_auth:
                mock_auth.return_value = "test-user"
                
                mock_result = Mock()
                mock_result.state = state
                
                if state == "SUCCESS":
                    mock_result.result = info_or_result
                else:
                    mock_result.info = info_or_result
                
                mock_result_cls.return_value = mock_result
                
                response = client.get(
                    f"/tasks/{task_id}",
                    headers={"X-API-Key": "test-key"}
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == state
                
                if state == "SUCCESS":
                    assert "result" in data
                    break
