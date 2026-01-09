# Quick Start: Async Task Queue

## Prerequisites

```bash
# Install Redis (if not already installed)
# Ubuntu/Debian:
sudo apt-get install redis-server

# macOS:
brew install redis

# Or use Docker:
docker run -d -p 6379:6379 redis:latest
```

## Starting the System

Open 3 separate terminals:

### Terminal 1: Start Redis

```bash
redis-server
# Redis will run on localhost:6379
```

### Terminal 2: Start Celery Worker

```bash
cd /home/peter/Code/sigmak
celery -A sigmak.tasks worker --loglevel=info
```

Expected output:
```
 -------------- celery@hostname v5.6.2 (...)
--- ***** ----- 
-- ******* ---- [tasks]
  . sigmak.analyze_filing
  . sigmak.index_filing
  . sigmak.batch_analyze
```

### Terminal 3: Start API Server

```bash
cd /home/peter/Code/sigmak
uv run uvicorn sigmak.api:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

## Testing the Async API

### 1. Create an API Key

```python
from sigmak.auth import APIKeyManager

manager = APIKeyManager()
api_key = manager.create_api_key(user="test_user", rate_limit="10/minute")
print(f"Your API Key: {api_key}")
```

### 2. Submit an Analysis Task

```bash
# Replace YOUR_API_KEY with the key from step 1
curl -X POST "http://localhost:8000/analyze" \
  -H "X-API-Key: YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "filing_year": 2025,
    "html_path": "data/sample_10k.html",
    "retrieve_top_k": 5
  }'
```

**Expected Response (< 100ms):**
```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status_url": "/tasks/a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "message": "Analysis task submitted successfully"
}
```

### 3. Poll Task Status

```bash
# Replace TASK_ID with the task_id from step 2
curl -X GET "http://localhost:8000/tasks/TASK_ID" \
  -H "X-API-Key: YOUR_API_KEY"
```

**Response Progression:**

**PENDING** (task queued):
```json
{
  "task_id": "a1b2c3d4-...",
  "status": "PENDING",
  "progress": null,
  "result": null,
  "error": null
}
```

**PROGRESS** (task running):
```json
{
  "task_id": "a1b2c3d4-...",
  "status": "PROGRESS",
  "progress": {
    "current": 3,
    "total": 5,
    "status": "Computing severity scores..."
  },
  "result": null,
  "error": null
}
```

**SUCCESS** (task complete):
```json
{
  "task_id": "a1b2c3d4-...",
  "status": "SUCCESS",
  "progress": null,
  "result": {
    "ticker": "AAPL",
    "filing_year": 2025,
    "risks": [
      {
        "text": "Supply chain disruptions...",
        "severity": {"value": 0.75, "explanation": "..."},
        "novelty": {"value": 0.82, "explanation": "..."}
      }
    ],
    "metadata": {
      "total_latency_ms": 5234.5,
      "chunks_indexed": 5
    }
  },
  "error": null
}
```

## Python Client Example

```python
import requests
import time

API_KEY = "your-api-key-here"
BASE_URL = "http://localhost:8000"

# Submit task
response = requests.post(
    f"{BASE_URL}/analyze",
    headers={"X-API-Key": API_KEY},
    json={
        "ticker": "AAPL",
        "filing_year": 2025,
        "html_path": "data/sample_10k.html",
        "retrieve_top_k": 10
    }
)

task_id = response.json()["task_id"]
print(f"Task submitted: {task_id}")

# Poll for results
while True:
    status_response = requests.get(
        f"{BASE_URL}/tasks/{task_id}",
        headers={"X-API-Key": API_KEY}
    )
    
    data = status_response.json()
    status = data["status"]
    
    if status == "PENDING":
        print("Task queued, waiting to start...")
    
    elif status == "PROGRESS":
        progress = data["progress"]
        print(f"Progress: {progress['current']}/{progress['total']} - {progress['status']}")
    
    elif status == "SUCCESS":
        result = data["result"]
        print(f"\n✅ Analysis complete!")
        print(f"Ticker: {result['ticker']}")
        print(f"Risks found: {len(result['risks'])}")
        
        for i, risk in enumerate(result['risks'], 1):
            print(f"\nRisk {i}:")
            print(f"  Severity: {risk['severity']['value']:.2f}")
            print(f"  Novelty: {risk['novelty']['value']:.2f}")
            print(f"  Text: {risk['text'][:100]}...")
        
        break
    
    elif status == "FAILURE":
        print(f"\n❌ Task failed: {data['error']}")
        break
    
    time.sleep(2)  # Poll every 2 seconds
```

## Monitoring

### Check Celery Workers

```bash
celery -A sigmak.tasks inspect active
celery -A sigmak.tasks inspect stats
```

### Check Redis Queue

```bash
redis-cli
> KEYS *
> GET celery-task-meta-TASK_ID
```

### API Documentation

Open in browser:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Troubleshooting

**Problem: "Task queue service unavailable"**
- Solution: Ensure Redis is running (`redis-server`)

**Problem: Tasks stay in PENDING forever**
- Solution: Ensure Celery worker is running

**Problem: "401 Unauthorized"**
- Solution: Check API key is valid and included in X-API-Key header

**Problem: Worker crashes**
- Tasks are automatically recovered due to acks_late=True
- Restart worker: `celery -A sigmak.tasks worker --loglevel=info`

## Performance Tips

**Multiple Workers:**
```bash
# Run 4 parallel workers
celery -A sigmak.tasks worker --concurrency=4 --loglevel=info
```

**Monitor with Flower:**
```bash
pip install flower
celery -A sigmak.tasks flower
# Open http://localhost:5555
```

**Adjust Result Expiration:**
```python
# In tasks.py, modify:
celery_app.conf.update(
    result_expires=7200,  # 2 hours instead of 1
)
```

## Production Deployment

```bash
# Use systemd or supervisor to manage processes
# Example systemd service for Celery:

# /etc/systemd/system/celery-sec-risk.service
[Unit]
Description=Celery Worker for SigmaK API
After=network.target redis.service

[Service]
Type=forking
User=www-data
Group=www-data
WorkingDirectory=/opt/sigmak
ExecStart=/opt/sigmak/.venv/bin/celery -A sigmak.tasks worker \
          --concurrency=8 --loglevel=info --pidfile=/var/run/celery.pid

[Install]
WantedBy=multi-user.target
```

---

For more details, see:
- [JOURNAL.md](JOURNAL.md) - Technical implementation details
- [README.md](README.md) - Full API documentation
- [ASYNC_TASK_QUEUE_SUMMARY.md](ASYNC_TASK_QUEUE_SUMMARY.md) - Implementation summary
