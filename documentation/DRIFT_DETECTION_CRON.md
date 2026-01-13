# Drift Detection Cron Job Setup

This document provides production-ready cron job and systemd timer configurations for automated drift detection.

## Quick Start: Cron Job

### 1. Edit Crontab

```bash
crontab -e
```

### 2. Add Daily Drift Review (2 AM UTC)

```cron
# SigmaK Drift Detection - Daily Review
0 2 * * * cd /app && /usr/local/bin/python -m sigmak.drift_scheduler --run-once --sample-size 50 >> /var/log/sigmak/drift.log 2>&1
```

**Parameters**:
- `--run-once`: Execute one drift review and exit
- `--sample-size 50`: Review 50 classifications (30 low-confidence, 20 old records)
- `>> /var/log/sigmak/drift.log`: Append logs to file
- `2>&1`: Redirect stderr to stdout

### 3. Verify Cron Job

```bash
# List cron jobs
crontab -l

# Monitor cron execution
tail -f /var/log/cron

# View drift detection logs
tail -f /var/log/sigmak/drift.log
```

## Systemd Timer (Recommended for Production)

Systemd timers provide better logging, monitoring, and reliability than cron.

### 1. Create Timer Unit

**File**: `/etc/systemd/system/sigmak-drift-review.timer`

```ini
[Unit]
Description=SigmaK Drift Detection Daily Review
Requires=sigmak-drift-review.service

[Timer]
# Run daily at 2:00 AM UTC
OnCalendar=daily
OnCalendar=02:00

# Ensure timer fires even if missed (e.g., system was off)
Persistent=true

# Add random delay (0-30 minutes) to avoid system load spikes
RandomizedDelaySec=30min

[Install]
WantedBy=timers.target
```

### 2. Create Service Unit

**File**: `/etc/systemd/system/sigmak-drift-review.service`

```ini
[Unit]
Description=SigmaK Drift Detection Review Job
After=network.target

[Service]
Type=oneshot
User=sigmak
Group=sigmak
WorkingDirectory=/app

# Set environment variables
Environment="PYTHONPATH=/app/src"
Environment="CHROMA_PERSIST_PATH=/app/database"

# Execute drift detection
ExecStart=/usr/local/bin/python -m sigmak.drift_scheduler \
    --run-once \
    --sample-size 50 \
    --db-path /app/database/risk_classifications.db \
    --chroma-path /app/database

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=sigmak-drift

# Resource limits
MemoryMax=2G
CPUQuota=50%

# Restart on failure
Restart=on-failure
RestartSec=5m

[Install]
WantedBy=multi-user.target
```

### 3. Enable and Start Timer

```bash
# Reload systemd configuration
sudo systemctl daemon-reload

# Enable timer (start on boot)
sudo systemctl enable sigmak-drift-review.timer

# Start timer immediately
sudo systemctl start sigmak-drift-review.timer

# Verify timer is active
sudo systemctl status sigmak-drift-review.timer
```

### 4. Monitor Systemd Timer

```bash
# List all timers
systemctl list-timers

# View timer logs
sudo journalctl -u sigmak-drift-review.timer -f

# View service logs
sudo journalctl -u sigmak-drift-review.service -f

# View last 100 lines
sudo journalctl -u sigmak-drift-review.service -n 100

# View logs from last 24 hours
sudo journalctl -u sigmak-drift-review.service --since "24 hours ago"
```

### 5. Manual Execution

```bash
# Run service manually (for testing)
sudo systemctl start sigmak-drift-review.service

# View real-time execution
sudo journalctl -u sigmak-drift-review.service -f
```

## Monitoring and Alerts

### Exit Codes

The drift detection script uses exit codes for monitoring:

- **0**: Success (agreement rate ≥ 75%)
- **1**: Critical drift detected (agreement rate < 75%)

### Email Alerts (cron)

```cron
# Send email on failure
MAILTO=ops@example.com
0 2 * * * cd /app && /usr/local/bin/python -m sigmak.drift_scheduler --run-once --sample-size 50
```

### Slack Webhook Integration

Add to `drift_scheduler.py`:

```python
import requests

def _send_critical_alert(self, metrics) -> None:
    """Send critical alert via Slack."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return
    
    message = {
        "text": f"🚨 *CRITICAL: Classification Drift Detected*",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Agreement Rate*: {metrics.agreement_rate:.1%}\n"
                        f"*Reviewed*: {metrics.total_reviewed}\n"
                        f"*Disagreements*: {metrics.disagreements}\n"
                        f"*Manual Review Required*"
                    )
                }
            }
        ]
    }
    
    requests.post(webhook_url, json=message)
```

### PagerDuty Integration

```python
def _send_critical_alert(self, metrics) -> None:
    """Send critical alert via PagerDuty."""
    api_key = os.getenv("PAGERDUTY_API_KEY")
    if not api_key:
        return
    
    event = {
        "routing_key": api_key,
        "event_action": "trigger",
        "payload": {
            "summary": f"Classification drift: {metrics.agreement_rate:.1%} agreement",
            "severity": "critical",
            "source": "sigmak-drift-detection",
            "custom_details": {
                "agreement_rate": metrics.agreement_rate,
                "total_reviewed": metrics.total_reviewed,
                "disagreements": metrics.disagreements
            }
        }
    }
    
    requests.post(
        "https://events.pagerduty.com/v2/enqueue",
        json=event
    )
```

## Logging Configuration

### Structured JSON Logging

Create `/app/config/logging.yaml`:

```yaml
version: 1
disable_existing_loggers: false

formatters:
  json:
    class: pythonjsonlogger.jsonlogger.JsonFormatter
    format: "%(asctime)s %(name)s %(levelname)s %(message)s"

handlers:
  console:
    class: logging.StreamHandler
    formatter: json
    stream: ext://sys.stdout
  
  file:
    class: logging.handlers.RotatingFileHandler
    formatter: json
    filename: /var/log/sigmak/drift.json
    maxBytes: 10485760  # 10MB
    backupCount: 10

loggers:
  sigmak.drift_detection:
    level: INFO
    handlers: [console, file]
    propagate: false
  
  sigmak.drift_scheduler:
    level: INFO
    handlers: [console, file]
    propagate: false

root:
  level: WARNING
  handlers: [console]
```

Load in script:

```python
import logging.config
import yaml

with open('/app/config/logging.yaml') as f:
    config = yaml.safe_load(f)
    logging.config.dictConfig(config)
```

## Performance Tuning

### Sample Size Recommendations

| Database Size | Sample Size | Frequency | Duration |
|---------------|-------------|-----------|----------|
| < 1,000       | 20          | Daily     | < 30s    |
| 1,000-10,000  | 50          | Daily     | 1-2 min  |
| 10,000-100,000| 100         | Daily     | 5-10 min |
| > 100,000     | 200         | Weekly    | 15-30 min|

### Resource Limits

Systemd service limits:
- **MemoryMax**: 2GB (sufficient for 200 sample size)
- **CPUQuota**: 50% (1 core = 100%, limits CPU usage)
- **TimeoutStartSec**: 600 (10 minutes max execution time)

### Database Optimization

Add to service unit:

```ini
[Service]
# Ensure database is not locked
ExecStartPre=/usr/bin/sqlite3 /app/database/risk_classifications.db "PRAGMA optimize;"

# Vacuum database weekly (separate timer)
# Create sigmak-db-vacuum.timer and .service
```

## Troubleshooting

### Timer Not Firing

```bash
# Check timer status
sudo systemctl status sigmak-drift-review.timer

# Check if timer is enabled
systemctl is-enabled sigmak-drift-review.timer

# View next scheduled execution
systemctl list-timers --all
```

### Service Failing

```bash
# View recent failures
sudo journalctl -u sigmak-drift-review.service --since today

# Run service manually with verbose logging
sudo -u sigmak /usr/local/bin/python -m sigmak.drift_scheduler --run-once --sample-size 5

# Check database permissions
ls -la /app/database/
```

### High Memory Usage

```bash
# Monitor memory during execution
watch -n 1 'ps aux | grep drift_scheduler'

# Check systemd memory limits
systemctl show sigmak-drift-review.service | grep Memory
```

## Best Practices

1. **Start Small**: Begin with sample_size=10-20 to verify system works
2. **Gradual Scale**: Increase sample size as database grows
3. **Monitor Alerts**: Set up Slack/PagerDuty integration immediately
4. **Log Retention**: Keep at least 30 days of drift metrics for trend analysis
5. **Database Backups**: Backup before major embedding model changes
6. **Testing**: Run manual drift review before enabling automated schedule

## Production Checklist

- [ ] Systemd timer configured and enabled
- [ ] Service unit with resource limits
- [ ] Logging configured (JSON format recommended)
- [ ] Alert integration (Slack/PagerDuty)
- [ ] Monitoring dashboard setup
- [ ] Database backup strategy
- [ ] Log rotation configured
- [ ] Sample size tuned for database size
- [ ] Manual test execution successful
- [ ] Documentation shared with operations team
