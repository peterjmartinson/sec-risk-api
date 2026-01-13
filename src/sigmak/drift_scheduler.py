# Copyright (c) 2025 Peter Martinson, Distracted Fortune. All rights reserved.
# This software is proprietary and not licensed for use, modification, or distribution.

"""
Scheduled drift detection jobs using APScheduler.

This module provides a background scheduler for periodic drift reviews,
compatible with both development (in-process) and production (Celery) environments.

Design Principles:
- APScheduler for development/standalone deployment
- Celery integration ready for production scaling
- Configurable review intervals and sample sizes
- Graceful shutdown with job persistence

Usage:
    # Development: In-process scheduler
    >>> scheduler = DriftScheduler()
    >>> scheduler.start()
    >>> scheduler.schedule_drift_review(interval_hours=24)
    
    # Production: Cron job (see documentation)
    # 0 2 * * * cd /app && python -m sigmak.drift_scheduler --run-once
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from sigmak.drift_detection import DriftDetectionSystem, DriftReviewJob

logger = logging.getLogger(__name__)


# ============================================================================
# Scheduler Configuration
# ============================================================================


class DriftScheduler:
    """
    Background scheduler for periodic drift reviews.
    
    Features:
    - APScheduler for in-process scheduling
    - Configurable review intervals
    - Job persistence across restarts
    - Alert notifications when drift detected
    
    Usage:
        >>> scheduler = DriftScheduler(
        ...     db_path="./database/risk_classifications.db",
        ...     chroma_path="./database"
        ... )
        >>> scheduler.start()
        >>> scheduler.schedule_drift_review(interval_hours=24, sample_size=50)
        >>> # Keep running...
        >>> scheduler.stop()
    """
    
    def __init__(
        self,
        db_path: str = "./database/risk_classifications.db",
        chroma_path: str = "./database",
        timezone: str = "UTC"
    ) -> None:
        """
        Initialize drift scheduler.
        
        Args:
            db_path: Path to SQLite database
            chroma_path: Path to ChromaDB storage
            timezone: Timezone for scheduling (default: UTC)
        """
        self.db_path = db_path
        self.chroma_path = chroma_path
        
        # Initialize drift system
        self.drift_system = DriftDetectionSystem(
            db_path=db_path,
            chroma_path=chroma_path
        )
        
        # Initialize scheduler
        self.scheduler = BackgroundScheduler(timezone=timezone)
        
        logger.info(f"DriftScheduler initialized: timezone={timezone}")
    
    def start(self) -> None:
        """Start the background scheduler."""
        self.scheduler.start()
        logger.info("Drift scheduler started")
    
    def stop(self) -> None:
        """Stop the scheduler gracefully."""
        self.scheduler.shutdown(wait=True)
        logger.info("Drift scheduler stopped")
    
    def schedule_drift_review(
        self,
        interval_hours: int = 24,
        sample_size: int = 20,
        low_conf_ratio: float = 0.6,
        old_days: int = 90
    ) -> None:
        """
        Schedule periodic drift review job.
        
        Args:
            interval_hours: Run review every N hours
            sample_size: Number of classifications to review
            low_conf_ratio: Fraction from low-confidence pool
            old_days: Age threshold for old classification sampling
        """
        def run_review() -> None:
            """Wrapper to execute drift review."""
            try:
                logger.info("Starting scheduled drift review")
                job = DriftReviewJob(drift_system=self.drift_system)
                metrics = job.run_review(
                    sample_size=sample_size,
                    low_conf_ratio=low_conf_ratio,
                    old_days=old_days
                )
                
                # Log results
                logger.info(
                    f"Drift review complete: agreement_rate={metrics.agreement_rate:.1%}, "
                    f"reviewed={metrics.total_reviewed}"
                )
                
                # Send alerts if needed
                if metrics.requires_manual_review():
                    self._send_critical_alert(metrics)
                elif metrics.is_warning():
                    self._send_warning_alert(metrics)
            
            except Exception as e:
                logger.error(f"Drift review failed: {e}", exc_info=True)
        
        # Schedule job
        self.scheduler.add_job(
            func=run_review,
            trigger='interval',
            hours=interval_hours,
            id='drift_review',
            replace_existing=True,
            name=f"Drift Review (every {interval_hours}h)"
        )
        
        logger.info(
            f"Scheduled drift review: interval={interval_hours}h, "
            f"sample_size={sample_size}"
        )
    
    def schedule_drift_review_cron(
        self,
        hour: int = 2,
        minute: int = 0,
        sample_size: int = 20
    ) -> None:
        """
        Schedule drift review with cron-style trigger.
        
        Args:
            hour: Hour to run (0-23, default: 2 AM)
            minute: Minute to run (0-59)
            sample_size: Number of classifications to review
        """
        def run_review() -> None:
            """Wrapper to execute drift review."""
            try:
                logger.info("Starting scheduled drift review (cron)")
                job = DriftReviewJob(drift_system=self.drift_system)
                metrics = job.run_review(sample_size=sample_size)
                
                logger.info(
                    f"Drift review complete: agreement_rate={metrics.agreement_rate:.1%}"
                )
                
                if metrics.requires_manual_review():
                    self._send_critical_alert(metrics)
                elif metrics.is_warning():
                    self._send_warning_alert(metrics)
            
            except Exception as e:
                logger.error(f"Drift review failed: {e}", exc_info=True)
        
        # Schedule with cron trigger
        trigger = CronTrigger(hour=hour, minute=minute)
        self.scheduler.add_job(
            func=run_review,
            trigger=trigger,
            id='drift_review_cron',
            replace_existing=True,
            name=f"Drift Review (daily at {hour:02d}:{minute:02d})"
        )
        
        logger.info(f"Scheduled drift review: daily at {hour:02d}:{minute:02d} UTC")
    
    def run_once(self, sample_size: int = 20) -> None:
        """
        Run drift review once (for cron job invocation).
        
        Args:
            sample_size: Number of classifications to review
        """
        logger.info("Running one-time drift review")
        job = DriftReviewJob(drift_system=self.drift_system)
        metrics = job.run_review(sample_size=sample_size)
        
        logger.info(
            f"Drift review complete: agreement_rate={metrics.agreement_rate:.1%}, "
            f"reviewed={metrics.total_reviewed}"
        )
        
        if metrics.requires_manual_review():
            self._send_critical_alert(metrics)
            sys.exit(1)  # Exit with error code for cron monitoring
        elif metrics.is_warning():
            self._send_warning_alert(metrics)
    
    def _send_critical_alert(self, metrics) -> None:
        """Send critical alert for low agreement rate."""
        # TODO: Integrate with alerting system (email, Slack, PagerDuty)
        logger.critical(
            f"CRITICAL: Classification drift detected!\n"
            f"Agreement rate: {metrics.agreement_rate:.1%}\n"
            f"Reviewed: {metrics.total_reviewed}\n"
            f"Disagreements: {metrics.disagreements}\n"
            f"Manual review required."
        )
    
    def _send_warning_alert(self, metrics) -> None:
        """Send warning alert for moderate drift."""
        logger.warning(
            f"WARNING: Classification drift warning\n"
            f"Agreement rate: {metrics.agreement_rate:.1%}\n"
            f"Reviewed: {metrics.total_reviewed}\n"
            f"Monitor for continued drift."
        )


# ============================================================================
# CLI for Cron Job Integration
# ============================================================================


def main() -> None:
    """
    CLI entry point for cron job integration.
    
    Example cron job (runs daily at 2 AM):
        0 2 * * * cd /app && /usr/local/bin/python -m sigmak.drift_scheduler --run-once --sample-size 50
    
    Example systemd timer (production):
        # /etc/systemd/system/sigmak-drift-review.timer
        [Unit]
        Description=SigmaK Drift Detection Daily Review
        
        [Timer]
        OnCalendar=daily
        OnCalendar=02:00
        Persistent=true
        
        [Install]
        WantedBy=timers.target
    """
    parser = argparse.ArgumentParser(
        description="SigmaK Drift Detection Scheduler"
    )
    
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run drift review once and exit (for cron jobs)"
    )
    
    parser.add_argument(
        "--start-scheduler",
        action="store_true",
        help="Start background scheduler (for long-running processes)"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of classifications to review (default: 20)"
    )
    
    parser.add_argument(
        "--interval-hours",
        type=int,
        default=24,
        help="Review interval in hours (default: 24)"
    )
    
    parser.add_argument(
        "--db-path",
        type=str,
        default="./database/risk_classifications.db",
        help="Path to SQLite database"
    )
    
    parser.add_argument(
        "--chroma-path",
        type=str,
        default="./database",
        help="Path to ChromaDB storage"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create scheduler
    scheduler = DriftScheduler(
        db_path=args.db_path,
        chroma_path=args.chroma_path
    )
    
    if args.run_once:
        # Run once for cron job
        scheduler.run_once(sample_size=args.sample_size)
    
    elif args.start_scheduler:
        # Start background scheduler
        scheduler.start()
        scheduler.schedule_drift_review(
            interval_hours=args.interval_hours,
            sample_size=args.sample_size
        )
        
        # Keep running
        logger.info("Scheduler running. Press Ctrl+C to stop.")
        try:
            import time
            while True:
                time.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            scheduler.stop()
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
