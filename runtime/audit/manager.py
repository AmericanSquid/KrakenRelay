# runtime/audit/manager.py

import json
import logging
import queue
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from .events import AuditEvent, Severity

log = logging.getLogger(__name__)


class AuditManager:
    def __init__(
        self,
        db_path="data/audit.db",
        retention_days=30,
        commit_interval=1.0,
        queue_size=5000,
    ):
        self.db_path = Path(db_path)

        self.retention_days = retention_days
        self.commit_interval = commit_interval

        self.queue = queue.Queue(maxsize=queue_size)

        self.running = False
        self.worker = None

        self.conn = None

        self.session_id = uuid.uuid4().hex[:8]

    # ============================================================
    # Lifecycle
    # ============================================================

    def start(self):
        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
        )

        self.conn.execute("PRAGMA journal_mode=WAL;")

        self.conn.execute("PRAGMA synchronous=NORMAL;")

        self._create_tables()

        self.running = True

        self.worker = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="AuditWriter",
        )

        self.worker.start()

        self.cleanup_old_entries()

        self.log_event(
            event_type=AuditEvent.SYSTEM_START,
            severity=Severity.INFO,
            source="audit",
            message="Audit system started",
            metadata={
                "session_id": self.session_id,
            },
        )

        log.info(
            "Audit manager started (session=%s)",
            self.session_id,
        )

    def stop(self):
        self.log_event(
            event_type=AuditEvent.SYSTEM_STOP,
            severity=Severity.INFO,
            source="audit",
            message="Audit system stopping",
        )

        self.running = False

        if self.worker:
            self.worker.join(timeout=5)

        try:
            if self.conn:
                self.conn.commit()
                self.conn.close()

        except Exception:
            log.exception("Failed to close audit database")

    # ============================================================
    # Database Setup
    # ============================================================

    def _create_tables(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,

                severity TEXT NOT NULL,
                event_type TEXT NOT NULL,

                source TEXT,
                username TEXT,

                message TEXT,

                metadata_json TEXT
            )
            """
        )

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_audit_timestamp
            ON audit_log(timestamp)
            """
        )

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_audit_event_type
            ON audit_log(event_type)
            """
        )

        self.conn.execute(
            """
            CREATE INDEX IF NOT EXISTS
            idx_audit_session
            ON audit_log(session_id)
            """
        )

        self.conn.commit()

    # ============================================================
    # Public Logging API
    # ============================================================

    def log_event(
        self,
        *,
        event_type,
        severity,
        message,
        source=None,
        username=None,
        metadata=None,
    ):
        if not self.running:
            return

        if not isinstance(event_type, AuditEvent):
            raise TypeError("event_type must be AuditEvent")

        if not isinstance(severity, Severity):
            raise TypeError("severity must be Severity")

        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": self.session_id,
            "severity": severity.value,
            "event_type": event_type.value,
            "source": source,
            "username": username,
            "message": message,
            "metadata_json": json.dumps(metadata or {}),
        }

        try:
            self.queue.put_nowait(event)

        except queue.Full:
            log.error("Audit queue overflow")

            self._write_internal_event(
                event_type=AuditEvent.AUDIT_QUEUE_OVERFLOW,
                severity=Severity.ERROR,
                source="audit",
                message="Audit queue overflow",
            )

    # ============================================================
    # Worker Thread
    # ============================================================

    def _worker_loop(self):
        last_commit = time.time()

        while self.running or not self.queue.empty():
            try:
                event = self.queue.get(timeout=0.5)

                self.conn.execute(
                    """
                    INSERT INTO audit_log (
                        timestamp,
                        session_id,
                        severity,
                        event_type,
                        source,
                        username,
                        message,
                        metadata_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event["timestamp"],
                        event["session_id"],
                        event["severity"],
                        event["event_type"],
                        event["source"],
                        event["username"],
                        event["message"],
                        event["metadata_json"],
                    ),
                )

            except queue.Empty:
                pass

            except Exception:
                log.exception("Audit writer failure")

                try:
                    self._write_internal_event(
                        event_type=AuditEvent.AUDIT_WRITE_FAILURE,
                        severity=Severity.ERROR,
                        source="audit",
                        message="Audit writer failure",
                    )

                except Exception:
                    pass

            now = time.time()

            if now - last_commit >= self.commit_interval:
                try:
                    self.conn.commit()

                except Exception:
                    log.exception("Audit commit failure")

                last_commit = now

        try:
            self.conn.commit()

        except Exception:
            log.exception("Final audit commit failure")

    # ============================================================
    # Internal Direct Write
    # ============================================================

    def _write_internal_event(
        self,
        *,
        event_type,
        severity,
        message,
        source=None,
        metadata=None,
    ):
        try:
            self.conn.execute(
                """
                INSERT INTO audit_log (
                    timestamp,
                    session_id,
                    severity,
                    event_type,
                    source,
                    username,
                    message,
                    metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    self.session_id,
                    severity.value,
                    event_type.value,
                    source,
                    None,
                    message,
                    json.dumps(metadata or {}),
                ),
            )

            self.conn.commit()

        except Exception:
            log.exception("Failed to write internal audit event")

    # ============================================================
    # Retention
    # ============================================================

    def cleanup_old_entries(self):
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)

        try:
            cursor = self.conn.execute(
                """
                DELETE FROM audit_log
                WHERE timestamp < ?
                """,
                (cutoff.isoformat(),),
            )

            deleted = cursor.rowcount

            self.conn.commit()

            log.info(
                "Audit retention cleanup removed %s entries",
                deleted,
            )

        except Exception:
            log.exception("Audit retention cleanup failed")

    # ============================================================
    # Convenience Helpers
    # ============================================================

    def info(
        self,
        event_type,
        message,
        **kwargs,
    ):
        self.log_event(
            event_type=event_type,
            severity=Severity.INFO,
            message=message,
            **kwargs,
        )

    def warning(
        self,
        event_type,
        message,
        **kwargs,
    ):
        self.log_event(
            event_type=event_type,
            severity=Severity.WARNING,
            message=message,
            **kwargs,
        )

    def error(
        self,
        event_type,
        message,
        **kwargs,
    ):
        self.log_event(
            event_type=event_type,
            severity=Severity.ERROR,
            message=message,
            **kwargs,
        )

    def critical(
        self,
        event_type,
        message,
        **kwargs,
    ):
        self.log_event(
            event_type=event_type,
            severity=Severity.CRITICAL,
            message=message,
            **kwargs,
        )
