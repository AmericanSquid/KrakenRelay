# audio/health.py

import logging
import os
import threading
import time

from runtime.audit import AuditEvent

log = logging.getLogger(__name__)


class AudioStreamFailure(RuntimeError):
    pass


class AudioHealthMonitor:
    def __init__(
        self,
        *,
        max_failures=3,
        reset_window=10.0,
        stale_timeout=5.0,
        watchdog_interval=1.0,
        tx_stuck_grace=3.0,
        audit=None,
    ):
        self.max_failures = max_failures
        self.reset_window = reset_window
        self.stale_timeout = stale_timeout
        self.watchdog_interval = watchdog_interval
        self.tx_stuck_grace = tx_stuck_grace
        self.audit = audit

        self.failures = 0
        self.last_failure_time = 0.0
        self.last_rx_activity = time.monotonic()

        self.tx_without_rx_since = None

        self._is_running = None
        self._is_squelch_open = None
        self._is_transmitting = None
        self._tail_time = None
        self._watchdog_thread = None
        self._watchdog_stop = threading.Event()

    def record_success(self):
        if self.failures == self.max_failures - 1:
            log.warning(
                "KRANKEN RELAY DONE DID IT AGAIN. RX almost died then came back to life."
            )

        self.failures = 0
        self.mark_rx_activity()

    def mark_rx_activity(self):
        self.last_rx_activity = time.monotonic()

    def record_failure(self, exc):
        now = time.monotonic()

        if now - self.last_failure_time > self.reset_window:
            self.failures = 0

        self.last_failure_time = now
        self.failures += 1

        log.warning(
            "[AudioHealth] RX stream failure %s/%s: %s",
            self.failures,
            self.max_failures,
            exc,
        )

        if self.failures >= self.max_failures:
            raise AudioStreamFailure(
                f"RX audio stream failed {self.failures} times"
            ) from exc

    def rx_stale(self):
        return time.monotonic() - self.last_rx_activity > self.stale_timeout

    def start_watchdog(
        self,
        is_running,
        is_squelch_open,
        is_transmitting,
        tail_time,
    ):
        self._is_running = is_running
        self._is_squelch_open = is_squelch_open
        self._is_transmitting = is_transmitting
        self._tail_time = tail_time

        if self._watchdog_thread and self._watchdog_thread.is_alive():
            return

        self._watchdog_stop.clear()

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="AudioHealthWatchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

        log.info(
            "[AudioHealth] Watchdog started: stale_timeout=%.1fs interval=%.1fs",
            self.stale_timeout,
            self.watchdog_interval,
        )

    def stop_watchdog(self, timeout=None):
        self._watchdog_stop.set()
        self._is_running = None
        self._is_squelch_open = None
        self._is_transmitting = None
        self._tail_time = None

        thread = self._watchdog_thread
        if not thread:
            return True

        if not thread.is_alive():
            self._watchdog_thread = None
            return True

        if thread is threading.current_thread():
            log.warning(
                "[AudioHealth] stop_watchdog called from watchdog thread; skipping self-join"
            )
            return True

        if timeout is None:
            timeout = self.watchdog_interval + 1.0

        log.info("[AudioHealth] Stopping watchdog thread")
        thread.join(timeout=timeout)

        if thread.is_alive():
            log.error("[AudioHealth] Watchdog thread still alive after %.1fs", timeout)
            return False

        self._watchdog_thread = None
        log.info("[AudioHealth] Watchdog stopped")
        return True

    def _watchdog_loop(self):
        try:
            while not self._watchdog_stop.is_set():
                time.sleep(self.watchdog_interval)

                is_running = self._is_running
                is_squelch_open = self._is_squelch_open
                is_transmitting = self._is_transmitting
                tail_time = self._tail_time
                if (
                    is_running is None
                    or is_squelch_open is None
                    or is_transmitting is None
                    or tail_time is None
                ):
                    continue

                if not is_running():
                    continue

                self._check_rx_stale()
                self._check_tx_stuck_after_rx_closed(
                    is_squelch_open,
                    is_transmitting,
                    tail_time,
                )

        except Exception as e:
            log.exception("[AudioHealth] Watchdog thread crashed")

            if self.audit:
                self.audit.critical(
                    event_type=AuditEvent.WATCHDOG_RECOVERY_FAILED,
                    source="audio_health",
                    message="Audio health watchdog thread crashed",
                    metadata={
                        "error": repr(e),
                    },
                )

    def _check_rx_stale(self):
        if self.rx_stale():
            age = time.monotonic() - self.last_rx_activity
            log.error(
                "[AudioHealth] RX stream stale for %.1fs; forcing service restart",
                age,
            )

            if self.audit:
                self.audit.error(
                    event_type=AuditEvent.USB_STREAM_STALLED,
                    source="audio_health",
                    message="RX stream stale watchdog triggered",
                    metadata={
                        "stale_age": round(age, 2),
                        "stale_timeout": self.stale_timeout,
                    },
                )

            os._exit(70)

    def _check_tx_stuck_after_rx_closed(
        self,
        is_squelch_open,
        is_transmitting,
        tail_time,
    ):
        rx_open = bool(is_squelch_open())
        tx_active = bool(is_transmitting())

        if tx_active and not rx_open:
            if self.tx_without_rx_since is None:
                self.tx_without_rx_since = time.monotonic()
                return

            age = time.monotonic() - self.tx_without_rx_since
            configured_tail_time = float(tail_time())
            limit = configured_tail_time + self.tx_stuck_grace

            if age > limit:
                log.error(
                    "[AudioHealth] TX remained active %.1fs after RX closed "
                    "(tail=%.1fs grace=%.1fs); forcing service restart",
                    age,
                    configured_tail_time,
                    self.tx_stuck_grace,
                )
                if self.audit:
                    self.audit.error(
                        event_type=AuditEvent.STUCK_TX_DETECTED,
                        source="audio_health",
                        message="TX stuck active after RX closed",
                        metadata={
                            "tx_stuck_age": round(age, 2),
                            "tail_time": round(configured_tail_time, 2),
                            "grace_time": round(self.tx_stuck_grace, 2),
                        },
                    )
                os._exit(70)

        else:
            self.tx_without_rx_since = None
