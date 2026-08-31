import logging
import threading

from runtime.audit import AuditEvent


class Lifecycle:
    def __init__(
        self,
        state,
        tx_state,
        signal_gate,
        audio_loop,
        process_audio,
        audio_health,
        streams,
        unkey_transmitter,
        plugins,
        request_cw,
        schedule_id,
        audit=None,
    ):
        self.state = state
        self.tx_state = tx_state
        self.signal_gate = signal_gate
        self.audio_loop = audio_loop
        self.process_audio = process_audio
        self.audio_health = audio_health
        self.streams = streams
        self.unkey_transmitter = unkey_transmitter
        self.plugins = plugins
        self.request_cw = request_cw
        self.schedule_id = schedule_id
        self.audit = audit

        self._shutdown_failed = False
        self._cleanup_lock = threading.Lock()

    def request_manual_id(self):
        self.request_cw.request_manual_id()

    def send_id(self):
        self.schedule_id.send_id()

    def start(self):
        if self.state.running:
            logging.warning("Repeater is already running.")
            return

        self.state.running = True
        # Start Input 2 reader ONCE, before the main audio loop starts.
        try:
            input2_reader = getattr(self.process_audio, "input2_reader", None)

            if input2_reader is not None:
                input2_reader.start()
                logging.info("[Lifecycle] Input2Reader started.")
        except Exception:
            logging.exception("[Lifecycle] Failed starting Input2Reader")

        self.audio_thread = threading.Thread(target=self.audio_loop.audio_loop)
        self.audio_thread.start()
        if self.audio_health:
            self.audio_health.start_watchdog(
                is_running=lambda: self.state.running,
                is_squelch_open=lambda: self.signal_gate.squelch_open,
                is_transmitting=lambda: self.tx_state.transmitting,
                tail_time=lambda: float(
                    self.audio_loop.config.config.get("repeater", {}).get(
                        "tail_time", 2.0
                    )
                ),
            )
            logging.info("Audio health watchdog started")
        logging.info("Repeater runtime started")

        if self.audit:
            self.audit.info(
                event_type=AuditEvent.CONTROLLER_START,
                source="lifecycle",
                message="Repeater runtime started",
            )

        logging.info(f"🧵 Active threads after start: {threading.active_count()}")
        for t in threading.enumerate():
            try:
                ident = t.ident  # Python thread ident
                # On Linux, get native TID (will match top/htop)
                tid = t.native_id if hasattr(t, "native_id") else None
                logging.info(f"Thread name: {t.name}, ident: {ident}, native_id: {tid}")
            except Exception as e:
                logging.info(f"Could not get info for thread: {t} ({e})")

    def cleanup(self) -> bool:
        tx = self.tx_state

        logging.info("========== [Repeater] Cleanup Starting ==========")

        if self.audit:
            self.audit.info(
                event_type=AuditEvent.CONTROLLER_STOP,
                source="lifecycle",
                message="Repeater runtime cleanup started",
            )

        logging.info(
            "Runtime state before cleanup: running=%s transmitting=%s skip_courtesy_tone=%s shutdown_failed=%s",
            self.state.running,
            tx.transmitting,
            tx.skip_courtesy_tone,
            self._shutdown_failed,
        )
        ok = True
        self._shutdown_failed = False

        # --- A) SAFETY FIRST: UNKEY ---
        logging.info(
            "[Cleanup Step A] Unkeying transmitter and suppressing courtesy tone..."
        )
        try:
            tx.transmitting = False
            tx.skip_courtesy_tone = True
            self.unkey_transmitter()
            logging.info("[Cleanup Step A] PTT unkeyed (forced cleanup).")
        except Exception:
            logging.exception("[Cleanup Step A] Failed to unkey PTT during cleanup.")
            ok = False

        # --- B) SIGNAL ALL LOOPS/THREADS TO STOP ---
        self.state.running = (
            False  # Stop the audio loop first so health checks stand down
        )
        logging.info(
            "[Cleanup Step B] Stopping audio loop: running=%s", self.state.running
        )

        try:
            input2_reader = getattr(self.process_audio, "input2_reader", None)

            if input2_reader is not None:
                logging.info("[Cleanup Step B] Stopping Input2Reader...")
                input2_reader.stop()
                logging.info("[Cleanup Step B] Input2Reader stopped.")
        except Exception:
            logging.exception("[Cleanup Step B] Failed stopping Input2Reader.")
            ok = False

        health_thread_ok = True
        if self.audio_health:
            try:
                health_thread_ok = self.audio_health.stop_watchdog(timeout=2.5)
                if health_thread_ok:
                    logging.info("[Cleanup Step B] Audio health watchdog stopped.")
                else:
                    logging.error(
                        "[Cleanup Step B] Audio health watchdog did not stop cleanly."
                    )
            except Exception:
                logging.exception(
                    "[Cleanup Step B] Failed while stopping audio health watchdog."
                )
                health_thread_ok = False
            ok &= health_thread_ok

        # --- C) JOIN THREADS ---
        def _join(th, name: str, timeout: float) -> bool:
            if not th:
                return True
            if not th.is_alive():
                return True
            if th is threading.current_thread():
                logging.warning(
                    "[Repeater] cleanup called from %s; skipping self-join", name
                )
                return True
            logging.info("[Cleanup Step C] Joining thread: %s", name)
            th.join(timeout=timeout)
            if th.is_alive():
                logging.error("[Repeater] %s still alive after %.1fs", name, timeout)
                return False
            logging.info("[Cleanup Step C] Joined thread: %s", name)
            return True

        ok &= _join(getattr(self, "audio_thread", None), "audio_thread", 3.0)

        # KR_PLUGIN_SHUTDOWN_START
        if self.plugins is not None:
            self.plugins.emit_shutdown()
        # KR_PLUGIN_SHUTDOWN_END

        if not ok:
            logging.critical(
                "[Repeater] cleanup incomplete - leaving PortAudio intact. Restarting service to recover."
            )
            if self.audit:
                self.audit.critical(
                    event_type=AuditEvent.CONTROLLER_CRASH,
                    source="lifecycle",
                    message="Cleanup failed; restart required",
                    metadata={
                        "shutdown_failed": True,
                    },
                )
            self._shutdown_failed = True
            return False

        # --- D) CLOSE STREAMS (threads are gone) ---
        if ok:
            for name in (
                "input_stream",
                "input_stream_2",
                "output_stream",
                "output_stream_2",
            ):
                s = getattr(self.streams, name, None)
                if not s:
                    continue
                try:
                    logging.info(f"[Cleanup Step D] Closing stream: {name}")
                    try:
                        s.stop_stream()
                    except Exception:
                        pass
                    s.close()
                    logging.info(f"[Cleanup Step D] Closed stream: {name}")
                except Exception as e:
                    logging.warning(f"[Cleanup Step D] Exception closing {name}: {e}")
                    ok = False
                setattr(self.streams, name, None)

        self._shutdown_failed = not ok
        if ok:
            logging.info("Cleanup completed successfully. No issues detected.")

            if self.audit:
                self.audit.info(
                    event_type=AuditEvent.CONTROLLER_STOP,
                    source="lifecycle",
                    message="Repeater runtime stopped cleanly",
                    metadata={
                        "shutdown_failed": False,
                    },
                )

        else:
            logging.warning(
                "Cleanup completed with ERRORS. Check previous logs for details."
            )

        logging.info(
            "[Repeater] cleanup() done. ok=%s shutdown_failed=%s",
            ok,
            self._shutdown_failed,
        )
        logging.info("🧵 Active threads after cleanup: %d", threading.active_count())
        for t in threading.enumerate():
            logging.info("  • %s (alive=%s)", t.name, t.is_alive())

        logging.info("========== [Repeater] Clean Up Ended ==========")
        logging.info("73 de K3AYV (American Squid).")
        return ok
