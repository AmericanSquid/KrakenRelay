import logging
import os
import time

from audio.health import AudioStreamFailure
from core.common import shutdown_transmitter
from runtime.audit import AuditEvent


class AudioLoop:
    def __init__(
        self,
        config,
        state,
        tx_state,
        send_pcm,
        stop_transmission,
        unkey_transmitter,
        process_audio,
        request_cw,
        schedule_id,
        tot_manager,
        plugins=None,
        audit=None,
    ):
        self.config = config
        self.state = state
        self.tx_state = tx_state
        self.send_pcm = send_pcm
        self.stop_transmission = stop_transmission
        self.unkey_transmitter = unkey_transmitter
        self.process_audio = process_audio
        self.request_cw = request_cw
        self.schedule_id = schedule_id
        self.tot_manager = tot_manager
        self.plugins = plugins
        self.audit = audit

    def _handle_cw_playback(self):
        cfg = self.config.config
        audio_cfg = cfg["audio"]
        chunk_size = audio_cfg["chunk_size"]
        tx = self.tx_state

        cw_gen = self.state.cw_gen
        if cw_gen is None:
            return False

        sr = int(audio_cfg["sample_rate"])
        frame_sec = float(chunk_size) / float(sr)

        now_m = time.monotonic()
        next_t = self.state.cw_next_t
        if next_t is None:
            next_t = now_m

        if now_m < next_t:
            time.sleep(next_t - now_m)
        else:
            # Do not burst queued chunks to make up for a late audio loop. In
            # particular, a catch-up burst can leave the final CW frame queued
            # when the transmitter is unkeyed.
            next_t = now_m

        try:
            chunk = next(cw_gen)
            self.send_pcm(chunk)
            # KR_PLUGIN_CW_TICK_START
            if self.plugins is not None:
                self.plugins.emit_tick()
            # KR_PLUGIN_CW_TICK_END
            self.state.cw_next_t = next_t + frame_sec
        except StopIteration:
            self.state.cw_gen = None
            self.state.cw_next_t = None
            tx.skip_courtesy_tone = True
            if getattr(tx, "transmitting", False):
                self.stop_transmission()

        return True

    def _handle_audio_error(self, e, consecutive_errors, max_backoff):
        logging.error(f"Error in audio loop: {e}")

        if not self.state.running:
            return consecutive_errors, True

        backoff = min(0.2 * (2 ** (consecutive_errors - 1)), max_backoff)
        time.sleep(backoff)

        return consecutive_errors, False

    def _handle_normal_audio(self, manual_id_event):
        try:
            self.process_audio.process_audio()

        except AudioStreamFailure as e:
            logging.exception("[AudioHealth] RX stream unhealthy; restarting repeater")

            if self.audit:
                self.audit.critical(
                    event_type=AuditEvent.WATCHDOG_TRIGGERED,
                    source="audio_loop",
                    message="Audio health failure threshold reached; exiting for service restart",
                    metadata={
                        "error": repr(e),
                        "exit_code": 70,
                    },
                )

            os._exit(70)

        if manual_id_event.is_set():
            manual_id_event.clear()
            logging.info("[WebUI] Manual ID requested.")
            try:
                self.schedule_id.send_id()
            except Exception:
                logging.exception("[Repeater] Manual ID failed.")

        self.schedule_id.check_and_send()

    def _shutdown_cleanup(self):
        shutdown_transmitter(
            self.tx_state,
            self.stop_transmission,
            self.unkey_transmitter,
        )

    def audio_loop(self):
        manual_id_event = self.request_cw.manual_id_event

        consecutive_errors = 0
        max_errors = 5
        max_backoff = 0.5
        fatal_reason = None

        logging.info("[Repeater] Audio thread started.")

        while self.state.running:
            try:
                self.tot_manager.check_lockout_expired()
                if self._handle_cw_playback():
                    continue

                self._handle_normal_audio(manual_id_event)

                # KR_PLUGIN_TICK_START
                if self.plugins is not None:
                    self.plugins.emit_tick()
                # KR_PLUGIN_TICK_END

                if consecutive_errors:
                    logging.info(
                        "[Repeater] Audio loop recovered after %d error(s).",
                        consecutive_errors,
                    )
                    consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1

                consecutive_errors, should_exit = self._handle_audio_error(
                    e, consecutive_errors, max_backoff
                )

                if should_exit:
                    break

                if consecutive_errors >= max_errors:
                    fatal_reason = f"{consecutive_errors} consecutive audio loop errors"
                    break

        if not self.state.running and fatal_reason is None:
            logging.info("[Repeater] Audio thread stopping (requested)")
        else:
            reason = fatal_reason or "unknown fatal error"
            logging.critical(f"[Repeater] Audio loop exiting due to: {reason}")
            if self.audit:
                self.audit.critical(
                    event_type=AuditEvent.CONTROLLER_CRASH,
                    source="audio_loop",
                    message="Audio loop exited unexpectedly",
                    metadata={
                        "reason": reason,
                        "consecutive_errors": consecutive_errors,
                        "max_errors": max_errors,
                    },
                )
        self.state.running = False
        self._shutdown_cleanup()

        logging.info("[Repeater] Audio thread exited.")
