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
        cw_playback=None,
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
        self.cw_playback = cw_playback

    def _handle_cw_playback(self):
        playback = self.cw_playback
        if playback is None or self.state.cw_gen is None:
            return False
        if playback.voice_emitted:
            return False
        if not self.tx_state.transmitting:
            playback.clear()
            return False

        chunk = playback.take()
        if chunk is None:
            self.tx_state.skip_courtesy_tone = True
            self.stop_transmission()
        else:
            self.send_pcm(chunk)
        return True

    def _handle_audio_error(self, e, consecutive_errors, max_backoff):
        logging.exception("Error in audio loop iteration: %s", e)

        if not self.state.running:
            return consecutive_errors, True

        backoff = min(0.2 * (2 ** (consecutive_errors - 1)), max_backoff)
        time.sleep(backoff)

        return consecutive_errors, False

    def _handle_normal_audio(self, manual_id_event):
        try:
            read_ok = self.process_audio.process_audio()

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
        return read_ok

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
                if self.cw_playback is not None:
                    self.cw_playback.begin_iteration()
                read_ok = self._handle_normal_audio(manual_id_event)
                if read_ok is not False:
                    self._handle_cw_playback()

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
