import logging
import time

from audio.primitives import fade_out
from runtime.audit import AuditEvent
from runtime.logging_utils import debug_enabled


class Control:
    def __init__(
        self,
        config,
        state,
        tx_state,
        ptt_manager,
        tot_manager,
        send_pcm,
        send_chunk,
        play_courtesy_tone,
        mark_post_tx,
        audit=None,
    ):
        self.config = config
        self.state = state
        self.tx_state = tx_state
        self.ptt_manager = ptt_manager
        self.tot_manager = tot_manager
        self.send_pcm = send_pcm
        self.send_chunk = send_chunk
        self.play_courtesy_tone = play_courtesy_tone
        self.mark_post_tx = mark_post_tx
        self.audit = audit
        self.tot_manager.reset()
        self.state.current_rms = 0.0

    def _handle_vox_delay(self, repeater_cfg):
        if self.ptt_manager.ptt_mode == "CM108":
            return

        delay_sec = float(repeater_cfg.get("carrier_delay", 0) or 0)

        if delay_sec <= 0:
            return

        audio_cfg = self.config.config["audio"]
        sample_rate = audio_cfg["sample_rate"]
        chunk_size = audio_cfg["chunk_size"]

        num_chunks = int((delay_sec * sample_rate) // chunk_size)

        if debug_enabled():
            logging.debug(f"[VOX] Wake burst: {num_chunks} chunks ({delay_sec:.2f}s)")

        import numpy as np

        for i in range(num_chunks):
            noise = np.random.normal(0, 1, chunk_size)
            noise = np.diff(np.concatenate(([0], noise)))
            noise *= 20000

            chunk = noise.astype(np.int16)

        for i in range(num_chunks):
            if debug_enabled() and i == 0:
                logging.debug("[VOX] Noise burst start")

            self.send_chunk(chunk)

        if debug_enabled():
            logging.debug("[VOX] Noise burst end")

    def _play_courtesy(self, repeater_cfg):
        tx = self.tx_state

        if repeater_cfg["courtesy_tone_enabled"] and not tx.skip_courtesy_tone:
            self.play_courtesy_tone()
        elif tx.skip_courtesy_tone and debug_enabled():
            logging.debug("Skipping courtesy tone after CW ID.")

    def start(self):
        tx = self.tx_state
        cfg = self.config.config
        repeater_cfg = cfg["repeater"]
        now = time.time()

        tx.tx_start_pending = False

        if self.tot_manager.is_locked():
            return

        tx.transmitting = True
        self.state.transmission_start_time = now
        self.tot_manager.reset()
        self.tot_manager.tx_start_time = now

        try:
            self.ptt_manager.safe_ptt_key()

        except Exception as e:
            if self.audit:
                self.audit.error(
                    event_type=AuditEvent.PTT_FAILURE,
                    source="tx_control",
                    message="Failed to key transmitter",
                    metadata={
                        "action": "key",
                        "error": repr(e),
                        "ptt_mode": getattr(self.ptt_manager, "ptt_mode", None),
                    },
                )

            logging.exception("[PTT] Failed to key transmitter")
            raise

        self._handle_vox_delay(repeater_cfg)

        logging.info("Starting transmission")

    def stop(self):
        tx = self.tx_state
        cfg = self.config.config
        audio_cfg = cfg["audio"]
        repeater_cfg = cfg["repeater"]
        now = time.time()

        # Prime ALSA before tone playback
        self.send_pcm(tx.silence_bytes)

        self._play_courtesy(repeater_cfg)

        fade_duration = 0.1
        sample_rate = audio_cfg["sample_rate"]
        chunk_size = audio_cfg["chunk_size"]
        num_fade_chunks = int((fade_duration * sample_rate) // chunk_size)

        fade_out(self.send_pcm, tx.silence_bytes, num_fade_chunks)

        tx.transmitting = False
        self.state.transmission_start_time = None
        self.tot_manager.reset()
        self.tot_manager.tx_start_time = None
        tx.last_transmission = now

        logging.info("Transmission stopped with fade-out")

        if not tx.skip_courtesy_tone:
            self.mark_post_tx()
        tx.skip_courtesy_tone = False

        try:
            self.ptt_manager.safe_ptt_unkey()
            logging.info("Transmitter unkeyed.")
        except Exception as e:
            if self.audit:
                self.audit.error(
                    event_type=AuditEvent.PTT_FAILURE,
                    source="tx_control",
                    message="Failed to unkey transmitter",
                    metadata={
                        "action": "unkey",
                        "error": repr(e),
                        "ptt_mode": getattr(self.ptt_manager, "ptt_mode", None),
                    },
                )
            logging.exception("[PTT] Failed to unkey transmitter")
            raise
