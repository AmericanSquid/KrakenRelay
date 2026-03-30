import numpy as np
import time
import logging
from ..signal_gate import carrier_validity_probe, update_squelch_state

from audio import (
    check_clipping,
    calculate_db_level
)

from audio.primitives import sanitize_audio, compute_rms, ensure_float32
from runtime.logging_utils import debug_enabled

class ProcessAudio:
    def __init__(self, controller, config):
        self.controller = controller
        self.config = config

    def _update_rms(self, samples):
        controller = self.controller
        controller.current_rms = compute_rms(samples)

    def _normalize_samples(self, samples):
        samples = ensure_float32(samples)
        samples = sanitize_audio(samples)
        return samples

    def _compute_levels(self, samples, raw_samples):
        return (
            float(calculate_db_level(samples)),
            float(calculate_db_level(raw_samples))
        )

    def _reset_carrier_probe(self):
        gate = self.controller.signal_gate

        _carrier_valid = False
        _carrier_probe_start = None
        _carrier_last_level_db = None

    def is_squelch_open_edge(self, squelch_open_now, prev_open):
        return squelch_open_now and not prev_open

    def is_squelch_close_edge(self, squelch_open_now, prev_open):
        return not squelch_open_now and prev_open

    def process_audio(self):
        controller = self.controller
        gate = controller.signal_gate
        tx = controller.tx_state
        audio_cfg = self.config.config["audio"]
        repeater_cfg = self.config.config["repeater"]
        debug_on = debug_enabled()

        def tail_expired(now, tx, repeater_cfg):
            return now - tx.last_audio_time > repeater_cfg['tail_time']

        try:
            data = controller.streams.input_stream.read(controller.chunk_size, exception_on_overflow=False)
        except Exception as e:
            # If we're shutting down, just exit quietly
            if not controller.running:
                return
            logging.error(f"[Input] Input stream read failed: {e}")
            raise

        raw_samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)

        if raw_samples.size == 0:
            controller.current_rms = 0.0
            raw_samples = np.zeros(controller.chunk_size, dtype=np.float32)
        else:
            raw_samples = sanitize_audio(raw_samples)

        # Keep a filtered RX copy that will be used for squelch / buffering / TX audio
        samples = raw_samples.copy()

        if (
            audio_cfg.get("highpass_enabled", False)
            or audio_cfg.get("notch_enabled", False)
        ):
            samples = controller.dsp_rx.process_int16_to_int16(samples)

        samples = self._normalize_samples(samples)

        # Meter / RMS should reflect the filtered RX audio
        self._update_rms(samples)
        controller.meter.update(samples, "rx")

        now = time.time()

        # IMPORTANT:
        # - squelch should see filtered RX audio
        # - carrier-validity should see raw RX level movement
        level_db, raw_level_db = self._compute_levels(samples, raw_samples)

        prev_open = gate.squelch_open
        squelch_open_now = update_squelch_state(
            gate, 
            audio_cfg=audio_cfg,
            level_db=level_db, 
            now=now
        )

        just_opened = False

        # Track squelch transitions for anti-kerchunk holdoff
        if self.is_squelch_open_edge(squelch_open_now, prev_open):
            if not tx.transmitting:
                just_opened = True
                gate.squelch_open_time = now
                gate.kerchunk_buffer = []  # clear buffer on a fresh open edge
                tx.tx_start_pending = True
                if debug_on:
                    logging.debug("[Anti-Kerchunk] Squelch open. Holding off")
            else:
                # Squelch reopened during an ongoing transmission – no kerchunk holdoff
                if debug_on:
                    logging.debug("[Anti-Kerchunk] Squelch reopened during transmit; kerchunk holdoff bypassed")
        elif self.is_squelch_close_edge(squelch_open_now, prev_open):
            if not tx.transmitting and gate.kerchunk_buffer:
                logging.info("[Anti-Kerchunk] Suppressed short key-up.")
                gate.kerchunk_buffer = []
            if not tx.transmitting:
                tx.tx_start_pending = False
            if debug_on:
                logging.debug("[Anti-Kerchunk] Squelch closed.")

        # --- Carrier Validity Gate --- #
        if squelch_open_now:
            carrier_validity_probe(
                state=gate,
                controller=controller,
                audio_cfg=audio_cfg,
                raw_level_db=raw_level_db,
                now=now,
                just_opened=just_opened,
                samples=samples
            )
        else:
            # squelch closed: reset probe state
            self._reset_carrier_probe()

        if squelch_open_now:
            s = self._normalize_samples(samples)
            self._update_rms(s)

            if not tx.transmitting:
                if just_opened:
                    logging.info("Squelch opened - Starting Tx")

            controller.tx_pipeline.feed(samples)
        else:
            controller.current_rms = 0
            if tx.transmitting:
                if tail_expired(now, tx, repeater_cfg):
                    logging.info("Silence persists beyond tail time. Stopping transmission.")
                    controller.tx_control.stop()
                else:
                    # Keep audio flowing during tail hang
                    controller.audio_io.send_pcm(tx.silence_chunk)
            else:
                pass
