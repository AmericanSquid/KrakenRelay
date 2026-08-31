import logging
import time

import numpy as np

from audio import calculate_db_level
from audio.fade_in import build_rx_fade_in
from audio.primitives import compute_rms, ensure_float32, sanitize_audio
from audio.resampling import resample_to_length
from runtime.audit import AuditEvent
from runtime.logging_utils import debug_enabled

from ..primitives import (
    is_squelch_close_edge,
    is_squelch_open_edge,
    reset_carrier_probe,
)
from ..signal_gate import carrier_validity_probe, update_squelch_state


class ProcessAudio:
    def __init__(
        self,
        config,
        dsp_rx,
        tx_pipeline,
        py_dsp_rx,
        streams,
        signal_gate,
        tx_state,
        audio_health,
        meter,
        state,
        send_pcm,
        stop_transmission,
        audit=None,
    ):
        self.config = config
        self.dsp_rx = dsp_rx
        self.py_dsp_rx = py_dsp_rx
        self.tx_pipeline = tx_pipeline
        self.streams = streams
        self.signal_gate = signal_gate
        self.tx_state = tx_state
        self.audio_health = audio_health
        self.meter = meter
        self.state = state
        self.send_pcm = send_pcm
        self.stop_transmission = stop_transmission
        self.audit = audit
        self._last_input2_warn = 0.0
        self.rx_fade_in = build_rx_fade_in(self.config.config)

    def _read_input_stream(self, stream, chunk_size, label):
        """
        Read a single mono int16 input stream and return float32 samples.
        """
        data = stream.read(chunk_size, exception_on_overflow=False)
        expected = chunk_size * 2  # int16 mono = 2 bytes per frame

        if not data or len(data) < expected:
            raise RuntimeError(
                f"short {label} RX read: got {len(data) if data else 0}, expected {expected}"
            )

        return np.frombuffer(data, dtype=np.int16).astype(np.float32)

    def _read_rx_sources(self, chunk_size):
        """
        Direct dual-input read.

        Input 1 is always read.
        Input 2 is read only if dual input is configured/opened.
        """
        primary = self._read_input_stream(
            self.streams.input_stream,
            chunk_size,
            "primary",
        )

        secondary = None
        input_stream_2 = self.streams.input_stream_2

        if input_stream_2 is not None:
            secondary = self._read_input_stream(
                input_stream_2,
                chunk_size,
                "secondary",
            )
            secondary = resample_to_length(secondary, chunk_size, label="input_2")

            if secondary.size != primary.size:
                now = time.time()
                if now - self._last_input2_warn > 5.0:
                    logging.warning(
                        "[DualInput] Secondary chunk size mismatch: %s != %s; ignoring secondary chunk",
                        secondary.size,
                        primary.size,
                    )
                    self._last_input2_warn = now
                secondary = None

        return primary, secondary

    def _mix_rx_sources(self, primary, secondary):
        """
        Simple v1 dual input behavior:
        Input 1 + Input 2, clip protected.

        No voting.
        No selector.
        No exposed mix gain.
        """
        if primary is None:
            chunk_size = int(self.config.config["audio"]["chunk_size"])
            return np.zeros(chunk_size, dtype=np.float32)

        if secondary is None:
            return sanitize_audio(primary.copy().astype(np.float32))

        mixed = primary.astype(np.float32) + secondary.astype(np.float32)
        mixed = np.clip(mixed, -32768.0, 32767.0)

        return sanitize_audio(mixed.astype(np.float32))

    def _link_samples_for_output_2(self, primary_raw, audio_cfg):
        """
        Return Input-1-only live audio for Output 2 link mode, or None.

        Link mode must not send Input 2, CW, courtesy tones, TOT tones, or
        tail silence to Output 2. The live RX path supplies this local-only
        chunk only when Input 1 itself appears open using the same coarse
        squelch threshold.
        """
        output_2_mode = str(
            audio_cfg.get("output_2_mode", "simulcast") or "simulcast"
        ).lower()

        if output_2_mode != "link":
            return None

        if primary_raw is None or primary_raw.size == 0:
            return None

        local_level_db = float(calculate_db_level(primary_raw))
        threshold = float(audio_cfg.get("squelch_threshold", -40))
        hysteresis = float(audio_cfg.get("squelch_hysteresis_db", 0) or 0)

        # Conservative open test so Input-2-only traffic does not leak idle
        # primary-input noise out to the EchoLink/node side.
        if local_level_db >= (threshold + max(0.0, hysteresis * 0.25)):
            return sanitize_audio(ensure_float32(primary_raw.copy()))

        return None

    def process_audio(self):
        gate = self.signal_gate
        tx = self.tx_state

        audio_cfg = self.config.config["audio"]
        repeater_cfg = self.config.config["repeater"]

        chunk_size = int(audio_cfg["chunk_size"])
        debug_on = debug_enabled()

        def tail_expired(now, tx, repeater_cfg):
            return now - tx.last_audio_time > repeater_cfg["tail_time"]

        try:
            primary_raw, secondary_raw = self._read_rx_sources(chunk_size)
            self.audio_health.record_success()

        except Exception as e:
            # If we're shutting down, just exit quietly.
            if not self.state.running:
                return

            error_text = repr(e)

            if self.audit:
                event_type = AuditEvent.AUDIO_STREAM_READ_FAILURE

                if "overflow" in error_text.lower():
                    event_type = AuditEvent.PYAUDIO_OVERFLOW

                self.audit.error(
                    event_type=event_type,
                    source="process_audio",
                    message="RX audio stream read failed",
                    metadata={
                        "error": error_text,
                        "chunk_size": chunk_size,
                        "expected_bytes": chunk_size * 2,
                        "failure_count_next": getattr(self.audio_health, "failures", 0)
                        + 1,
                        "max_failures": getattr(
                            self.audio_health, "max_failures", None
                        ),
                    },
                )

            self.audio_health.record_failure(e)
            return

        raw_samples = self._mix_rx_sources(primary_raw, secondary_raw)

        if raw_samples.size == 0:
            self.state.current_rms = 0.0
            raw_samples = np.zeros(chunk_size, dtype=np.float32)
        else:
            raw_samples = sanitize_audio(raw_samples)

        # Filtered RX copy used for squelch / buffering / TX audio.
        # In dual-input mode, this is already Input 1 + Input 2.
        samples = raw_samples.copy()

        if audio_cfg.get("highpass_enabled", False) or audio_cfg.get(
            "notch_enabled", False
        ):
            samples = self.dsp_rx.process_int16_to_int16(samples)

        if self.py_dsp_rx is not None:
            samples = self.py_dsp_rx.process_int16_to_int16(samples)

        samples = sanitize_audio(ensure_float32(samples))

        # Meter / RMS should reflect the filtered RX audio.
        self.state.current_rms = compute_rms(samples)
        self.meter.update(samples, "rx")

        now = time.time()

        # Squelch sees filtered RX audio.
        # Carrier validity sees raw RX level movement.
        level_db = float(calculate_db_level(samples))
        raw_level_db = float(calculate_db_level(raw_samples))

        prev_open = gate.squelch_open
        squelch_open_now = update_squelch_state(
            gate,
            audio_cfg=audio_cfg,
            level_db=level_db,
            now=now,
        )

        just_opened = False

        # Track squelch transitions for anti-kerchunk holdoff.
        if is_squelch_open_edge(squelch_open_now, prev_open):
            if not tx.transmitting:
                just_opened = True
                gate.squelch_open_time = now
                gate.kerchunk_buffer = []
                tx.tx_start_pending = True

                if debug_on:
                    logging.debug("[Anti-Kerchunk] Squelch open. Holding off")
            else:
                if debug_on:
                    logging.debug(
                        "[Anti-Kerchunk] Squelch reopened during transmit; "
                        "kerchunk holdoff bypassed"
                    )

        elif is_squelch_close_edge(squelch_open_now, prev_open):
            if not tx.transmitting and gate.kerchunk_buffer:
                logging.info("[Anti-Kerchunk] Suppressed short key-up.")
                gate.kerchunk_buffer = []

            if not tx.transmitting:
                tx.tx_start_pending = False

            if debug_on:
                logging.debug("[Anti-Kerchunk] Squelch closed.")

        # Carrier validity gate.
        if squelch_open_now:
            carrier_validity_probe(
                state=gate,
                tx_state=tx,
                stop_transmission=self.stop_transmission,
                audit=self.audit,
                audio_cfg=audio_cfg,
                raw_level_db=raw_level_db,
                now=now,
                just_opened=just_opened,
                samples=samples,
            )
        else:
            reset_carrier_probe(gate)

        if squelch_open_now:
            s = sanitize_audio(ensure_float32(samples))
            self.state.current_rms = compute_rms(s)

            if not tx.transmitting:
                if just_opened:
                    logging.info("Squelch opened - Starting Tx")

            # In link mode, Output 2 gets only Input 1/local live audio.
            # In simulcast mode, this returns None and Output 2 mirrors normally.
            link_samples = self._link_samples_for_output_2(primary_raw, audio_cfg)
            tx_samples, link_samples = self.rx_fade_in.apply_many(s, link_samples)

            self.tx_pipeline.feed(tx_samples, link_samples=link_samples)

        else:
            self.rx_fade_in.reset()
            self.state.current_rms = 0.0

            if tx.transmitting:
                if tail_expired(now, tx, repeater_cfg):
                    logging.info(
                        "Silence persists beyond tail time. Stopping transmission."
                    )
                    self.stop_transmission()
                else:
                    # Keep audio flowing during tail hang.
                    # In Output 2 link mode this intentionally does not go to the node
                    # because no link_samples/link_pcm is supplied.
                    self.send_pcm(tx.silence_chunk)
