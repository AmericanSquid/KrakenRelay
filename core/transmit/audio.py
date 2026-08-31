import logging

import numpy as np

from audio import check_clipping
from runtime.audit import AuditEvent


class TxAudio:
    def __init__(self, config, dsp_tx, meter, send_pcm, audit=None):
        self.config = config
        self.dsp_tx = dsp_tx
        self.meter = meter
        self.send_pcm = send_pcm
        self.audit = audit

    def send_chunk(
        self, samples: np.ndarray, link_samples: np.ndarray | None = None
    ) -> None:
        cfg = self.config.config
        audio_cfg = cfg["audio"]

        if (
            audio_cfg.get("limiter_enabled", False)
            or audio_cfg.get("compressor_enabled", False)
            or audio_cfg.get("highpass_enabled", False)
            or audio_cfg.get("notch_enabled", False)
            or audio_cfg.get("tx_tone_eq_enabled", False)
        ):
            samples = self.dsp_tx.process_int16_to_int16(samples)

        self.meter.update(samples, "tx")
        check_clipping(samples)

        try:
            self.send_pcm(samples, link_pcm=link_samples)
        except Exception as e:
            error_text = repr(e)

            if self.audit:
                event_type = AuditEvent.AUDIO_STREAM_WRITE_FAILURE
                if "underflow" in error_text.lower():
                    event_type = AuditEvent.PYAUDIO_UNDERFLOW
                self.audit.error(
                    event_type=event_type,
                    source="tx_audio",
                    message="TX audio stream write failed",
                    metadata={
                        "error": error_text,
                        "samples": int(len(samples)) if samples is not None else 0,
                    },
                )
            logging.exception("[TxAudio] TX audio stream write failed")
            raise
