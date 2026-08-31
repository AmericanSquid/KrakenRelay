import logging

import numpy as np
from speexdsp_ns import NoiseSuppression


class SpeexNSStage:
    def __init__(
        self,
        config,
        frame_size: int,
        is_transmitting,
        sample_rate: int,
        enabled: bool = True,
    ):
        self.frame_size = frame_size
        self.sample_rate = sample_rate
        self.enabled = enabled
        self.is_transmitting = is_transmitting
        self.config = config

        self.ns = NoiseSuppression.create(
            frame_size=frame_size,
            sample_rate=sample_rate,
        )

        audio_cfg = self.config.config["audio"]
        toggle_denoise = int(audio_cfg.get("speex_ns", 1))
        self.ns.set_denoise(toggle_denoise)

        suppress_db = int(audio_cfg.get("speex_ns_db", -20))
        self.ns.set_noise_suppress(suppress_db)

        toggle_agc = int(audio_cfg.get("speex_agc", 0))
        self.ns.set_agc(toggle_agc)

        agc_level = int(audio_cfg.get("speex_agc_level", 0))
        self.ns.set_agc_level(agc_level)

        toggle_vad = int(audio_cfg.get("speex_vad", 0))
        self.ns.set_vad(toggle_vad)

        self._last_log_time = 0.0

    def reset(self):
        pass

    def _rms(self, samples):
        return np.sqrt(np.mean(samples.astype(np.float32) ** 2) + 1e-9)

    def process_int16_to_int16(self, samples: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return samples

        if len(samples) != self.frame_size:
            logging.debug("[SpeexNS] Frame size mismatch, skipping")
            return samples

        raw_in = np.asarray(samples, dtype=np.int16, order="C").tobytes()
        raw_out = self.ns.process(raw_in)
        out = np.frombuffer(raw_out, dtype=np.int16).copy()

        in_rms = self._rms(samples)
        out_rms = self._rms(out)

        # Convert to dBFS for readable logging
        in_db = 20 * np.log10(in_rms / 32768.0 + 1e-9)
        out_db = 20 * np.log10(out_rms / 32768.0 + 1e-9)

        # 🔴 Detect aggressive suppression
        if in_rms > 200.0 and out_rms < (in_rms * 0.10):
            logging.warning(
                f"[SpeexNS] Strong suppression detected."
                f"(in={in_db:.1f} dBFS → out={out_db:.1f} dBFS)"
            )
        #    return samples

        # 🟡 Occasional debug logging (throttled)
        import time

        now = time.time()
        if self.is_transmitting():
            if now - self._last_log_time > 2.0:
                logging.debug(f"[SpeexNS] in={in_db:.1f} dBFS out={out_db:.1f} dBFS")
                self._last_log_time = now

        return out
