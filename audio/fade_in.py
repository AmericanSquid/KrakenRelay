import logging
from typing import Optional

import numpy as np

from audio.primitives import (
    ensure_float32,
    partial_linear_gain_ramp,
    sanitize_audio,
)


class RxFadeIn:
    """
    Stateful fade-in for live RX audio after squelch opens.

    This is chunk-aware: it fades the first fade_samples after RX opens,
    then passes later chunks through unchanged until reset() is called.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        fade_ms: float = 0.0,
        start_gain: float = 0.25,
    ):
        self.sample_rate = int(sample_rate or 48000)
        self.fade_ms = max(0.0, float(fade_ms or 0.0))
        self.start_gain = min(1.0, max(0.0, float(start_gain)))

        self.fade_samples = int(round((self.fade_ms / 1000.0) * self.sample_rate))
        self.pos = 0
        self.active = False

    @property
    def enabled(self) -> bool:
        return self.fade_samples > 0

    def reset(self) -> None:
        self.pos = 0
        self.active = False

    def _gain_for(self, frame_count: int) -> np.ndarray:
        return partial_linear_gain_ramp(
            start_pos=self.pos,
            length=frame_count,
            total_length=self.fade_samples,
            start_gain=self.start_gain,
            end_gain=1.0,
        )

    def _apply_gain(self, samples: np.ndarray, gain: np.ndarray) -> np.ndarray:
        if samples is None or len(samples) == 0:
            return samples

        original_dtype = samples.dtype

        out = ensure_float32(samples)
        out = sanitize_audio(out).copy()

        if out.ndim > 1:
            gain = gain.reshape(-1, 1)

        out *= gain

        if np.issubdtype(original_dtype, np.integer):
            info = np.iinfo(original_dtype)
            out = np.clip(out, info.min, info.max)

        return out.astype(original_dtype, copy=False)

    def apply(self, samples: np.ndarray) -> np.ndarray:
        (faded,) = self.apply_many(samples)
        return faded

    def apply_many(self, samples: np.ndarray, *extra_samples):
        """
        Apply one shared fade envelope to main RX audio plus optional extras.

        This lets main TX audio and Output-2 link audio fade together without
        advancing the fade state more than once per audio callback.
        """
        if not self.enabled:
            return (samples, *extra_samples)

        if samples is None or len(samples) == 0:
            return (samples, *extra_samples)

        if self.pos >= self.fade_samples:
            return (samples, *extra_samples)

        if not self.active:
            self.active = True
            self.pos = 0

        logging.info(
            "[RX Fade] Starting fade-in: fade_ms=%.1f, start_gain=%.2f, fade_samples=%d",
            self.fade_ms,
            self.start_gain,
            self.fade_samples,
        )

        frame_count = int(samples.shape[0])
        gain = self._gain_for(frame_count)

        faded_samples = self._apply_gain(samples, gain)

        faded_extras = []
        for extra in extra_samples:
            if extra is None or len(extra) == 0:
                faded_extras.append(extra)
                continue

            extra_count = int(extra.shape[0])
            if extra_count == frame_count:
                extra_gain = gain
            else:
                extra_gain = partial_linear_gain_ramp(
                    start_pos=self.pos,
                    length=extra_count,
                    total_length=self.fade_samples,
                    start_gain=self.start_gain,
                    end_gain=1.0,
                )

            faded_extras.append(self._apply_gain(extra, extra_gain))

        self.pos += frame_count
        if self.pos >= self.fade_samples:
            logging.info("[RX Fade] Fade-in complete")

        return (faded_samples, *faded_extras)


def build_rx_fade_in(config: dict, sample_rate: Optional[int] = None) -> RxFadeIn:
    audio_cfg = config.get("audio", {})

    resolved_sample_rate = int(
        sample_rate or audio_cfg.get("sample_rate") or config.get("sample_rate", 48000)
    )

    fade_ms = float(audio_cfg.get("rx_fade_in_ms", 75.0))
    start_gain = float(audio_cfg.get("rx_fade_in_start_gain", 0.30))

    fade = RxFadeIn(
        sample_rate=resolved_sample_rate,
        fade_ms=fade_ms,
        start_gain=start_gain,
    )

    if fade.enabled:
        logging.info(
            "[RX Fade] Enabled: %.1f ms, start_gain=%.2f, sample_rate=%d",
            fade.fade_ms,
            fade.start_gain,
            fade.sample_rate,
        )

    return fade
