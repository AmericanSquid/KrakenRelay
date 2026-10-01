"""CW frames shared by the live-audio mixer and standalone ID playback."""

import numpy as np


class CWPlayback:
    def __init__(self, state, config):
        self.state = state
        self.config = config
        self.voice_emitted = False
        self._cw_gain = None
        self._cw_target = None
        self._cw_remaining = 0
        self._output_gain = 1.0

    def start(self, generator):
        self._cw_gain = None
        self._cw_target = None
        self._cw_remaining = 0
        self.state.cw_gen = generator

    def begin_iteration(self):
        self.voice_emitted = False

    def clear(self):
        self.state.cw_gen = None
        self.state.cw_next_t = None
        self._cw_gain = None
        self._cw_target = None
        self._cw_remaining = 0

    def _take_frame(self):
        generator = self.state.cw_gen
        if generator is None:
            return None
        try:
            return next(generator)
        except StopIteration:
            self.clear()
            return None

    def _sample_rate(self):
        return max(1.0, float(self.config.config.get("audio", {}).get("sample_rate", 48000)))

    def _scale_cw(self, samples, target):
        samples = np.asarray(samples, dtype=np.float32)
        if self._cw_gain is None:
            self._cw_gain = target
        # Move the fader over 50 ms when traffic starts/stops or settings change.
        if target != self._cw_target:
            self._cw_target = target
            self._cw_remaining = max(1, int(self._sample_rate() * 0.050))
            self._cw_step = (target - self._cw_gain) / self._cw_remaining
        progress = np.minimum(np.arange(1, samples.size + 1), self._cw_remaining)
        gains = self._cw_gain + self._cw_step * progress
        self._cw_remaining = max(0, self._cw_remaining - samples.size)
        if samples.size:
            self._cw_gain = float(gains[-1]) if self._cw_remaining else target
        return samples * gains

    def _protect_output(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if not samples.size:
            return samples
        peak = float(np.max(np.abs(samples)))
        required_gain = min(1.0, 32767.0 / peak) if peak else 1.0
        # Scale the whole block on overload, preserving the summed waveform.
        # Recover over 100 ms instead of repeatedly flattening its peaks.
        recovery = 1.0 - np.exp(-samples.size / (self._sample_rate() * 0.100))
        gain = min(required_gain, self._output_gain + (1.0 - self._output_gain) * recovery)
        self._output_gain = gain
        return samples * gain

    def take(self):
        """Play remaining CW alone, smoothly returning to its normal level."""
        cw = self._take_frame()
        if cw is None:
            return None
        return self._protect_output(self._scale_cw(cw, 1.0))

    def mix_voice(self, samples):
        self.voice_emitted = True
        cw = self._take_frame()
        if cw is None:
            # Only retain protection on voice alone while an overload releases.
            return samples if self._output_gain == 1.0 else self._protect_output(samples)

        if len(cw) != len(samples):
            raise ValueError("CW and user audio chunks must have equal lengths")

        voice = np.asarray(samples, dtype=np.float32)
        cw = np.asarray(cw, dtype=np.float32)
        cfg = self.config.config
        id_cfg = cfg.get("identification", {})
        ratio = (
            np.clip(float(id_cfg.get("cw_mix_ratio", 50)), 0.0, 100.0) / 100.0
        )
        attenuation_db = np.clip(
            float(id_cfg.get("cw_mix_attenuation_db", 6)), 0.0, 30.0
        )

        # Fixed mixer gain: voice loudness never moves the CW fader.
        cw_gain = ratio * 10.0 ** (-attenuation_db / 20.0)
        mixed = voice + self._scale_cw(cw, cw_gain)
        return self._protect_output(mixed)
