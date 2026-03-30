import numpy as np
import logging
from ..tone_vol import _tone_vol
from .constants import (
    COURTESY_FADE_MS,
    COURTESY_DECAY,
    COURTESY_TONE_F_START,
    COURTESY_TONE_F_END,
    PIP_DUR,
    PIP_GAP,
    TOT_TONE_DURATION,
    TOT_FADE_MS,
    PCM16_MAX,
)
from audio.primitives import (
    timebase,
    sine,
    chirp,
    decay,
    silence,
    fade,
    to_pcm16,
)

class ToneGenerator:
    def __init__(self, config):
        self.config = config

    def _sample_rate(self):
        cfg = self.config.config
        audio_cfg = cfg['audio']
        return audio_cfg['sample_rate']

    def generate_courtesy_tone(self):
        cfg = self.config.config
        repeater_cfg = cfg['repeater']

        volume = _tone_vol(repeater_cfg.get('courtesy_tone_volume', 100))
        sr = self._sample_rate()

        duration = PIP_DUR
        gap = PIP_GAP

        def pip(freq):
            t = timebase(duration, sr)
            tone = sine(freq, t)
            tone = decay(tone, t, COURTESY_DECAY, duration)
            tone = fade(tone, sr, COURTESY_FADE_MS)
            return tone

        tone = np.concatenate([
            pip(COURTESY_TONE_F_START),
            silence(int(sr * gap), np.float32),
            pip(COURTESY_TONE_F_END)
        ])

        logging.info(f"Generated courtesy tone {COURTESY_TONE_F_START}->{COURTESY_TONE_F_END} Hz at {volume} % volume")
        return to_pcm16(tone, volume, PCM16_MAX)

    def generate_tot_tone(self):
        tot_cfg = self.config.config['tot']
        volume=_tone_vol(tot_cfg.get('tot_volume', 100))
        sr = self._sample_rate()
        duration = TOT_TONE_DURATION

        t = timebase(duration, sr)
        freq = tot_cfg['tot_tone_freq']
        tone = sine(freq, t)
        tone = fade(tone, sr, TOT_FADE_MS)

        logging.info(f"Generated TOT tone {freq} Hz at {volume} % volume")

        return to_pcm16(tone, volume, PCM16_MAX)
