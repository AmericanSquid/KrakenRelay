import numpy as np
import logging

def _tone_vol(v, safe_max=0.5):
    """
    Backwards compatible:
      - old style: 0.0..1.0   (1.0 == 100%)
      - new style: 0..100     (100 == 100%)
      - strings:   "50", "50%"
    Then maps 100% -> safe_max (your old 0.5).
    """
    if v is None:
        return safe_max

    if isinstance(v, str):
        s = v.strip()
        if s.endswith("%"):
            try:
                pct = float(s[:-1])
            except ValueError:
                return safe_max
            pct = max(0.0, min(100.0, pct))
            return (pct / 100.0) * safe_max
        try:
            v = float(s)
        except ValueError:
            return safe_max

    try:
        v = float(v)
    except (TypeError, ValueError):
        return safe_max

    # interpret <=1.0 as normalized; >1.0 as percent
    pct = (v * 100.0) if v <= 1.0 else v
    pct = max(0.0, min(100.0, pct))
    return (pct / 100.0) * safe_max

class ToneGenerator:
    def __init__(self, config):
        self.config = config

    def generate_courtesy_tone(self):
        volume=_tone_vol(self.config.config['repeater'].get('courtesy_tone_volume', 100))
        sample_rate = self.config.config['audio']['sample_rate']

        f1 = self.config.config['repeater'].get('courtesy_tone_f_start', 700)
        f2 = self.config.config['repeater'].get('courtesy_tone_f_end', 520)
        pip_dur = 0.040
        gap = 0.025

        def pip(freq):
            t = np.linspace(0, pip_dur, int(sample_rate * pip_dur), endpoint=False)
            x = np.sin(2 * np.pi * freq * t)

            a = int(sample_rate * 0.004)
            r = int(sample_rate * 0.010)
            x[:a] *= np.linspace(0, 1, a)
            x[-r:] *= np.linspace(1, 0, r)

            return x

        tone = np.concatenate([
            pip(f1),
            np.zeros(int(sample_rate * gap)),
            pip(f2)
        ])

        pcm = tone * volume
        pcm = np.clip(pcm, -1.0, 1.0) 

        logging.info(f"Generated courtesy tone at {f1} Hz, {f2} Hz with volume {volume}")
        return (pcm * 32767).astype(np.int16)
        

    def generate_tot_tone(self):
        volume=_tone_vol(self.config.config['tot'].get('tot_volume', 100))
        sample_rate = self.config.config['audio']['sample_rate']
        duration = 1  # Duration of TOT warning tone in seconds
    
        # Generate a pure tone at configured frequency
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = np.sin(2 * np.pi * self.config.config['tot']['tot_tone_freq'] * t)
    
        # Apply fade-in and fade-out
        fade_len = int(sample_rate * 0.01)  # 10ms fade
        fade = np.linspace(0, 1, fade_len)
        tone[:fade_len] *= fade
        tone[-fade_len:] *= fade[::-1]
    
        # Convert to 16-bit PCM
        logging.info(f"Generated TOT tone at {self.config.config['tot']['tot_tone_freq']}Hz at volume {volume}")
        return (tone * volume * 32767).astype(np.int16)
        

class TonePlayer:
    def __init__(self, config, send_pcm_callable, tx_state_callable, tone_generator):
        self.config = config
        self._send_pcm = send_pcm_callable
        self.is_transmitting = tx_state_callable
        self.generator = tone_generator

        self.courtesy_tone = self.generator.generate_courtesy_tone()
        self.tot_tone = self.generator.generate_tot_tone()
    def play_courtesy_tone(self):
        if self.config.config['repeater']['courtesy_tone_enabled']:
            self._send_pcm(self.courtesy_tone)
            logging.info("Played courtesy tone")

    def play_tot_tone(self):
        if self.is_transmitting():
            self._send_pcm(self.tot_tone)
            logging.info("Played TOT warning tone") 
