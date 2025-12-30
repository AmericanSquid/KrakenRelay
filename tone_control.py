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
        duration = 0.12  # a little longer for clarity
        frequency = 523

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        #tone = np.sin(2 * np.pi * frequency * t)
        # "boop" = short down-chirp + quick decay
        f0 = self.config.config['repeater'].get('courtesy_tone_f_start', 520)
        f1 = self.config.config['repeater'].get('courtesy_tone_f_end', 390)

        k = (f1 - f0) / max(duration, 1e-6)  # Hz/sec
        phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
        tone = np.sin(phase)

        # quick decay so it doesn't feel like a lab tone
        tone *= np.exp(-5.0 * t / max(duration, 1e-6))

        # tiny fade to prevent click
        fade_len = int(sample_rate * 0.005)  # 5ms
        fade = np.linspace(0, 1, fade_len)
        tone[:fade_len] *= fade
        tone[-fade_len:] *= fade[::-1]

        logging.info(f"Generated courtesy tone at {frequency} Hz with volume {volume}")
        return (tone * volume * 32767).astype(np.int16)
        

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
