
import numpy as np
import logging

class ToneGenerator:
    def __init__(self, config):
        self.config = config

    def generate_courtesy_tone(self):
        volume = self.config.config['repeater']['courtesy_tone_volume']
        sample_rate = self.config.config['audio']['sample_rate']
        duration = 0.12  # a little longer for clarity
        frequency = 659

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = np.sin(2 * np.pi * frequency * t)

        fade_len = int(sample_rate * 0.01)
        fade = np.linspace(0, 1, fade_len)
        tone[:fade_len] *= fade
        tone[-fade_len:] *= fade[::-1]

        logging.info(f"Generated courtesy tone at {frequency} Hz with volume {volume}")
        return (tone * volume * 32767).astype(np.int16)
        

    def generate_tot_tone(self):
        volume = self.config.config['tot']['tot_volume']
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
        if self.is_transmitting:
            self._send_pcm(self.tot_tone)
            logging.info("Played TOT warning tone") 