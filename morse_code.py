import numpy as np
import logging
import time
from tone_control import _tone_vol

MORSE_CODE = {
    'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.',
    'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---',
    'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---',
    'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-',
    'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--',
    'Z': '--..', '0': '-----', '1': '.----', '2': '..---', '3': '...--',
    '4': '....-', '5': '.....', '6': '-....', '7': '--...', '8': '---..',
    '9': '----.'
}

class MorseCode:
    def __init__(self, wpm=20, frequency=800, sample_rate=8000, volume=0.5):
        self.dot_len = int(1.2 / wpm * sample_rate)
        self.dash_len = self.dot_len * 3
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.volume = volume
        self._intra = np.zeros(self.dot_len, dtype=np.int16)
        self._inter_char = np.zeros(self.dot_len * 2, dtype=np.int16)

    def tone(self, length):
        t = np.arange(length) / self.sample_rate
        return (np.sin(2 * np.pi * self.frequency * t) * self.volume * 32767).astype(np.int16)

    def silence(self, length):
        return np.zeros(length, dtype=np.int16)

    def generate_chunks(self, text, chunk_size):
        """Yield int16 PCM chunks suitable for real-time playback."""
        for char in text.upper():
            if char not in MORSE_CODE:
                continue
            for symbol in MORSE_CODE[char]:
                tone = self.tone(self.dot_len if symbol == "." else self.dash_len)
                yield from self._chunk(tone, chunk_size)
                # Intra-symbol space
                yield from self._chunk(self._intra, chunk_size)
            # Inter-letter space
            yield from self._chunk(self._inter_char, chunk_size)

    def _chunk(self, data, chunk_size):
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i + chunk_size]
            yield chunk
    
class ScheduleID:
    def __init__(self, controller, config, tx_start_fn, tx_stop_fn, send_pcm_fn, tx_state_fn, set_skip_courtesy_fn):
        self.config = config
        self.morse = MorseCode(
            wpm=config.config['identification']['cw_wpm'],
            frequency=config.config['identification']['cw_pitch'],
            sample_rate=config.config['audio']['sample_rate'],
            volume=_tone_vol(config.config['identification'].get('cw_volume', 100))
            )
        self.tx_start = tx_start_fn
        self.tx_stop = tx_stop_fn
        self.send_pcm = send_pcm_fn
        self.is_transmitting = tx_state_fn
        self.set_skip_courtesy = set_skip_courtesy_fn
        self.last_id_time = time.time()
        self.post_tx = False
        self.sending_id = False
        self.last_stop_time = time.time()
        self.cooldown = 0.25
        self.controller = controller

    def mark_post_tx(self):
        self.post_tx = True
        self.last_stop_time = time.time()
    
    def check_and_send(self):
        # === Idle & Post-TX CW ID === #
        if self.is_transmitting():
            return
        
        if self.config.config['identification']['cw_enabled']:
            interval = self.config.config['identification']['interval_minutes'] * 60
            should_id = time.time() - self.last_id_time > interval

            if should_id and (self.post_tx or not self.sending_id):
                if time.time() - self.last_stop_time > self.cooldown:
                    if self.post_tx:
                        logging.info("Sending CW ID after user transmission.")
                        self.post_tx = False
                    else:
                        logging.info("Sending CW ID while idle.")
                    
                    self.send_id()
    
    def send_id(self):
        if self.is_transmitting() or self.sending_id:
            logging.warning("Unable to Send ID: already transmitting")
            return
        
        self.sending_id = True
        try:
            callsign = self.config.config['identification']['callsign']
            if self.config.config['identification']['cw_enabled']:
                self.controller.start_cw_id(callsign)
                logging.info(f"Sent CW ID: {callsign}")
        except Exception as e:
            logging.error(f"ID failed: {e}")
        finally:
            self.last_id_time = time.time()
            self.sending_id = False
