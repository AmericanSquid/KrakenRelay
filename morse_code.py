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

    def generate_chunks(self, text, chunk_size):
        """Yield fixed-size int16 PCM chunks for real-time playback (timing-accurate)."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        buf = np.zeros(0, dtype=np.int16)

        def append(seg: np.ndarray):
            nonlocal buf
            if seg.size == 0:
                return
            buf = seg if buf.size == 0 else np.concatenate((buf, seg))

        def flush_full_chunks():
            nonlocal buf
            while buf.size >= chunk_size:
                out = buf[:chunk_size]
                buf = buf[chunk_size:]
                yield out

        for char in text.upper():
            if char not in MORSE_CODE:
                continue

            for symbol in MORSE_CODE[char]:
                # tone
                append(self.tone(self.dot_len if symbol == "." else self.dash_len))
                yield from flush_full_chunks()

                # intra-symbol gap (1 dot)
                append(self._intra)
                yield from flush_full_chunks()

            # inter-letter gap (2 dots here, because we already added 1 dot after last element)
            append(self._inter_char)
            yield from flush_full_chunks()

        # Pad ONLY ONCE at the end so timing doesn't get stretched per element
        if buf.size:
            pad = np.zeros(chunk_size - buf.size, dtype=np.int16)
            yield np.concatenate((buf, pad))

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
        now = time.time()
        # === Idle & Post-TX CW ID === #
        if self.is_transmitting():
            return
        
        if now - self.last_stop_time < self.cooldown:
            return

        if not self.config.config['identification']['cw_enabled']:
            return

        interval = self.config.config['identification']['interval_minutes'] * 60
        should_id = time.time() - self.last_id_time > interval

        if not should_id:
            return

        if self.post_tx:
            logging.info("Sending CW ID after user transmission.")
            self.post_tx = False
        elif not self.sending_id:
            logging.info("Sending CW ID while idle.")
        else:
            return

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
