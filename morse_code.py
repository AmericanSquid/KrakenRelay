import numpy as np
import logging
import time

class MorseCode:
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

    def __init__(self, wpm=20, frequency=800, sample_rate=8000, volume=0.5):
        self.dot_length = int(1.2 / wpm * sample_rate)
        self.dash_length = self.dot_length * 3
        self.frequency = frequency
        self.sample_rate = sample_rate
        self.volume = volume

    def generate(self, text):
        t = np.arange(self.dot_length) / self.sample_rate
        dot = np.sin(2 * np.pi * self.frequency * t)
        dash = np.sin(2 * np.pi * self.frequency * np.arange(self.dash_length) / self.sample_rate)
        
        output = np.array([])
        space = np.zeros(self.dot_length)
        
        for char in text.upper():
            if char in self.MORSE_CODE:
                for symbol in self.MORSE_CODE[char]:
                    output = np.append(output, dot if symbol == '.' else dash)
                    output = np.append(output, space)
                output = np.append(output, space * 2)
                
        return (output * self.volume * 32767).astype(np.int16)
    
class ScheduleID:
    def __init__(self, config, tx_start_fn, tx_stop_fn, send_pcm_fn, tx_state_fn, set_skip_courtesy_fn):
        self.config = config
        self.morse = MorseCode(
            wpm=config.config['identification']['cw_wpm'],
            frequency=config.config['identification']['cw_pitch'],
            sample_rate=config.config['audio']['sample_rate'],
            volume=config.config['identification']['cw_volume']
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

            if should_id and self.post_tx and not self.sending_id:
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
                self.tx_start()
                cw_audio = self.morse.generate(callsign)
                self.send_pcm(cw_audio)
                self.set_skip_courtesy()
                self.tx_stop()
                logging.info(f"Sent CW ID: {callsign}")
        except Exception as e:
            logging.error(f"ID failed: {e}")
        finally:
            self.last_id_time = time.time()
            self.sending_id = False