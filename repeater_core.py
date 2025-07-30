import numpy as np
import time
import threading
import logging
from scipy import signal
from morse_code import MorseCode
from audio_manager import ToneDetector, AudioBuffer
from mumble_interface import MumbleLink
from ptt_controller import CM108PTT

class RepeaterController:

    #------------------------#
    # Initialization & Setup #
    #------------------------# 
    def __init__(self, input_device, output_device, config, audio_manager):
        # Core Configuration
        self.config = config
        self.audio_manager = audio_manager
        self.input_device = input_device
        self.output_device = output_device

        # PTT Setup
        ptt_cfg = config.config.get('ptt', {})
        self.ptt_mode = ptt_cfg.get('mode', 'NONE').upper()

        if self.ptt_mode == 'CM108':
            self.ptt = CM108PTT(
                device=ptt_cfg.get('device_path', '/dev/hidraw0'),
                pin=int(ptt_cfg.get('gpio_pin', 3))
            )
            logging.info(f"PTT mode set to CM108 (device={self.ptt.device}, pin={self.ptt.pin})")
        else:
            self.ptt = None
            logging.info("PTT mode set to VOX (no GPIO control)")

        # Audio Stream Set Up
        self.input_stream = None
        self.output_stream = None

        # State Tracking & Timing
        self.running = False
        self.transmitting = False
        self.tot_timer = 0
        self.tot_locked = False
        self.transmission_start_time = None
        self.last_audio_time = time.time()
        self.last_transmission = time.time()
        self.last_id_time = time.time()
        self.current_rms = 0
        self.mumble: MumbleLink | None = None
        
        # Initialize audio buffers
        self.input_buffer = AudioBuffer()
        self.output_buffer = AudioBuffer()

        # Tone Detector Setup (broken)
        self.tone_detector = ToneDetector(
            sample_rate=config.config['audio']['sample_rate'],
            target_freq=config.config['repeater']['pl_tone_freq'],
            bandwidth=3.0,
            threshold=config.config['repeater']['pl_threshold'],
            alpha=0.3
        )

        # CW IDer Setup
        self.morse = MorseCode(
            wpm=config.config['repeater']['cw_wpm'],
            frequency=config.config['repeater']['cw_pitch'],
            sample_rate=config.config['audio']['sample_rate']
        )
        # Pregenerate CW audio once using the callsign
        self.cw_audio = self.morse.generate(self.config.config['repeater']['callsign'])
        
        #Generate Tones & Setup Streams
        self.generate_courtesy_tone()
        self.generate_tot_tone()
        self.setup_audio_streams()
        self._init_mumble_link()
        logging.info("RepeaterController initialized")

    def setup_audio_streams(self):
        try:
            audio_config = self.config.config['audio']
            sample_rate = audio_config['sample_rate']
            chunk_size = audio_config['chunk_size']
            
            logging.info(f"Setting up audio streams with rate: {sample_rate}")
            
            self.input_stream = self.audio_manager.create_input_stream(
                device_index=self.input_device,
                rate=sample_rate,
                chunk=chunk_size
            )
            
            self.output_stream = self.audio_manager.create_output_stream(
                device_index=self.output_device,
                rate=sample_rate,  # Use same rate as input
                chunk=chunk_size
            )
            
            logging.info(f"Audio streams setup complete: rate={sample_rate}, chunk={chunk_size}")
            
        except Exception as e:
            logging.error(f"Failed to setup audio streams: {e}")
            raise
   
    # ---------- Mumble plumbing ----------
    def _init_mumble_link(self):
        cfg = self.config.config['mumble']
        if not cfg.get('enabled', False):
            self.mumble = None
            return
        try:
            self.mumble = MumbleLink(cfg)
            logging.info("Mumble link initialised (mode=%s)", cfg['mode'])
        except Exception as exc:
            self.mumble = None
            logging.error("Could not start Mumble link: %s", exc, exc_info=True)

    def reload_mumble_link(self):
        """Call this from the Settings dialog after the user presses Save."""
        if self.mumble:
            self.mumble.disconnect()
        self._init_mumble_link()

    #---------------#
    # Audio Filters #
    #---------------#
    def apply_highpass_filter(self, samples):
        if not self.config.config['audio']['highpass_enabled']:
            return samples
        nyquist = self.config.config['audio']['sample_rate'] / 2
        cutoff = self.config.config['audio']['highpass_cutoff']
        b, a = signal.butter(4, cutoff/nyquist, btype='high')
        return signal.filtfilt(b, a, samples)

    def apply_noise_gate(self, samples):
        if not self.config.config['audio']['noise_gate_enabled']:
            return samples
        threshold = self.config.config['audio']['noise_gate_threshold']
        samples[abs(samples) < threshold] = 0
        return samples

    #----------------#
    # System Control #
    #----------------#
    def start(self):
        if self.running:
            logging.warning("Repeater is already running.")
            return
        self.running = True
        self.audio_thread = threading.Thread(target=self.audio_loop)
        self.audio_thread.start()
        logging.info("Repeater controller started")

        logging.info(f"🧵 Active threads after start: {threading.active_count()}")
        for t in threading.enumerate():
            logging.info(f"  • {t.name}")

    def audio_loop(self):
        while self.running:
            try:
                self.process_audio()
                #if time.time() - self.last_id_time > self.config.config['identification']['interval_minutes'] * 60:
                #    self.send_id()
                            # If CW ID is enabled and repeater is idle
                if self.config.config['identification']['cw_enabled']:
                    interval = self.config.config['identification']['interval_minutes'] * 60
                    if not self.transmitting and time.time() - self.last_id_time > interval:
                        logging.info("Sending CW ID while idle")
                        self.start_transmission()
                        time.sleep(0.2)
                        self.play_audio_chunks(self.cw_audio)
                        time.sleep(0.2)
                        self.transmitting = False
                        self.transmission_start_time = None

                        fade_duration = 0.1
                        sample_rate = self.config.config['audio']['sample_rate']
                        chunk_size = self.config.config['audio']['chunk_size']
                        num_fade_chunks = int((fade_duration * sample_rate) // chunk_size)
        
                        for i in range(num_fade_chunks):
                          silent_chunk = (np.zeros(chunk_size)).astype(np.int16).tobytes()
                          self.output_stream.write(silent_chunk)
                          time.sleep(chunk_size / sample_rate)
        
                        self.transmitting = False
                        self.tot_timer = 0
                        self.last_transmission = time.time()
                        self.ptt.unkey() 

                        self.last_id_time = time.time()
                        logging.info("Idle ID complete. Transmitter unkeyed.")

            except Exception as e:
                logging.error(f"Error in audio loop: {e}")

    def cleanup(self):
        logging.info("Cleaning up repeater controller...")
        self.running = False # Stop the loop

        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=1.0) # Ensure thread stops safely

        if self.input_stream:
            try:
                self.input_stream.stop_stream()
                self.input_stream.close()
            except Exception as e:
                logging.error(f"Error closing input stream: {e}")

        if self.output_stream:
            try:
                self.output_stream.stop_stream()
                self.output_stream.close()
            except Exception as e:
                logging.error(f"Error closing output stream: {e}")
        
        if self.ptt:
            self.ptt.unkey()
        
        if self.mumble:
            self.mumble.disconnect()
            self.mumble = None  # clear reference

        logging.info("Repeater controller stopped and cleaned up")

        logging.info(f"🧵 Active threads after cleanup: {threading.active_count()}")
        for t in threading.enumerate():
            logging.info(f"  • {t.name}")

    #-----------------------#
    # Core Audio Processing #
    #-----------------------#
    def check_squelch(self, samples):
        db_level = 20 * np.log10(np.sqrt(np.mean(samples**2)) / 32767)
        return db_level > self.config.config['audio']['squelch_threshold']
    
    def calculate_db_level(self, samples):
        rms = np.sqrt(np.mean(samples**2))
        if rms > 0:
            db_level = 20 * np.log10(rms / 32767)
        else:
            db_level = -60  # Set minimum dB level for silence
        return db_level

    def process_audio(self):
        data = self.input_stream.read(self.config.config['audio']['chunk_size'], exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        self.current_rms = self.calculate_db_level(samples)

        m_cfg = self.config.config['mumble']
        m_frame = None  # ← FIX: always define first!

        if self.mumble and m_cfg.get('direction') != 'rf_to_mumble':
            m_frame = self.mumble.read_frame()
        if m_frame is not None:
            self.handle_transmission(m_frame.astype(np.float32))
            self.output_stream.write(m_frame.astype(np.int16).tobytes())
        
        # Check squelch first
        if self.check_squelch(samples):
            # Process audio chain
            samples = self.apply_noise_gate(samples)
            samples = self.apply_highpass_filter(samples)
            
            # Apply input gain
            samples *= 10 ** (self.config.config['audio']['input_gain'] / 20)
            
            # Update meter with processed audio
            self.current_rms = np.sqrt(np.mean(samples**2))
            
            # Check for PL tone
            if self.tone_detector.detect_tone(samples):
                if not self.transmitting:
                    freq = self.config.config['repeater']['pl_tone_freq']   # pull from YAML
                    logging.info(f"{freq} Hz PL tone detected")
                self.handle_transmission(samples)
        else:
            self.current_rms = 0
            if self.transmitting and (time.time() - self.last_audio_time) > self.config.config['repeater']['tail_time']:
                logging.info("Silence persists beyond tail time. Stopping transmission.")
                self.stop_transmission()            

    def handle_transmission(self, samples):
        if self.config.config['tot']['tot_enabled'] and self.transmission_start_time:
            elapsed_time = time.time() - self.transmission_start_time
            self.tot_timer = elapsed_time  # Update timer with actual elapsed time
            
            if elapsed_time >= self.config.config['tot']['tot_time']:
                logging.warning(f"TOT limit reached at {self.tot_timer:.1f} seconds")
                self.play_tot_tone()
                if self.config.config['tot']['tot_lockout_enabled']:
                    logging.info("TOT lockout activated")
                    self.tot_locked = True
                    time.sleep(self.config.config['tot']['tot_lockout_time'])
                    self.tot_locked = False
                    logging.info("TOT lockout released")
                self.stop_transmission()
                return   

        if not self.transmitting:
            self.start_transmission()

        # Apply output gain
        output_samples = samples * 10 ** (self.config.config['audio']['output_gain'] / 20)
        self._send_pcm(output_samples.astype(np.int16))
        
        self.last_audio_time = time.time()

    # ---------- unified TX funnel ----------   ← paste helper here
    def _send_pcm(self, pcm: np.ndarray):
        """
        Forward audio to the RF transmitter, and—unless the user picked
        'mumble_to_rf' (RX-only)—also pipe it to Mumble.
        """
        # Fix any NaNs/infs
        if np.isnan(pcm).any() or np.isinf(pcm).any():
            logging.warning("NaN/Inf in audio! Zeroing bad samples.")
            pcm = np.nan_to_num(pcm, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip to int16 range
        pcm = np.clip(pcm, -32768, 32767).astype(np.int16)
        cfg = self.config.config['mumble']
        if self.mumble and cfg.get('direction') != 'mumble_to_rf':
            self.mumble.write_frame(pcm)
        self.output_stream.write(pcm.tobytes())

    #----------------#
    # Tone Functions #
    #----------------#
    def send_id(self):
        try:
            callsign = self.config.config['repeater']['callsign']
            if self.config.config['identification']['cw_enabled']:
                self.start_transmission()
                time.sleep(0.2)
                cw_audio = self.morse.generate(callsign)
                self.play_audio_chunks(cw_audio)
                time.sleep(0.2)
                self.stop_transmission()
                logging.info(f"Sent CW ID: {callsign}")
        except Exception as e:
            logging.error(f"ID failed: {e}")
        finally:
            self.last_id_time = time.time()

    def generate_courtesy_tone(self):
        sample_rate = self.config.config['audio']['sample_rate']
        duration = 0.15  # a little longer for clarity
        frequency = 700

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        tone = np.sin(2 * np.pi * frequency * t)

        fade_len = int(sample_rate * 0.01)
        fade = np.linspace(0, 1, fade_len)
        tone[:fade_len] *= fade
        tone[-fade_len:] *= fade[::-1]

        self.courtesy_tone = (tone * 32767).astype(np.int16)
        logging.info(f"Generated courtesy tone at {frequency} Hz")

    def play_courtesy_tone(self):
        if self.config.config['repeater']['courtesy_tone_enabled']:
            self._send_pcm(self.courtesy_tone)
            logging.info("Played courtesy tone")

    def generate_tot_tone(self):
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
        self.tot_tone = (tone * 32767).astype(np.int16)
        logging.info(f"Generated TOT tone at {self.config.config['tot']['tot_tone_freq']}Hz")

    def play_tot_tone(self):
        if self.transmitting:
            self._send_pcm(self.tot_tone)
            logging.info("Played TOT warning tone") 

    #----------------------#
    # Transmission Control #
    #----------------------#

    def start_transmission(self):
        if not self.tot_locked:
            if time.time() - self.last_transmission > self.config.config['repeater']['anti_kerchunk_time']:
                self.transmitting = True
                self.transmission_start_time = time.time() # Set start time
                self.tot_timer = 0

            if self.ptt_mode == 'CM108' and self.ptt:
                self.ptt.key()
            else:
                time.sleep(self.config.config['repeater']['carrier_delay'])

                logging.info("Starting transmission")

    def stop_transmission(self):
        self.transmitting = False
        self.transmission_start_time = None

        if self.config.config['repeater']['courtesy_tone_enabled']:
            time.sleep(0.1)
            self.play_courtesy_tone()
            time.sleep(0.2)
        
        fade_duration = 0.1
        sample_rate = self.config.config['audio']['sample_rate']
        chunk_size = self.config.config['audio']['chunk_size']
        num_fade_chunks = int((fade_duration * sample_rate) // chunk_size)
        
        for i in range(num_fade_chunks):
            silent_chunk = (np.zeros(chunk_size)).astype(np.int16).tobytes()
            self.output_stream.write(silent_chunk)
            time.sleep(chunk_size / sample_rate)
        
        self.transmitting = False
        self.tot_timer = 0
        self.last_transmission = time.time()
        logging.info("Transmission stopped with fade-out")

        # Check if it’s time to ID
        if self.config.config['identification']['cw_enabled']:
            interval = self.config.config['identification']['interval_minutes'] * 60
            if time.time() - self.last_id_time > interval:
                logging.info("Sending CW ID after user transmission")
                self.start_transmission()
                time.sleep(0.2)
                self.play_audio_chunks(self.cw_audio)
                time.sleep(0.2)
                self.transmitting = False
                self.last_id_time = time.time()
                logging.info("CW ID complete")

        if self.ptt_mode == 'CM108' and self.ptt:
            self.ptt.unkey()

    def play_audio_chunks(self, audio):
        chunk_size = self.config.config['audio']['chunk_size']
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            self._send_pcm(chunk)
