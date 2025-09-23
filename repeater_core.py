import numpy as np
import time
import threading
import logging
from scipy import signal
from morse_code import MorseCode
from ptt_controller import CM108PTT
import os

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
        self.output_stream_2 = None # Optional mirrored output

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
        
        # CW IDer Setup
        self.morse = MorseCode(
            wpm=config.config['identification']['cw_wpm'],
            frequency=config.config['identification']['cw_pitch'],
            sample_rate=config.config['audio']['sample_rate'],
            volume=config.config['identification']['cw_volume']
        )
        # Pregenerate CW audio once using the callsign
        self.cw_audio = self.morse.generate(self.config.config['identification']['callsign'])
        
        # Thread Safety
        self.ptt_lock = threading.Lock()

        # Generate Tones & Setup Streams
        self.generate_courtesy_tone()
        self.generate_tot_tone()
        self.setup_audio_streams()
        logging.info("RepeaterController initialized")

    def setup_audio_streams(self):
        try:
            audio_config = self.config.config['audio']
            sample_rate = audio_config['sample_rate']
            chunk_size = audio_config['chunk_size']
            
            logging.info(f"Setting up audio streams with rate: {sample_rate}")
            
            if audio_config.get('dual_output', False):
                dev2_arg = audio_config.get('output_device_2', None)
                second_index = None

            # Resolve second device
            if dev2_arg in (None, ""):
                try:
                    second_index = int(self.audio_manager.pa.get_default_output_device_info()['index'])
                except Exception as e:
                    logging.warning(f"Dual output: could not get default output device: {e}")
                    second_index = None
            elif isinstance(dev2_arg, int) or (isinstance(dev2_arg, str) and dev2_arg.isdigit()):
                second_index = int(dev2_arg)
            else:
                second_index = self.audio_manager.find_device_by_name(str(dev2_arg))

            # Avoid duplicating the primary device
            if second_index is None:
                logging.warning("Dual output requested but no valid output_device_2 resolved; continuing single-output.")
            elif int(second_index) == int(self.output_device):
                logging.info("Dual output resolved to the same device as primary; skipping second stream.")
            else:
                try:
                    self.output_stream_2 = self.audio_manager.create_output_stream(
                        device_index=second_index,
                        rate=audio_config['sample_rate'],
                        chunk=audio_config['chunk_size']
                    )
                    logging.info(f"Dual output enabled on device index {second_index}")
                except Exception as e:
                    logging.warning(f"Dual output: failed to open second stream (idx={second_index}): {e}")
                    self.output_stream_2 = None


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
        consecutive_errors = 0
        max_errors = 5
        max_backoff = 0.5
        fatal_reason = 5.0
        
        logging.info("[Repeater] Audio thread started.")

        while self.running:
            try:
                self.process_audio()

                # === Idle CW ID === #
                if self.config.config['identification']['cw_enabled']:
                    interval = self.config.config['identification']['interval_minutes'] * 60
                    if not self.transmitting and time.time() - self.last_id_time > interval:
                        logging.info("Sending CW ID while idle")
                        self.start_transmission()
                        self.play_audio_chunks(self.cw_audio)
                        self.transmitting = False
                        self.transmission_start_time = None

                        fade_duration = 0.1
                        sample_rate = self.config.config['audio']['sample_rate']
                        chunk_size = self.config.config['audio']['chunk_size']
                        num_fade_chunks = int((fade_duration * sample_rate) // chunk_size)
        
                        for i in range(num_fade_chunks):
                            silent_chunk = (np.zeros(chunk_size)).astype(np.int16).tobytes()
                            try:
                                self.output_stream.write(silent_chunk)
                            except Exception as e:
                                logging.debug(f"Silence write (primary) failed (non-fatal): {e}")
                            if self.output_stream_2:
                                try:
                                    self.output_stream_2.write(silent_chunk)
                                except Exception as e:
                                    logging.debug(f"Silence write (secondary) failed (non-fatal): {e}")
                            time.sleep(chunk_size / sample_rate)
            
                        self.transmitting = False
                        self.tot_timer = 0
                        self.last_transmission = time.time()
                        self.safe_ptt_unkey()
                        self.last_id_time = time.time()
                        logging.info("Idle ID complete. Transmitter unkeyed.")

                if consecutive_errors:
                    logging.info("[Repeater] Audio loop recovered after %d error(s).", conseuctive_errors)
                    consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logging.error(f"Error in audio loop: {e}")

                # Backoff before retrying
                backoff = min(0.2 * (2 ** (consecutive_errors - 1)), max_backoff)
                time.sleep(backoff)

                # Too many consecutive errors? Time to bail.
                if consecutive_errors >= max_errors:
                    fatal_reason = f"{consecutive_errors} consecutive audio loop errors"
                    break

        # === CLEAN EXIT ===
        if not self.running and not fatal_reason:
            logging.info("[Repeater] Audio thread stopping (requested).")
        else:
            logging.critical(f"[Repeater] Audio loop exiting due to: {fatal_reason or 'unknown fatal error'}")

        self.running = False

        try:
            if getattr(self, "transmitting", False):
                try:
                    self.stop_transmission()
                except Exception:
                    self.safe_ptt_unkey()
            else:
                self.safe_ptt_unkey()
        except Exception:
            logging.exception("[Repeater] Error while unkeying during shutdown.")

        logging.info("[Repeater] Audio thread exited.")

    def cleanup(self):
        logging.info("Cleaning up repeater controller...")
        self.running = False # Stop the loop

        if hasattr(self, 'audio_thread'):
            self.audio_thread.join(timeout=1.0) # Ensure thread stops safely
           
            if self.audio_thread.is_alive():
                logging.warning("[Repeater] Audio thread did not terminate after join().")
            else:
                logging.info("[Repeater] Audio thread stopped cleanly.") 

        logging.info("[Repeater] Cleaning up audio streams.")

        try:
            if self.input_stream:
                self.input_stream.stop_stream()
                self.input_stream.close()
        except Exception as e:
            logging.warning(f"Failed to stop/close input stream: {e}")

        try:
            if self.output_stream:
                self.output_stream.stop_stream()
                self.output_stream.close()
        except Exception as e:
            logging.warning(f"Failed to stop/close output stream: {e}")

        try:
            if self.output_stream_2:
                self.output_stream_2.stop_stream()
                self.output_stream_2.close()
        except Exception as e:
            logging.warning(f"Failed to stop/close output stream 2: {e}")

        self.audio_manager.cleanup()
        self.safe_ptt_unkey()
        
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
        chunk  = self.config.config['audio']['chunk_size']

        # Check squelch first
        if self.check_squelch(samples):
            # Process audio chain
            samples = self.apply_highpass_filter(samples)
            
            self.current_rms = np.sqrt(np.mean(samples**2))
            
            if not self.transmitting:
                logging.info("Squelch opened - Starting Tx")
            self.handle_transmission(samples)
        else:
            self.current_rms = 0
            if self.transmitting and (time.time() - self.last_audio_time) > self.config.config['repeater']['tail_time']:
                logging.info("Silence persists beyond tail time. Stopping transmission.")
                self.stop_transmission()
            else:
                # 🚨 We're in the tail hang time — KEEP AUDIO FLOWING
                silent_chunk = np.zeros(self.config.config['audio']['chunk_size'], dtype=np.int16)
                try:
                    self.output_stream.write(silent_chunk.tobytes())            
                except Exception as e:
                    logging.debug(f"Silence write (primary) failed (non-fatal): {e}")
                if self.output_stream_2:
                    try:
                        self.output_stream_2.write(silent_chunk)
                    except Exception as e:
                        logging.debug(f"Silence write (secondary) failed (non-fatal): {e}")
    def handle_transmission(self, samples):
        if self.config.config['tot']['tot_enabled'] and self.transmission_start_time:
            elapsed_time = time.time() - self.transmission_start_time
            self.tot_timer = elapsed_time
            
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

        self._send_pcm(samples.astype(np.int16))
        
        self.last_audio_time = time.time()

    # ---------- unified TX funnel ----------   ← paste helper here
    def _send_pcm(self, pcm: np.ndarray):
        # Fix any NaNs/infs
        if np.isnan(pcm).any() or np.isinf(pcm).any():
            logging.warning("NaN/Inf in audio! Zeroing bad samples.")
            pcm = np.nan_to_num(pcm, nan=0.0, posinf=0.0, neginf=0.0)
        # Clip to int16 range
        pcm = np.clip(pcm, -32768, 32767).astype(np.int16)

        data = pcm.tobytes()
        # Primary (unchanged, but wrapped)
        try:
            self.output_stream.write(data)
        except Exception as e:
            logging.warning(f"Primary output write failed: {e}")

        # Secondary (best-effort)
        if self.output_stream_2:
            try:
                self.output_stream_2.write(data)
            except Exception as e:
                logging.warning(f"Secondary output write failed — disabling dual output: {e}")
                try:
                    self.output_stream_2.stop_stream()
                    self.output_stream_2.close()
                except Exception:
                    pass
                self.output_stream_2 = None

    #----------------#
    # Tone Functions #
    #----------------#
    def send_id(self):
        try:
            callsign = self.config.config['identification']['callsign']
            if self.config.config['identification']['cw_enabled']:
                self.start_transmission()
                cw_audio = self.morse.generate(callsign)
                self.play_audio_chunks(cw_audio)
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

        self.courtesy_tone = (tone * 0.5 * 32767).astype(np.int16)
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

            if self.ptt_mode == 'CM108':
                self.safe_ptt_key()
            else:
                threading.Timer(self.config.config['repeater']['carrier_delay'], lambda: logging.info("Carrier delay elapsed")).start()
                logging.info("Starting transmission")

    def stop_transmission(self):
        # Prime ALSA before tone playback
        b = np.zeros(self.config.config['audio']['chunk_size'], dtype=np.int16).tobytes()
        try:
            self.output_stream.write(b)
        except Exception as e:
            logging.debug(f"ALSA prime (primary) failed: {e}")
        if self.output_stream_2:
            try:
                self.output_stream_2.write(b)
            except Exception as e:
                logging.debug(f"ALSA prime (secondary) failed: {e}")

        if self.config.config['repeater']['courtesy_tone_enabled']:
            self.play_courtesy_tone()

        fade_duration = 0.1
        sample_rate = self.config.config['audio']['sample_rate']
        chunk_size = self.config.config['audio']['chunk_size']
        num_fade_chunks = int((fade_duration * sample_rate) // chunk_size)
        
        start = time.time()
        for i in range(num_fade_chunks):
            b = np.zeros(chunk_size, dtype=np.int16).tobytes()
            try:
                self.output_stream.write(b)
            except Exception as e:
                logging.debug(f"Fade-out write (primary) failed: {e}")
            if self.output_stream_2:
                try:
                    self.output_stream_2.write(b)
                except Exception as e:
                    logging.debug(f"Fade-out write (secondary) failed: {e}")
            target_time = start + (i + 1) * (chunk_size / sample_rate)
            now = time.time()
            delay = target_time - now
            if delay > 0:
                time.sleep(delay)
        
        self.transmitting = False
        self.transmission_start_time = None
        self.tot_timer = 0
        self.last_transmission = time.time()
        logging.info("Transmission stopped with fade-out")

        # Check if it’s time to ID
        if self.config.config['identification']['cw_enabled']:
            interval = self.config.config['identification']['interval_minutes'] * 60
            if time.time() - self.last_id_time > interval:
                logging.info("Sending CW ID after user transmission")
                self.start_transmission()
                self.play_audio_chunks(self.cw_audio)
                self.transmitting = False
                self.last_id_time = time.time()
                logging.info("CW ID complete")

        if self.ptt_mode == 'CM108':
            self.safe_ptt_unkey()

    def play_audio_chunks(self, audio):
        chunk_size = self.config.config['audio']['chunk_size']
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            self._send_pcm(chunk)
    
    ################
    # Keying Logic #
    ################

    def safe_ptt_key(self):
        try:
            if self.ptt and getattr(self.ptt, "working", True):
                self.ptt.key()
                return True
            else:
                logging.warning("PTT missing or dead—skipping key() or using VOX fallback.")
                if self.ptt_mode != "VOX":
                    self.ptt_fallback = True
                self.ptt_mode = "VOX"
                return False
        except Exception as e:
            logging.error(f"RepeaterCore: PTT key error: {e}")
            self.ptt = None
            if self.ptt_mode != "VOX":
                self.ptt_fallback = True
            self.ptt_mode = "VOX"
            return False

    def safe_ptt_unkey(self):
        try:
            if self.ptt and getattr(self.ptt, "working", True):
                self.ptt.unkey()
                return True
            else:
                logging.warning("PTT missing or dead—skipping unkey() or using VOX fallback.")
                if self.ptt_mode != "VOX":
                    self.ptt_fallback = True
                self.ptt_mode = "VOX"
                return False
        except Exception as e:
            logging.error(f"RepeaterCore: PTT unkey error: {e}")
            self.ptt = None
            if self.ptt_mode != "VOX":
                self.ptt_fallback = True
            self.ptt_mode = "VOX"
            return False
    
    def get_ptt_status(self):
        # Get the device path from config, fallback to CM108 default if missing
        device_file = self.config.config.get('ptt', {}).get('device_path', '/dev/hidraw0')
        config_mode = self.config.config.get('ptt', {}).get('mode', 'NONE').upper()
        mode = self.ptt_mode

        # Fallback detected: user wanted CM108, but we’re in VOX now
        if config_mode == 'CM108' and mode == 'VOX':
            return ("VOX Mode: Fell Back from CM108 (Check Hardware)", "orange")

        if mode == 'CM108':
            if not self.ptt:
                return ("CM108: Not Configured", "red")
            if not os.path.exists(device_file):
                return (f"CM108: Device File Missing ({device_file})", "red")
            try:
                with open(device_file, "wb", buffering=0) as f:
                    pass
                if getattr(self.ptt, "working", True):
                    return (f"CM108: Hardware Detected ({device_file})", "green")
                else:
                    return (f"CM108: Detected, But Last Operation Failed ({device_file})", "orange")
            except PermissionError:
                return (f"CM108: Device Found, Permission Denied ({device_file})", "orange")
            except Exception:
                return (f"CM108: Device Found, Unusable ({device_file})", "orange")
        elif mode == 'VOX':
            return ("VOX Mode: No Hardware PTT", "blue")
        elif mode == 'NONE':
            return ("VOX Mode: No Hardware PTT", "blue")
        else:
            return ("Unknown Mode", "gray")
