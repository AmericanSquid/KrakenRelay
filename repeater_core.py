import numpy as np
import time
import threading
import logging
from scipy import signal
from morse_code import ScheduleID
from ptt_controller import PTTManager
from tone_control import ToneGenerator, TonePlayer
from audio_utils import check_clipping, limiter, apply_highpass_filter, calculate_db_level

class RepeaterController:

    #------------------------#
    # Initialization & Setup #
    #------------------------# 
    def __init__(self, input_device, output_device, config, audio_manager):
        # Core Configuration
        self.config = config
        self.audio_manager = audio_manager
        self.ptt_manager = PTTManager(config)
        self.input_device = input_device
        self.output_device = output_device

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
        self.squelch_open_time = None
        self.last_audio_time = time.time()
        self.last_transmission = time.time()
        self.current_rms = 0
        
        # CW IDer Setup
        self.schedule_id = ScheduleID(
            config=self.config,
            tx_start_fn=self.start_transmission,
            tx_stop_fn=self.stop_transmission,
            send_pcm_fn=self.play_audio_chunks,
            tx_state_fn=lambda: self.transmitting,
            set_skip_courtesy_fn=lambda: setattr(self, "skip_courtesy_tone", True)
        )
        self.skip_courtesy_tone = False

        # Thread Safety
        self.ptt_lock = threading.Lock()

        # Tone Control Setup
        self.tone_generator = ToneGenerator(self.config)
        self.tone_player = TonePlayer(
            config=self.config,
            send_pcm_callable=self._send_pcm,
            tx_state_callable=lambda: self.transmitting,
            tone_generator=self.tone_generator
        )

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
            else:
                logging.debug("Dual output disabled in config")

            # ---- Primary input/output streams ----
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
            if self.tot_locked:
                lockout_time = self.config.config['tot'].get('tot_lockout_time', 180)
                if time.time() - self.lockout_start_time > lockout_time:
                    self.tot_locked = False
                    logging.info("🔓 TOT lockout released")
            try:
                self.process_audio()

                self.schedule_id.check_and_send()
                            
                if consecutive_errors:
                    logging.info("[Repeater] Audio loop recovered after %d error(s).", consecutive_errors)
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
                    self.ptt_manager.safe_ptt_unkey()
            else:
                self.ptt_manager.safe_ptt_unkey()
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
        self.ptt_manager.safe_ptt_unkey()
        
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

    def process_audio(self):
        data = self.input_stream.read(self.config.config['audio']['chunk_size'], exception_on_overflow=False)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        self.current_rms = calculate_db_level(samples)
        chunk  = self.config.config['audio']['chunk_size']

        # Check squelch first
        if self.check_squelch(samples):
            # Process audio chain
            if self.config.config['audio'].get('highpass_enabled', True):
                samples = apply_highpass_filter(
                    samples,
                    sample_rate=self.config.config['audio']['sample_rate'],
                    cutoff=self.config.config['audio'].get('highpass_cutoff', 300)
                )
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
                self.tone_player.play_tot_tone()
                if self.config.config['tot']['tot_lockout_enabled']:
                    logging.info("TOT lockout activated")
                    self.tot_locked = True
                    self.lockout_start_time = time.time()
                    self.ptt_manager.safe_ptt_unkey()
                    self.transmitting = False
                    self.transmission_start_time = None

                self.stop_transmission()
                return   

        if not self.transmitting:
            self.start_transmission()

        if self.config.config['audio'].get('limiter_enabled', True):
            samples = limiter(samples, threshold=self.config.config['audio'].get('limiter_threshold', 0.85))

        check_clipping(samples) 
        self._send_pcm(samples.astype(np.int16))
        
        self.last_audio_time = time.time()

    # ---------- unified TX funnel ----------   ← paste helper here
    def _send_pcm(self, pcm: np.ndarray):
        # Step 1: Convert bytes or int16 to float32 for safety checks
        if isinstance(pcm, bytes):
            pcm = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
        elif not isinstance(pcm, np.ndarray):
            logging.error(f"_send_pcm received unsupported type: {type(pcm)}")
            return
        elif pcm.dtype != np.float32:
            pcm = pcm.astype(np.float32)

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

    #----------------------#
    # Transmission Control #
    #----------------------#

    def start_transmission(self):
        if self.tot_locked:
            logging.warning("Attempted to start transmission during TOT lockout. Blocking.")
            return

        self.transmitting = True
        self.transmission_start_time = time.time()
        self.tot_timer = 0

        self.ptt_manager.safe_ptt_key()

        if self.ptt_manager.ptt_mode!= 'CM108':
            delay_sec = self.config.config['repeater']['carrier_delay']
            if delay_sec > 0:
                logging.debug(f"[Repeater] Sending VOX keying burst for {delay_sec:.2f} seconds")
        
                sample_rate = self.config.config['audio']['sample_rate']
                chunk_size = self.config.config['audio']['chunk_size']
                num_chunks = int((delay_sec * sample_rate) // chunk_size)

                for _ in range(num_chunks):
                    noise = (np.random.randint(-20000, 20000, chunk_size)).astype(np.int16).tobytes()
                    try:
                        self.output_stream.write(noise)
                    except Exception as e:
                        logging.debug(f"VOX carrier delay burst write failed: {e}")
                
                    if self.output_stream_2:
                        try:
                            self.output_stream_2.write(noise)
                        except Exception as e:
                            logging.debug(f"VOX carrier delay burst write (output 2) failed: {e}")
                
                    time.sleep(chunk_size / sample_rate)

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

        if self.config.config['repeater']['courtesy_tone_enabled'] and not self.skip_courtesy_tone:
            self.tone_player.play_courtesy_tone()
        else:
            if self.skip_courtesy_tone:
                logging.debug("Skipping courtesy tone after CW ID.")

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

        self.schedule_id.mark_post_tx()
        self.skip_courtesy_tone = False
        self.ptt_manager.safe_ptt_unkey()
        logging.info("Transmitter unkeyed.")

    def play_audio_chunks(self, audio):
        chunk_size = self.config.config['audio']['chunk_size']
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            self._send_pcm(chunk)


