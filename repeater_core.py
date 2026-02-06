import numpy as np
import time
import threading
import logging
from morse_code import ScheduleID, MorseCode
from ptt_controller import PTTManager
from tone_control import ToneGenerator, TonePlayer
from tot_manager import TOTManager
from audio_utils import get_dbfs, check_clipping, calculate_db_level
from kraken_dsp.kraken_dsp_wrapper import DSPChain

class RepeaterController:

    #------------------------#
    # Initialization & Setup #
    #------------------------# 
    def __init__(self, input_device, output_device, config, audio_manager):
        # Core Configuration
        self.config = config
        self.chunk_size = self.config.config['audio']['chunk_size']
        self.audio_manager = audio_manager
        self.ptt_manager = PTTManager(config)
        self.tot_manager = TOTManager(config, self.ptt_manager.safe_ptt_unkey)
        self.input_device = input_device
        self.output_device = output_device
        
        self._manual_id_event = threading.Event()
        self._manual_id_last = 0.0
        self._manual_id_requested = False
        self._manual_id_lock = threading.Lock()

        self.dsp_rx = DSPChain()
        self.dsp_tx = DSPChain()

        # -----------------------------
        # DSP CHAIN CONFIGURATION
        # -----------------------------
        a = self.config.config["audio"]

        sr = float(a.get("sample_rate", 48000))
        chunk = int(a.get("chunk_size", 1920))

        def compressor_macro(percent: float):
            """
            percent: 0–100 (UI-facing)
            Returns (threshold_db, ratio, makeup_db)
            Tuned for FM repeater TX audio.
            """
            p = max(0.0, min(100.0, percent))
            s = p / 100.0

            threshold_db = -15.0 - (10.0 * s)   # -15 → -25 dB
            ratio = 1.8 + (2.4 * s)             # 1.8 → 4.2
            makeup_db = 2.5 + (2.5 * s)          # +2.5 → +5.0 dB

            return threshold_db, ratio, makeup_db

        strength_pct = float(a.get("compressor_strength", 50))
        threshold_db, ratio, makeup_db = compressor_macro(strength_pct)

        # ---------- RX DSP CHAIN ----------
        # RX: HPF only by default (compressor optional, limiter off)
        self.dsp_rx.configure_hpf(
            enabled=bool(a.get("highpass_enabled", True)),
            order=4,
            cutoff_hz=float(a.get("highpass_cutoff", 300)),
            sample_rate=sr,
        )

        # RX compressor: OFF (keep RX neutral and predictable)
        self.dsp_rx.configure_compressor(
            enabled=False,
            threshold_db=0.0,
            ratio=1.0,
            sample_rate=sr,
            chunk_len=chunk,
        )

        self.dsp_rx.configure_limiter(
            enabled=False,
            threshold=0.85,
            sample_rate=sr,
            chunk_len=chunk,
        )

        # ---------- TX DSP CHAIN ----------
        # TX: HPF off (or enable if you want), compressor + limiter
        self.dsp_tx.configure_hpf(
            enabled=False,
            order=4,
            cutoff_hz=300,
            sample_rate=sr,
        )

        self.dsp_tx.configure_compressor(
            enabled=bool(a.get("compressor_enabled", False)),
            threshold_db=threshold_db,   # FROM MACRO
            ratio=ratio,                 # FROM MACRO
            sample_rate=sr,
            chunk_len=chunk,
            attack_ms=8.0,
            release_ms=160.0,
            makeup_db=makeup_db,         # FROM MACRO
        )
        self.dsp_tx.configure_limiter(
            enabled=bool(a.get("limiter_enabled", True)),
            threshold=float(a.get("limiter_threshold", 0.85)),
            sample_rate=sr,
            chunk_len=chunk,
        )

        # Audio Stream Set Up
        self.input_stream = None
        self.output_stream = None
        self.output_stream_2 = None # Optional mirrored output

        # State Tracking & Timing
        self.running = False
        self.transmitting = False
        self.tot_manager.reset()
        self.last_audio_time = time.time()
        self.last_transmission = time.time()
        self.current_rms = 0
        self._carrier_valid = False
        self._carrier_probe_start = None
        self._carrier_last_level_db = None
        self.squelch_open = False
        self.squelch_open_time = 0.0
        self._last_above_squelch = 0.0
        self.tx_start_pending = False
        self._shutdown_failed = False
        self._cleanup_lock = threading.Lock()

        # Buffer Logic
        self.kerchunk_buffer = []
        self.kerchunk_backlog_max = 50

        # Silence Buffer
        self._silence_chunk = np.zeros(self.config.config['audio']['chunk_size'], dtype=np.int16)
        self._silence_bytes = self._silence_chunk.tobytes()
        # Noise Buffer
        self._noise_chunk = (np.random.randint(-20000, 20000, self.chunk_size)).astype(np.int16)

        # CW ID (new logic)
        self._cw_gen = None
        self._cw_id_buffer = None
        self._cw_id_offset = 0
        self._sending_id = False
        self._cw_id_post_action = None

        self.morse = MorseCode(
            wpm=self.config.config['identification']['cw_wpm'],
            frequency=self.config.config['identification']['cw_pitch'],
            sample_rate=self.config.config['audio']['sample_rate'],
            volume=self.config.config['identification']['cw_volume'] / 100.0
        )

        # Stats
        self._meter_lock = threading.Lock()
        self.rx_db = -60.0
        self.tx_db = -60.0
        self.rx_peak_db = -60.0
        self.tx_peak_db = -60.0
        self.last_clip_time = 0.0
        self.last_limit_time = 0.0

        # CW IDer Setup
        self.schedule_id = ScheduleID(
            self,
            config=self.config,
            tx_start_fn=self.start_transmission,
            tx_stop_fn=self.stop_transmission,
            send_pcm_fn=self.play_audio_chunks,
            tx_state_fn=lambda: self.transmitting,
            set_skip_courtesy_fn=lambda: setattr(self, "skip_courtesy_tone", True)
        )
        self.skip_courtesy_tone = False

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
                if logging.getLogger().isEnabledFor(logging.DEBUG):
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

    def request_manual_id(self) -> None:
        """
        Non-blocking: just requests an ID; audio thread will execute it.
        Includes a small debounce so you don't spam IDs by accident.
        """
        now = time.time()
        if now - self._manual_id_last < 0.5:
            return
        self._manual_id_last = now
        self._manual_id_event.set()

    def start_cw_id(self, callsign):
        """
        Begin chunked playback of the give callsign as CW ID.
        Assumes self.morse is a MorseCode instance,
        and self.chunk_size is your audio chunk size.
        """
        self._cw_gen = self.morse.generate_chunks(callsign, self.chunk_size)
        self.start_transmission()

    def _update_squelch_state(self, level_db: float, now: float) -> bool:
        """
        Debounced squelch:
        - Open when level_db > open_thr
        - Once open, stay open until level_db < close_thr for >= hang_time
        """
        audio_cfg = self.config.config.get("audio", {})

        open_thr = float(audio_cfg.get("squelch_threshold", -40))
        hyst_db = float(audio_cfg.get("squelch_hysteresis_db", 3.0))
        hang_time = float(audio_cfg.get("squelch_hang_time", 0.25))

        close_thr = open_thr - abs(hyst_db)

        if not self.squelch_open:
            if level_db > open_thr:
                self.squelch_open = True
                self.squelch_open_time = now
                self._last_above_squelch = now
                self.kerchunk_buffer = []
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug("[Squelch] OPEN db=%.1f thr=%.1f", level_db, open_thr)
        else:
            if level_db > close_thr:
                self._last_above_squelch = now
            elif (now - self._last_above_squelch) >= hang_time:
                self.squelch_open = False
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(
                        "[Squelch] CLOSE db=%.1f close_thr=%.1f hang=%.2f",
                        level_db, close_thr, hang_time
                    )

        return self.squelch_open

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
            try:
                ident = t.ident  # Python thread ident
                # On Linux, get native TID (will match top/htop)
                tid = t.native_id if hasattr(t, 'native_id') else None
                logging.info(f"Thread name: {t.name}, ident: {ident}, native_id: {tid}")
            except Exception as e:
                logging.info(f"Could not get info for thread: {t} ({e})")

    def audio_loop(self):
        consecutive_errors = 0
        max_errors = 5
        max_backoff = 0.5
        fatal_reason = None
        
        logging.info("[Repeater] Audio thread started.")

        while self.running:

            self.tot_manager.check_lockout_expired()
            if self._cw_gen is not None:
                sr = int(self.config.config["audio"]["sample_rate"])
                frame_sec = float(self.chunk_size) / float(sr)

                now_m = time.monotonic()
                next_t = getattr(self, "_cw_next_t", None)
                if next_t is None:
                    next_t = now_m

                if now_m < next_t:
                    time.sleep(next_t - now_m)

                try:
                    chunk = next(self._cw_gen)
                    self._send_pcm(chunk)
                    self._cw_next_t = next_t + frame_sec
                except StopIteration:
                    self._cw_gen = None
                    self._cw_next_t = None
                    self.skip_courtesy_tone = True
                    if getattr(self, "transmitting", False):
                        self.stop_transmission()

                continue

            try:
                self.process_audio()
                if self._manual_id_event.is_set():
                    self._manual_id_event.clear()
                    logging.info("[WebUI] Manual ID requested.")
                    try:
                        self.schedule_id.send_id()
                    except Exception:
                        logging.exception("[Repeater] Manual ID failed.")

                self.schedule_id.check_and_send()

                if consecutive_errors:
                    logging.info("[Repeater] Audio loop recovered after %d error(s).", consecutive_errors)
                    consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
                logging.error(f"Error in audio loop: {e}")

                if not self.running:
                    break

                # Backoff before retrying
                backoff = min(0.2 * (2 ** (consecutive_errors - 1)), max_backoff)
                time.sleep(backoff)

                # Too many consecutive errors? Time to bail.
                if consecutive_errors >= max_errors:
                    fatal_reason = f"{consecutive_errors} consecutive audio loop errors"
                    break
        # === CLEAN EXIT ===
        if not self.running and fatal_reason is None:
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

    def cleanup(self) -> bool:
        logging.info("========== [Repeater] Cleanup Starting ==========")
        logging.info(
            "Controller state before cleanup: running=%s transmitting=%s skip_courtesy_tone=%s shutdown_failed=%s",
            self.running, self.transmitting, self.skip_courtesy_tone, self._shutdown_failed
        )
        ok = True
        self._shutdown_failed = False

        # --- A) SAFETY FIRST: UNKEY ---
        logging.info("[Cleanup Step A] Unkeying transmitter and suppressing courtesy tone...")
        try:
            self.transmitting = False
            self.skip_courtesy_tone = True
            self.ptt_manager.safe_ptt_unkey()
            logging.info("[Cleanup Step A] PTT unkeyed (forced cleanup).")
        except Exception:
            logging.exception("[Cleanup Step A] Failed to unkey PTT during cleanup.")
            ok = False

        # --- B) SIGNAL ALL LOOPS/THREADS TO STOP ---
        self.running = False # Stop the loop
        logging.info("[Cleanup Step B] Stopping audio loop: running=%s", self.running)

        # --- C) JOIN THREADS ---
        def _join(th, name: str, timeout: float) -> bool:
            if not th:
                return True
            if not th.is_alive():
                return True
            if th is threading.current_thread():
                logging.warning("[Repeater] cleanup called from %s; skipping self-join", name)
                return True
            logging.info("[Cleanup Step C] Joining thread: %s", name)
            th.join(timeout=timeout)
            if th.is_alive():
                logging.error("[Repeater] %s still alive after %.1fs", name, timeout)
                return False
            logging.info("[Cleanup Step C] Joined thread: %s", name)
            return True

        ok &= _join(getattr(self, "audio_thread", None), "audio_thread", 3.0)

        if not ok:
            logging.critical("[Repeater] cleanup incomplete - leaving PortAudio intact. Restarting service to recover.")
            self._shutdown_failed = True
            return False
    
        # --- D) CLOSE STREAMS (threads are gone) ---
        if ok:
            for name in ("input_stream", "output_stream", "output_stream_2"):
                s = getattr(self, name, None)
                if not s:
                    continue
                try:
                    logging.info(f"[Cleanup Step D] Closing stream: {name}")
                    try:
                        s.stop_stream()
                    except Exception:
                        pass
                    s.close()
                    logging.info(f"[Cleanup Step D] Closed stream: {name}")
                except Exception as e:
                    logging.warning(f"[Cleanup Step D] Exception closing {name}: {e}")
                    ok = False
                setattr(self, name, None)

        # --- G) TERMINATE PortAudio LAST ---
        #try:
        #    self.audio_manager.cleanup()
        #except Exception:
        #    logging.exception("[Repeater] Audio manager cleanup failed.")
        #    ok = False

        self._shutdown_failed = not ok
        if ok:
            logging.info("Cleanup completed successfully. No issues detected.")
        else:  
            logging.warning("Cleanup completed with ERRORS. Check previous logs for details.")

        logging.info("[Repeater] cleanup() done. ok=%s shutdown_failed=%s", ok, self._shutdown_failed)
        logging.info("🧵 Active threads after cleanup: %d", threading.active_count())
        for t in threading.enumerate():
            logging.info("  • %s (alive=%s)", t.name, t.is_alive())

        logging.info("========== [Repeater] Clean Up Ended ==========")
        logging.info("73 de K3AYV (American Squid).")
        return ok

    #-----------------------#
    # Core Audio Processing #
    #-----------------------#
    def process_audio(self):
        try:
            data = self.input_stream.read(self.chunk_size, exception_on_overflow=False)
        except Exception as e:
            # If we're shutting down, just exit quietly
            if not self.running:
                return
            logging.error(f"[Input] Input stream read failed: {e}")
            raise

        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)

        if samples.size == 0:
            self.current_rms = 0.0
        else:
            samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
            self.current_rms = float(np.sqrt(np.mean(samples * samples)))

        self._update_meter(samples, "rx")
        # Check squelch first
        now = time.time()
        level_db = float(calculate_db_level(samples))
    
        prev_open = self.squelch_open
        squelch_open_now = self._update_squelch_state(level_db, now)

        just_opened = False

        # Track squelch transitions for anti-kerchunk holdoff
        if squelch_open_now and not prev_open:
            if not self.transmitting:
                just_opened = True
                self.squelch_open_time = now
                self.kerchunk_buffer = []  # clear buffer on a fresh open edge
                self.tx_start_pending = True
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug("[Anti-Kerchunk] Squelch open. Holding off")
            else:
                # Squelch reopened during an ongoing transmission – no kerchunk holdoff
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug("[Anti-Kerchunk] Squelch reopened during transmit; kerchunk holdoff bypassed")
        elif not squelch_open_now and prev_open:
            if not self.transmitting and self.kerchunk_buffer:
                logging.info("[Anti-Kerchunk] Suppressed short key-up.")
                self.kerchunk_buffer = []
            if not self.transmitting:
                self.tx_start_pending = False
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug("[Anti-Kerchunk] Squelch closed.")

        # --- Carrier Validity Gate --- #
        if squelch_open_now:
            audio_cfg = self.config.config.get("audio", {})
            cv_min_delta = float(audio_cfg.get("carrier_validity_min_delta", 0.1))
            cv_timeout = float(audio_cfg.get("carrier_validity_timeout", 2.5))

            # Start / reset on a fresh open edge
            if just_opened or self._carrier_probe_start is None:
                self._carrier_valid = False
                self._carrier_probe_start = now  # "no-movement timer" starts now
                self._carrier_last_level_db = level_db

            # Compute movement since last frame
            last = self._carrier_last_level_db
            delta = 0.0 if last is None else abs(level_db - float(last))
            self._carrier_last_level_db = level_db

            if delta >= cv_min_delta:
                # We saw "movement" -> mark valid and reset the no-movement timer
                if not self._carrier_valid:
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug("[CarrierValidity] VALID delta=%.3f >= %.3f", delta, cv_min_delta)
                self._carrier_valid = True
                self._carrier_probe_start = now
            else:
                # No movement: if it's been too long, declare invalid and drop
                if (now - float(self._carrier_probe_start or now)) >= cv_timeout:
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(
                            "[CarrierValidity] INVALID (delta=%.3f < %.3f for %.2fs)",
                            delta, cv_min_delta, cv_timeout
                        )

                    # Force-close and dump buffers
                    self.squelch_open = False
                    squelch_open_now = False
                    self._carrier_valid = False
                    self._carrier_probe_start = None
                    self._carrier_last_level_db = None
                    self.kerchunk_buffer = []
                    self.tx_start_pending = False

                    # If we were already keyed, drop immediately and do NOT courtesy-beep
                    if self.transmitting:
                        self.skip_courtesy_tone = True
                        self.stop_transmission()
                    return

                # While still "probing", only buffer pre-TX so we don't key early
                if not self.transmitting:
                    self.kerchunk_buffer.append(samples.copy())
                    if len(self.kerchunk_buffer) > self.kerchunk_backlog_max:
                        self.kerchunk_buffer.pop(0)
                    return

        else:
            # squelch closed: reset probe state
            self._carrier_valid = False
            self._carrier_probe_start = None
            self._carrier_last_level_db = None

        if squelch_open_now:
            # Process audio chain
            if (
                self.config.config["audio"].get("highpass_enabled", False)
                or self.config.config["audio"].get("compressor_enabled", False)
            ):
                #if np.max(np.abs(samples)) > 16:
                samples = self.dsp_rx.process_int16_to_int16(samples)

            s = np.asarray(samples, dtype=np.float32)
            s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
            self.current_rms = float(np.sqrt(np.mean(s*s)))

            if not self.transmitting:
                if just_opened:
                    logging.info("Squelch opened - Starting Tx")
            self.handle_transmission(samples)
        else:
            self.current_rms = 0
            if self.transmitting:
                if time.time() - self.last_audio_time > self.config.config['repeater']['tail_time']:
                    logging.info("Silence persists beyond tail time. Stopping transmission.")
                    self.stop_transmission()
                else:
                    # 🚨 We're in the tail hang time — KEEP AUDIO FLOWING
                    self._send_pcm(self._silence_chunk)
            else:
                pass
    def handle_transmission(self, samples):
        if self.tot_manager.check_timeout(self.transmitting):
            self.tone_player.play_tot_tone()
            self.stop_transmission()
            return

        if getattr(self, "vox_delay_active", False):
            now = time.time()
            if now < self.vox_delay_end_time:
                # Send noise burst chunk instead of real audio
                self._send_pcm(self._noise_chunk)
                # Buffer the real samples for immediate transmit after delay
                self.vox_buffer.append(samples.copy())
                return
            
            else:
                # Done: flush buffered chunks
                for chunk in self.vox_buffer:
                    self._send_tx_chunk(chunk)
                self.vox_buffer = []
                self.vox_delay_active = False

        if not self.transmitting:
            anti_s = float(self.config.config.get("repeater", {}).get("anti_kerchunk_time", 0) or 0)

            if anti_s > 0 and self.squelch_open:
                held = time.time() - float(self.squelch_open_time or 0.0)
                if held < anti_s:
                    remaining = anti_s - held
                    # Don’t key yet — buffer audio until we decide it’s real
                    self.kerchunk_buffer.append(samples.copy())

                    # Log occasionally so you don’t spam
                    if len(self.kerchunk_buffer) == 1 or len(self.kerchunk_buffer) % 10 == 0:
                        if logging.getLogger().isEnabledFor(logging.DEBUG):
                            logging.debug(
                                "[AntiKerchunk] Holdoff: held=%.3fs remaining=%.3fs buffered=%d",
                                held, remaining, len(self.kerchunk_buffer)
                            )
                    return

                # Gate passed
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(
                        "[AntiKerchunk] Gate passed: held=%.3fs >= %.3fs. Starting TX. buffered=%d",
                        held, anti_s, len(self.kerchunk_buffer)
                    )

            self.start_transmission()

            if self.kerchunk_buffer:
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug("[AntiKerchunk] Flushing %d buffered chunks", len(self.kerchunk_buffer))
                for chunk in self.kerchunk_buffer:
                    self._send_tx_chunk(chunk)
                self.kerchunk_buffer = []

        self._send_tx_chunk(samples)

        self.last_audio_time = time.time()

    def _send_tx_chunk(self, samples: np.ndarray) -> None:
        # samples should be int16

        if (
            self.config.config["audio"].get("limiter_enabled", False)
            or self.config.config["audio"].get("compressor_enabled", False)
            or self.config.config["audio"].get("highpass_enabled", False)
        ):
            #if np.max(np.abs(samples)) > 16:
            samples = self.dsp_tx.process_int16_to_int16(samples)

        self._update_meter(samples, "tx")
        check_clipping(samples)
        self._send_pcm(samples)


    # ---------- unified TX funnel ----------   ← paste helper here
    def _send_pcm(self, pcm: np.ndarray):
        # Step 1: Convert bytes or int16 to float32 for safety checks
        if isinstance(pcm, bytes):
            data = pcm
        elif isinstance(pcm, np.ndarray) and pcm.dtype == np.int16:
            data = pcm.tobytes()
        else:
            # Handle Conversion (slow/fallback path)
            # Clip to int16 range
            pcm = np.clip(np.asarray(pcm), -32768, 32767).astype(np.int16)
            data = pcm.tobytes()

        # Primary output
        try:
            if self.output_stream:
                self.output_stream.write(data)
        except Exception as e:
            logging.warning(f"[Output] Primary output write failed: {e}")

        # Secondary output (dual output) – best effort
        if self.output_stream_2:
            try:
                self.output_stream_2.write(data)
            except Exception as e:
                logging.warning(f"[Dual Output] Secondary output failed, disabling: {e}")
                try:
                    self.output_stream_2.stop_stream()
                    self.output_stream_2.close()
                except Exception:
                    pass
                self.output_stream_2 = None    

    def _update_meter(self, samples: np.ndarray, direction: str):
        db = get_dbfs(samples)
        db = max(-60.0, float(db))
        peak = float(np.max(np.abs(samples))) / 32767.0
        peak_db = 20.0 * np.log10(peak) if peak > 0 else -60.0
        peak_db = max(-60.0, float(peak_db))
        
        with self._meter_lock:
            if direction == "rx":
                self.rx_db = db
                self.rx_peak_db = peak_db
            else:
                self.tx_db = db
                self.tx_peak_db = peak_db


    #----------------------#
    # Transmission Control #
    #----------------------#

    def start_transmission(self):
        self.tx_start_pending = False
        if self.tot_manager.is_locked():
            return
        self.transmitting = True
        self.transmission_start_time = time.time()
        self.tot_manager.reset()
        self.tot_manager.tx_start_time = time.time()

        self.ptt_manager.safe_ptt_key()

        if self.ptt_manager.ptt_mode != 'CM108':
            delay_sec = float(self.config.config['repeater'].get('carrier_delay', 0) or 0)
            if delay_sec > 0:
                self.vox_delay_active = True
                self.vox_delay_end_time = time.time() + delay_sec
                self.vox_buffer = []
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug(f"[Repeater] VOX carrier delay active for {delay_sec:.2f} seconds (non-blocking)")
            else:
                self.vox_delay_active = False
                self.vox_buffer = []

        logging.info("Starting transmission")

    def stop_transmission(self):
        # Prime ALSA before tone playback
        self._send_pcm(self._silence_bytes)

        if self.config.config['repeater']['courtesy_tone_enabled'] and not self.skip_courtesy_tone:
            self.tone_player.play_courtesy_tone()
        else:
            if self.skip_courtesy_tone:
                if logging.getLogger().isEnabledFor(logging.DEBUG):
                    logging.debug("Skipping courtesy tone after CW ID.")

        fade_duration = 0.1
        sample_rate = self.config.config['audio']['sample_rate']
        chunk_size = self.config.config['audio']['chunk_size']
        num_fade_chunks = int((fade_duration * sample_rate) // chunk_size)
        
        start = time.time()
        for i in range(num_fade_chunks):
            self._send_pcm(self._silence_bytes)
            pass
        
        self.transmitting = False
        self.transmission_start_time = None
        self.tot_manager.reset()
        self.tot_manager.tx_start_time = None
        self.last_transmission = time.time()

        logging.info("Transmission stopped with fade-out")

        if not self.skip_courtesy_tone:
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
            self._send_tx_chunk(chunk)


