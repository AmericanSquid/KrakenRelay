import numpy as np
import time
import logging

from audio import (
    check_clipping,
    calculate_db_level
)

class AudioLoop:
    def __init__(self, controller, config):
        self.controller = controller
        self.config = config

    def _handle_cw_playback(self):
        controller = self.controller
        cfg = self.config.config
        audio_cfg = cfg['audio']
        tx = controller.tx_state
        
        cw_gen = controller._cw_gen
        if cw_gen is None:
            return False

        sr = int(audio_cfg["sample_rate"])
        frame_sec = float(controller.chunk_size) / float(sr)

        now_m = time.monotonic()
        next_t = getattr(controller, "_cw_next_t", None)
        if next_t is None:
            next_t = now_m

        if now_m < next_t:
            time.sleep(next_t - now_m)

        try:
            chunk = next(cw_gen)
            controller.audio_io.send_pcm(chunk)
            controller._cw_next_t = next_t + frame_sec
        except StopIteration:
            controller._cw_gen = None
            controller._cw_next_t = None
            tx.skip_courtesy_tone = True
            if getattr(tx, "transmitting", False):
                controller.tx_control.stop()

        return True

    def _handle_audio_error(self, e, consecutive_errors, max_backoff):
        controller = self.controller
        
        logging.error(f"Error in audio loop: {e}")
    
        if not controller.running:
            return consecutive_errors, True
    
        backoff = min(0.2 * (2 ** (consecutive_errors - 1)), max_backoff)
        time.sleep(backoff)
    
        return consecutive_errors, False

    def _handle_normal_audio(self, manual_id_event):
        controller = self.controller
    
        controller.process_audio.process_audio()
    
        if manual_id_event.is_set():
            manual_id_event.clear()
            logging.info("[WebUI] Manual ID requested.")
            try:
                controller.schedule_id.send_id()
            except Exception:
                logging.exception("[Repeater] Manual ID failed.")
    
        controller.schedule_id.check_and_send()

    def _shutdown_cleanup(self):
        controller = self.controller
        tx = controller.tx_state
    
        try:
            if getattr(tx, "transmitting", False):
                try:
                    controller.tx_control.stop()
                except Exception:
                    controller.ptt_manager.safe_ptt_unkey()
            else:
                controller.ptt_manager.safe_ptt_unkey()
        except Exception:
            logging.exception("[Repeater] Error while unkeying during shutdown.")

    def audio_loop(self):
        controller = self.controller
        request_cw = controller.request_cw
        manual_id_event = request_cw.manual_id_event

        consecutive_errors = 0
        max_errors = 5
        max_backoff = 0.5
        fatal_reason = None

        logging.info("[Repeater] Audio thread started.")

        while controller.running:
            try:
                controller.tot_manager.check_lockout_expired()
                if self._handle_cw_playback():
                    continue
            
                self._handle_normal_audio(manual_id_event)

                if consecutive_errors:
                    logging.info("[Repeater] Audio loop recovered after %d error(s).", consecutive_errors)
                    consecutive_errors = 0

            except Exception as e:
                consecutive_errors += 1
            
                consecutive_errors, should_exit = self._handle_audio_error(
                    e,
                    consecutive_errors,
                    max_backoff
                )

                if should_exit:
                    break

                if consecutive_errors >= max_errors:
                    fatal_reason = f"{consecutive_errors} consecutive audio loop errors"
                    break
                        

        if not controller.running and fatal_reason is None:
            logging.info("[Repeater] Audio thread stopping (requested)")
        else:
            logging.critical(
                f"[Repeater] Audio loop exiting due to: {fatal_reason or 'unknown fatal error'}"
            )
        controller.running = False
        self._shutdown_cleanup()
        
        logging.info("[Repeater] Audio thread exited.")
