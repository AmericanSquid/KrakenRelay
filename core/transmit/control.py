import time
import logging
from runtime.logging_utils import debug_enabled
from audio.primitives import fade_out

class Control:
    def __init__(self, controller, config):
        self.controller = controller
        self.controller.tot_manager.reset()
        self.controller.current_rms = 0
        self.config = config
              
    def _handle_vox_delay(self, repeater_cfg):
        controller = self.controller
        now = time.time()

        if controller.ptt_manager.ptt_mode == 'CM108':
            return

        delay_sec = float(repeater_cfg.get('carrier_delay', 0) or 0)

        if delay_sec <= 0:
            return

        audio_cfg = self.config.config['audio']
        sample_rate = audio_cfg['sample_rate']
        chunk_size = audio_cfg['chunk_size']

        num_chunks = int((delay_sec * sample_rate) // chunk_size)

        if debug_enabled():
            logging.debug(
                f"[VOX] Wake burst: {num_chunks} chunks ({delay_sec:.2f}s)"
            )

        import numpy as np

        for i in range(num_chunks):
            noise = np.random.normal(0, 1, chunk_size)
            noise = np.diff(np.concatenate(([0], noise)))
            noise *= 20000

            chunk = noise.astype(np.int16)

        send_chunk = controller.tx_audio.send_chunk

        for i in range(num_chunks):
            if debug_enabled() and i == 0:
                logging.debug("[VOX] Noise burst start")

            send_chunk(chunk)

        if debug_enabled():
            logging.debug("[VOX] Noise burst end")
            
    def _play_courtesy(self, repeater_cfg):
        controller = self.controller
        tx = controller.tx_state
        
        if repeater_cfg['courtesy_tone_enabled'] and not tx.skip_courtesy_tone:
            controller.tone_player.play_courtesy_tone()
        elif tx.skip_courtesy_tone and debug_enabled():
            logging.debug("Skipping courtesy tone after CW ID.")        	    

    def start(self):
        controller = self.controller
        tx = controller.tx_state
        cfg = self.config.config
        audio_cfg = cfg['audio']
        repeater_cfg = cfg['repeater']
        now = time.time()

        tx.tx_start_pending = False

        if controller.tot_manager.is_locked():
            return
            
        tx.transmitting = True
        controller.transmission_start_time = now
        controller.tot_manager.reset()
        controller.tot_manager.tx_start_time = now

        controller.ptt_manager.safe_ptt_key()

        self._handle_vox_delay(repeater_cfg)

        logging.info("Starting transmission")

    def stop(self):
        controller = self.controller
        tx = controller.tx_state
        cfg = self.config.config
        audio_cfg = cfg['audio']
        repeater_cfg = cfg['repeater']
        now = time.time()
        
        # Prime ALSA before tone playback
        controller.audio_io.send_pcm(tx.silence_bytes)

        self._play_courtesy(repeater_cfg)

        fade_duration = 0.1
        sample_rate = audio_cfg['sample_rate']
        chunk_size = audio_cfg['chunk_size']
        num_fade_chunks = int((fade_duration * sample_rate) // chunk_size)

        start = now
        fade_out(controller.audio_io.send_pcm, tx.silence_bytes, num_fade_chunks)
        
        tx.transmitting = False
        controller.transmission_start_time = None
        controller.tot_manager.reset()
        controller.tot_manager.tx_start_time = None
        tx.last_transmission = now

        logging.info("Transmission stopped with fade-out")

        if not tx.skip_courtesy_tone:
            controller.schedule_id.mark_post_tx()
        tx.skip_courtesy_tone = False
        controller.ptt_manager.safe_ptt_unkey()
        logging.info("Transmitter unkeyed.")
