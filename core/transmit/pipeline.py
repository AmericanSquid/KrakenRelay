import numpy as np
import logging
import time

from runtime.logging_utils import debug_enabled

class Pipeline:
    def __init__(self, controller, config):
        self.controller = controller
        self.config = config

    def _handle_anti_kerchunk(self, gate, repeater_cfg, debug_on, now, samples):
        anti_s = float(repeater_cfg.get("anti_kerchunk_time", 0) or 0)
    
        if anti_s <= 0 or not gate.squelch_open:
            return False
    
        squelch_open_time = float(gate.squelch_open_time or 0.0)
        held = now - squelch_open_time
    
        if held < anti_s:
            remaining = anti_s - held
            gate.kerchunk_buffer.append(samples.copy())
    
            if len(gate.kerchunk_buffer) == 1 or len(gate.kerchunk_buffer) % 10 == 0:
                if debug_on:
                    logging.debug(
                        "[AntiKerchunk] Holdoff: held=%.3fs remaining=%.3fs buffered=%d",
                        held, remaining, len(gate.kerchunk_buffer)
                    )
            return True
    
        if debug_on:
            logging.debug(
                "[AntiKerchunk] Gate passed: held=%.3fs >= %.3fs. Starting TX. buffered=%d",
                held, anti_s, len(gate.kerchunk_buffer)
            )
    
        return False    	
    
    def feed(self, samples):
        controller = self.controller
        gate = controller.signal_gate 
        tx = controller.tx_state
        cfg = self.config.config
        repeater_cfg = cfg.get("repeater", {})

        debug_on = debug_enabled()
        now = time.time()
        send_chunk = controller.tx_audio.send_chunk

        if controller.tot_manager.check_timeout(tx.transmitting):
            controller.tone_player.play_tot_tone()
            controller.tx_control.stop()
            return

        if not tx.transmitting:
            if self._handle_anti_kerchunk(gate, repeater_cfg, debug_on, now, samples):
                return

            controller.tx_control.start()

            if gate.kerchunk_buffer:
                if debug_on:
                    logging.debug("[AntiKerchunk] Flushing %d buffered chunks", len(gate.kerchunk_buffer))
                for chunk in gate.kerchunk_buffer:
                   send_chunk(chunk)
                gate.kerchunk_buffer = []

        send_chunk(samples)

        tx.last_audio_time = now
