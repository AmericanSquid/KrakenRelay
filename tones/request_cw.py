import time
import logging
import threading

class RequestID:
    def __init__(self, controller):
        self.controller = controller
        
        self.manual_id_event = threading.Event()
        self.manual_id_last = 0.0
        self.manual_id_requested = False
        self.manual_id_lock = threading.Lock()

        self.controller._cw_gen = None
                    
    def request_manual_id(self) -> None:
        """
        Non-blocking: just requests an ID; audio thread will execute it.
        Includes a small debounce so you don't spam IDs by accident.
        """
        now = time.time()
        if now - self.manual_id_last < 0.5:
            return
        self.manual_id_last = now
        self.manual_id_event.set()

    def start_cw_id(self, callsign):
        controller = self.controller
        """
        Begin chunked playback of the give callsign as CW ID.
        Assumes self.morse is a MorseCode instance,
        and self.chunk_size is your audio chunk size.
        """
        controller._cw_gen = controller.morse.generate_chunks(callsign, controller.chunk_size)
        controller.tx_control.start()
