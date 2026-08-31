import threading
import time

from .engine import CWGenerator
from .tone_vol import _tone_vol


class RequestID:
    def __init__(self, config, start_transmission, set_cw_generator):
        self.config = config
        self.start_transmission = start_transmission
        self.set_cw_generator = set_cw_generator

        self.manual_id_event = threading.Event()
        self.manual_id_last = 0.0

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
        """
        Begin chunked playback of the given callsign as CW ID.

        Build the CW generator at send time so live config changes
        such as cw_volume, cw_pitch, and cw_wpm are honored.
        """
        cfg = self.config.config
        audio_cfg = cfg["audio"]
        id_cfg = cfg["identification"]

        chunk_size = audio_cfg["chunk_size"]

        morse = CWGenerator(
            wpm=id_cfg["cw_wpm"],
            frequency=id_cfg["cw_pitch"],
            sample_rate=audio_cfg["sample_rate"],
            volume=_tone_vol(id_cfg.get("cw_volume", 100), safe_max=0.25),
        )

        self.set_cw_generator(morse.generate_chunks(callsign, chunk_size))
        self.start_transmission()
