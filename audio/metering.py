import numpy as np
from .utils import get_dbfs
import threading

class Metering:
    def __init__(self, controller):
        self.controller = controller
        self.controller.last_clip_time = 0.0
        self.controller.last_limit_time = 0.0

        self.lock = threading.Lock()

        self.rx_db = -60.0
        self.tx_db = -60.0

        self.rx_peak_db = -60.0
        self.tx_peak_db = -60.0


    def update(self, samples: np.ndarray, direction: str):
        db = get_dbfs(samples)
        db = max(-60.0, float(db))
        peak = float(np.max(np.abs(samples))) / 32767.0
        peak_db = 20.0 * np.log10(peak) if peak > 0 else -60.0
        peak_db = max(-60.0, float(peak_db))

        with self.lock:
            if direction == "rx":
                self.rx_db = db
                self.rx_peak_db = peak_db
            else:
                self.tx_db = db
                self.tx_peak_db = peak_db
