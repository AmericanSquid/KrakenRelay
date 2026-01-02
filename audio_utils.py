import numpy as np
import logging

# ---- Audio Levels ----
def get_dbfs(samples: np.ndarray) -> float:
    rms = np.sqrt(np.mean(samples.astype(np.float32)**2))
    return 20 * np.log10(rms / 32767) if rms > 0 else -100.0

def check_clipping(samples: np.ndarray):
    peak = np.max(np.abs(samples))
    if peak >= 32767:
        logging.warning("⚠️ Clipping detected: signal hit digital full scale!")

def calculate_db_level(samples):
    rms = np.sqrt(np.mean(samples**2))
    if rms > 0:
        db_level = 20 * np.log10(rms / 32767)
    else:
        db_level = -60  # Set minimum dB level for silence
    return db_level
