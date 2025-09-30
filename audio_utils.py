import numpy as np
from scipy import signal
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

def check_squelch(samples, squelch_threshold_db):
    db_level = 20 * np.log10(np.sqrt(np.mean(samples**2)) / 32767)
    return db_level > squelch_threshold_db

# ---- Audio Effects ----
def limiter(samples: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    peak = np.max(np.abs(samples))
    limit_val = int(threshold * 32767)

    if peak > limit_val:
        scale = limit_val / peak
        samples = samples * scale
        logging.debug(f"Limiter: Scaled peak from {peak} to {limit_val}")
    return samples.astype(np.int16)

def apply_highpass_filter(samples, sample_rate, cutoff):
    if samples is None or samples.size == 0:
        return samples
  
    nyq = sample_rate * 0.5
    if cutoff <= 0 or cutoff >= nyq:
        logging.warning(f"HPF: invalid cutoff={cutoff} Hz (nyquist={nyq:.1f} Hz) — bypassing")
        return samples

    x = samples.astype(np.float32, copy=False)
    sos = signal.butter(4, cutoff / nyq, btype="highpass", output="sos")
    y = signal.sosfilt(sos, x)
  
    logging.debug(f"HPF: applied (sosfilt) sr={sample_rate} Hz cutoff={cutoff} Hz n={x.shape[0]}")
    return y.astype(np.float32, copy=False)