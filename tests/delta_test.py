import os
import time
import yaml
import numpy as np
import sounddevice as sd
from collections import deque

# -----------------------------
# Load KrakenRelay config
# -----------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "..", "config.yaml")

with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

audio_cfg = cfg["audio"]

DEVICE_INDEX = audio_cfg["input_index"]
SAMPLE_RATE  = audio_cfg["sample_rate"]
CHUNK_SIZE   = audio_cfg["chunk_size"]
CHANNELS     = 1

print("=== Delta Test ===")
print("Config:", CONFIG_PATH)
print("RX device index:", DEVICE_INDEX)
print("Sample rate:", SAMPLE_RATE)
print("Chunk size:", CHUNK_SIZE)
print("------------------")

# -----------------------------
# Helpers
# -----------------------------
def dbfs(x: float) -> float:
    if x <= 0.0:
        return -120.0
    return 20.0 * np.log10(x)

# Keep ~10 seconds of history
FRAMES_PER_SEC = SAMPLE_RATE / CHUNK_SIZE
WINDOW = int(FRAMES_PER_SEC * 10)

rms_hist   = deque(maxlen=WINDOW)
delta_hist = deque(maxlen=WINDOW)

last_rms_db = None
last_stats  = 0.0

# -----------------------------
# Audio callback
# -----------------------------
def callback(indata, frames, time_info, status):
    global last_rms_db, last_stats

    if status:
        print("PortAudio:", status)

    audio = indata[:, 0].astype(np.float32)

    # Guard against ALSA nonsense
    if not np.isfinite(audio).all():
        print("NON-FINITE AUDIO (NaN/Inf)")
        return

    rms  = float(np.sqrt(np.mean(audio * audio)))
    peak = float(np.max(np.abs(audio)))

    rms_db  = dbfs(rms)
    peak_db = dbfs(peak)

    if last_rms_db is None:
        delta_db = 0.0
    else:
        delta_db = abs(rms_db - last_rms_db)

    last_rms_db = rms_db

    rms_hist.append(rms_db)
    delta_hist.append(delta_db)

    print(
        f"RMS {rms_db:6.1f} dBFS | "
        f"Peak {peak_db:6.1f} dBFS | "
        f"ΔRMS {delta_db:5.2f} dB"
    )

    now = time.time()
    if now - last_stats >= 2.0 and len(delta_hist) > 10:
        last_stats = now

        d50 = np.percentile(delta_hist, 50)
        d90 = np.percentile(delta_hist, 90)
        d99 = np.percentile(delta_hist, 99)

        r50 = np.percentile(rms_hist, 50)
        r90 = np.percentile(rms_hist, 90)

        print(
            "  [STATS ~10s] "
            f"RMS p50={r50:5.1f} p90={r90:5.1f} dBFS | "
            f"Δ p50={d50:4.2f} p90={d90:4.2f} p99={d99:4.2f} dB"
        )

# -----------------------------
# Main
# -----------------------------
print("Starting stream… Ctrl+C to stop")

with sd.InputStream(
    device=DEVICE_INDEX,
    channels=CHANNELS,
    samplerate=SAMPLE_RATE,
    blocksize=CHUNK_SIZE,
    dtype="float32",
    callback=callback,
):
    while True:
        time.sleep(0.5)
