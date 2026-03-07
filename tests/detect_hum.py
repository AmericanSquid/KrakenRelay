import numpy as np
import sounddevice as sd
import time

SAMPLE_RATE = 48000
BLOCK_SIZE = 4096

LOW_FREQ = 40
HIGH_FREQ = 200


def detect_hum(samples, sample_rate):
    """Return dominant hum frequency."""
    
    samples = samples.astype(np.float32)

    # windowing reduces FFT artifacts
    samples *= np.hanning(len(samples))

    spectrum = np.abs(np.fft.rfft(samples))
    freqs = np.fft.rfftfreq(len(samples), 1 / sample_rate)

    mask = (freqs >= LOW_FREQ) & (freqs <= HIGH_FREQ)

    if not np.any(mask):
        return None

    peak = np.argmax(spectrum[mask])
    return freqs[mask][peak], spectrum[mask][peak]


def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)

    samples = indata[:, 0]

    result = detect_hum(samples, SAMPLE_RATE)

    if result:
        freq, strength = result
        print(f"Dominant hum: {freq:.2f} Hz")


def main():

    print("\nListening for hum...")
    print("Press CTRL+C to stop\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK_SIZE,
        channels=1,
        callback=audio_callback,
    ):
        while True:
            time.sleep(1)


if __name__ == "__main__":
    main()
