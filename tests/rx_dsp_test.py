import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt

from kraken_dsp.kraken_dsp_wrapper import DSPChain


sr = 48000
duration = 2.0
t = np.arange(int(sr * duration)) / sr


# -----------------------------
# HUM (what we want to remove)
# -----------------------------

hum = 0.6 * np.sin(2 * np.pi * 120 * t)


# -----------------------------
# SYNTHETIC VOICE GENERATION
# -----------------------------

f0 = 120.0   # fundamental pitch (male voice)

voice = np.zeros_like(t)

# generate harmonics
for n in range(1, 20):
    voice += (1/n) * np.sin(2*np.pi*f0*n*t)

# formant shaping (vowel-like response)
def formant(freq, center, width):
    return np.exp(-((freq-center)/width)**2)

# build frequency response
fft_voice = np.fft.rfft(voice)
freqs = np.fft.rfftfreq(len(voice), 1/sr)

formant_env = (
    formant(freqs, 700, 200) +
    formant(freqs, 1200, 250) +
    formant(freqs, 2600, 300)
)

fft_voice *= formant_env

voice = np.fft.irfft(fft_voice)

voice *= 0.4 / np.max(np.abs(voice))


# -----------------------------
# COMBINE SIGNAL
# -----------------------------

signal = hum + voice

samples = (signal * 32767).astype(np.int16)


# -----------------------------
# DSP CHAIN
# -----------------------------

dsp = DSPChain()

dsp.configure_notch(
    enabled=True,
    freq_hz=117.0,
    q=40.0,
    harmonics=1,
    sample_rate=sr
)

chunk = 1920
filtered = np.zeros_like(samples)

for i in range(0, len(samples), chunk):
    block = samples[i:i+chunk]
    out = dsp.process_int16_to_int16(block)
    filtered[i:i+len(out)] = out


# -----------------------------
# FFT ANALYSIS
# -----------------------------

samples_f = samples.astype(np.float32)
filtered_f = filtered.astype(np.float32)

def spectrum(x):
    sp = 20*np.log10(np.abs(np.fft.rfft(x)) + 1e-9)
    fr = np.fft.rfftfreq(len(x), 1 / sr)
    return fr, sp

f1, s1 = spectrum(samples_f)
f2, s2 = spectrum(filtered_f)


# -----------------------------
# PLOT
# -----------------------------

plt.figure(figsize=(10,5))
plt.plot(f1, s1, label="Before Notch")
plt.plot(f2, s2, label="After Notch")

plt.xlim(0, 4000)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Notch Filter Test With Voice Spectrum")
plt.legend()

plt.show()
