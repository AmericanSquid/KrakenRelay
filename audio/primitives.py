import numpy as np

def timebase(duration, sample_rate):
    return np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

def sine(freq, t):
    return np.sin(2 * np.pi * freq * t)

def chirp(f0, f1, duration, t):
    k = (f1 - f0) / max(duration, 1e-6)
    phase = 2 * np.pi * (f0 * t + 0.5 * k * t * t)
    return np.sin(phase)

def decay(signal, t, strength, duration):
    signal *= np.exp(-strength * t / max(duration, 1e-6))
    return signal

def fade(signal, sample_rate, fade_ms):
    fade_len = int(sample_rate * (fade_ms / 1000))

    fade = np.linspace(0, 1, fade_len)
    signal[:fade_len] *= fade
    signal[-fade_len:] *= fade[::-1]

    return signal
    
def silence(length, dtype=np.int16):
    return np.zeros(length, dtype=dtype)

def to_pcm16(signal, volume, max_val):
    return (signal * volume * max_val).astype(np.int16)
    
# Process Audio
def sanitize_audio(samples):
    return np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    	
def compute_rms(samples):
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples * samples)))

def ensure_float32(samples):
    return np.asarray(samples, dtype=np.float32)
   
# IO
def pcm_to_int16_bytes(pcm):
    if isinstance(pcm, bytes):
        return pcm
    elif isinstance(pcm, np.ndarray) and pcm.dtype == np.int16:
        return pcm.tobytes()
    else:
        pcm = np.clip(np.asarray(pcm), -32768, 32767).astype(np.int16)
        return pcm.tobytes()
        
def fade_out(send_pcm, silence_bytes, num_chunks):
    for _ in range(num_chunks):
        send_pcm(silence_bytes)
