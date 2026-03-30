import numpy as np
from audio.primitives import sine, to_pcm16, silence
from .constants import MORSE_CODE, PCM16_MAX

class CWGenerator:
    def __init__(self, wpm=20, frequency=800, sample_rate=8000, volume=0.5):
        self.dot_len = int(1.2 / wpm * sample_rate)
        self.dash_len = self.dot_len * 3
        self.sample_rate = sample_rate
        self.frequency = frequency
        self.volume = volume
        self._intra = silence(self.dot_len)
        self._inter_char = silence(self.dot_len * 2)

    def tone(self, length):
        t = np.arange(length) / self.sample_rate
        signal = sine(self.frequency, t)
        return to_pcm16(signal, self.volume, PCM16_MAX)

    def generate_chunks(self, text, chunk_size):
        """Yield fixed-size int16 PCM chunks for real-time playback (timing-accurate)."""
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        buf = np.zeros(0, dtype=np.int16)

        def append(seg: np.ndarray):
            nonlocal buf
            if seg.size == 0:
                return
            if buf.size:
                buf = np.concatenate((buf, seg))
            else:
                buf = seg

        def flush_full_chunks():
            nonlocal buf
            while buf.size >= chunk_size:
                out = buf[:chunk_size]
                buf = buf[chunk_size:]
                yield out
        
        def append_and_flush(seg):
            append(seg)
            yield from flush_full_chunks()

        for char in text.upper():
            if char not in MORSE_CODE:
                continue

            for symbol in MORSE_CODE[char]:
                # tone
                tone_len = self.dot_len if symbol == "." else self.dash_len
                yield from append_and_flush(self.tone(tone_len))

                # intra-symbol gap (1 dot)
                yield from append_and_flush(self._intra)

            # inter-letter gap (2 dots here, because we already added 1 dot after last element)
            yield from append_and_flush(self._inter_char)

        # Pad ONLY ONCE at the end so timing doesn't get stretched per element
        if buf.size:
            pad = silence(chunk_size - buf.size)
            yield np.concatenate((buf, pad))
