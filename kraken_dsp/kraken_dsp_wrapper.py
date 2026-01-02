import numpy as np
from ._kraken_dsp import ffi, lib

def _coeff(dt, ms):
    return np.exp(-dt / max(ms * 0.001, 1e-6))

class DSPChain:
    def __init__(self):
        self._ch = ffi.new("DSPChain *")
        lib.dspchain_init(self._ch)

    def reset(self):
        lib.dspchain_reset(self._ch)

    def configure_hpf(self, enabled, order, cutoff_hz, sample_rate):
        lib.dspchain_set_hpf(self._ch, int(enabled), order, cutoff_hz, sample_rate)

    def configure_compressor(
        self,
        enabled,
        threshold_db,
        ratio,
        sample_rate,
        chunk_len,
        attack_ms=10.0,
        release_ms=200.0,
        makeup_db=0.0
    ):
        dt = 1 / sample_rate
        att = _coeff(dt, attack_ms)
        rel = _coeff(dt, release_ms)
        makeup = 10.0 ** (makeup_db / 20.0)

        lib.dspchain_set_compressor(
            self._ch,
            int(enabled),
            threshold_db,
            ratio,
            att,
            rel,
            makeup
        )

    def configure_limiter(self, enabled, threshold, sample_rate, chunk_len,
                          attack_ms=2.0, release_ms=80.0):
        dt = chunk_len / sample_rate
        att = _coeff(dt, attack_ms)
        rel = _coeff(dt, release_ms)

        lib.dspchain_set_limiter(
            self._ch,
            int(enabled),
            threshold,
            att,
            rel
        )

    def process(self, x_float):
        buf = np.asarray(x_float, dtype=np.float32, order="C")
        lib.dspchain_process_inplace(
            self._ch,
            ffi.cast("float *", buf.ctypes.data),
            buf.size
        )
        return buf

    def process_int16_to_int16(self, x_i16: np.ndarray) -> np.ndarray:
        x = x_i16.astype(np.float32) / 32768.0
        y = self.process(x)
        return np.clip(y * 32768.0, -32768, 32767).astype(np.int16)
