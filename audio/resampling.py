# audio/resampling.py

import logging
import time

import numpy as np

from runtime.logging_utils import debug_enabled

log = logging.getLogger(__name__)

_last_call_log_time = 0.0
_call_count = 0
_last_resample_log_time = 0.0


def resample_to_length(
    samples,
    target_len,
    *,
    label="audio",
    call_log_interval=5.0,
    resample_log_interval=5.0,
):
    """
    Resample a 1D audio chunk to exactly target_len samples.

    This is intentionally lightweight. It uses linear interpolation for small
    chunk-size corrections / drift experiments.

    Logs:
      - a rate-limited heartbeat showing call count and latest input length
      - a rate-limited correction log only when length actually changes
    """
    global _last_call_log_time, _call_count, _last_resample_log_time

    target_len = int(target_len)
    _call_count += 1

    now = time.monotonic()

    input_len = 0
    if samples is not None:
        try:
            input_len = np.asarray(samples).size
        except Exception:
            input_len = -1

    if now - _last_call_log_time >= call_log_interval:
        debug_on = debug_enabled()
        if debug_on:
            log.debug(
                "[Resampling] %s called %s times; latest len=%s target=%s",
                label,
                _call_count,
                input_len,
                target_len,
            )
        _last_call_log_time = now

    if target_len <= 0:
        log.warning(
            "[Resampling] %s requested invalid target_len=%s", label, target_len
        )
        return np.zeros(0, dtype=np.float32)

    if samples is None:
        log.warning(
            "[Resampling] %s samples=None; returning silence len=%s", label, target_len
        )
        return np.zeros(target_len, dtype=np.float32)

    samples = np.asarray(samples, dtype=np.float32)
    original_len = samples.size

    if original_len == 0:
        log.warning(
            "[Resampling] %s empty input; returning silence len=%s", label, target_len
        )
        return np.zeros(target_len, dtype=np.float32)

    if original_len == target_len:
        return samples.astype(np.float32, copy=False)

    if now - _last_resample_log_time >= resample_log_interval:
        log.info(
            "[Resampling] %s length correction: %s -> %s samples",
            label,
            original_len,
            target_len,
        )
        _last_resample_log_time = now

    old_x = np.linspace(0.0, 1.0, num=original_len, endpoint=False)
    new_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)

    corrected = np.interp(new_x, old_x, samples).astype(np.float32)

    if not np.all(np.isfinite(corrected)):
        log.warning("[Resampling] %s produced non-finite samples; sanitizing", label)
        corrected = np.nan_to_num(
            corrected,
            nan=0.0,
            posinf=32767.0,
            neginf=-32768.0,
        ).astype(np.float32)

    return corrected
