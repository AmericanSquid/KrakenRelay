from .state import SignalGateState
from runtime.logging_utils import debug_enabled

import logging

def _open_squelch(state: SignalGateState, now):
    state.squelch_open = True
    state.squelch_open_time = now
    state._last_above_squelch = now
    state.kerchunk_buffer = []

def _close_squelch(state: SignalGateState):
    state.squelch_open = False

def update_squelch_state(state: SignalGateState, audio_cfg: dict, level_db: float, now: float) -> bool:
    """
    Debounced squelch:
    - Open when level_db > open_thr
    - Once open, stay open until level_db < close_thr for >= hang_time
    """
    open_thr = float(audio_cfg.get("squelch_threshold", -40))
    hyst_db = float(audio_cfg.get("squelch_hysteresis_db", 3.0))
    hang_time = float(audio_cfg.get("squelch_hang_time", 0.25))

    close_thr = open_thr - abs(hyst_db)
    debug_on = debug_enabled()

    if not state.squelch_open:
        if level_db > open_thr:
            _open_squelch(state, now)
            if debug_on:
                logging.debug("[Squelch] OPEN db=%.1f thr=%.1f", level_db, open_thr)
    else:
        if level_db > close_thr:
            state._last_above_squelch = now
        elif (now - state._last_above_squelch) >= hang_time:
            _close_squelch(state)
            if debug_on:
                logging.debug(
                    "[Squelch] CLOSE db=%.1f close_thr=%.1f hang=%.2f",
                    level_db, close_thr, hang_time
                )

    return state.squelch_open
