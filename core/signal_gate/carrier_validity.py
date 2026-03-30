from .state import SignalGateState
from runtime.logging_utils import debug_enabled
import logging

def _start_probe(state:SignalGateState, now, raw_level_db):
    state._carrier_valid = False
    state._carrier_probe_start = now
    state._carrier_last_level_db = raw_level_db


def _compute_delta(state: SignalGateState, raw_level_db):
    # Compute movement since last frame USING RAW LEVEL
    last = state._carrier_last_level_db
    delta = 0.0 if last is None else abs(raw_level_db - float(last))
    state._carrier_last_level_db = raw_level_db
    return delta

def _mark_valid(state: SignalGateState, now):
    state._carrier_valid = True
    state._carrier_probe_start = now

def _probe_expired(state: SignalGateState, now, timeout):
    start = state._carrier_probe_start or now
    return (now - float(start)) >= timeout

def _reset_carrier(state: SignalGateState, tx):
    state.squelch_open = False
    state.squelch_open_now = False
    state._carrier_valid = False
    state._carrier_probe_start = None
    state._carrier_last_level_db = None
    state.kerchunk_buffer = []
    tx.tx_start_pending = False

def _buffer_sample(state: SignalGateState, samples):
    state.kerchunk_buffer.append(samples.copy())
    if len(state.kerchunk_buffer) > state.kerchunk_backlog_max:
        state.kerchunk_buffer.pop(0)

def carrier_validity_probe(state: SignalGateState, audio_cfg: dict, controller, raw_level_db, now, just_opened, samples):
    cv_min_delta = float(getattr(audio_cfg, "carrier_validity_min_delta", 0.1))
    cv_timeout = float(getattr(audio_cfg, "carrier_validity_timeout", 2.5))

    tx = controller.tx_state

    # Start / reset on a fresh open edge
    if just_opened or state._carrier_probe_start is None:
        _start_probe(state, now, raw_level_db)

    delta = _compute_delta(state, raw_level_db)

    if delta >= cv_min_delta:
        # We saw "movement" -> mark valid and reset the no-movement timer
        if not state._carrier_valid:
            if debug_enabled():
                logging.debug("[CarrierValidity] VALID delta=%.3f >= %.3f", delta, cv_min_delta)
        _mark_valid(state, now)
        return
    
    # No movement: if it's been too long, declare invalid and drop
    if _probe_expired(state, now, cv_timeout):
        if debug_enabled():
            logging.debug(
                "[CarrierValidity] INVALID (delta=%.3f < %.3f for %.2fs)",
                delta, cv_min_delta, cv_timeout
            )

        _reset_carrier(state, tx)

        # If we were already keyed, drop immediately and do NOT courtesy-beep
        if tx.transmitting:
            tx.skip_courtesy_tone = True
            controller.tx_control.stop()
        return

    # While still "probing", only buffer pre-TX so we don't key early
    if not tx.transmitting:
        _buffer_sample(state, samples)
        return
