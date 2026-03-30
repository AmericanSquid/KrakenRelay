from flask import Blueprint, jsonify
import web_ui.app as state
import time
import math
import logging
from runtime.logging_utils import debug_enabled
from .common import _status_default, _recent_event

status_bp = Blueprint("status", __name__)

@status_bp.route("/status")
def get_status():
    c = state.controller  # snapshot
    cfg = state.config.config
    now = time.time()

    if c is None:
        return jsonify(_status_default())

    try:
        with state.state_lock:
            tx = bool(getattr(c.tx_state, "transmitting", False))

            # Prefer rx_db if you’ve added it; fallback to current_rms
            rx_db = getattr(c, "rx_db", None)
            if rx_db is None:
                current_level = float(getattr(c, "current_rms", 0.0) or 0.0)
                if current_level > 0:
                    current_db = 20.0 * math.log10(current_level / 32767.0)
                else:
                    current_db = -60.0
            else:
                current_db = float(rx_db)

            squelch_db = float(cfg.get("audio", {}).get("squelch_threshold", -40))
            rx = bool(current_db > squelch_db)

            # PTT status
            status_text, status_color = ("PTT: Unknown", "#ccc")
            if getattr(c, "ptt_manager", None):
                status_text, status_color = c.ptt_manager.get_ptt_status()

            # TOT
            tot_conf = cfg.get("tot", {})
            tot_enabled = bool(tot_conf.get("tot_enabled", False))
            tot_limit = float(tot_conf.get("tot_time", 0.0) or 0.0)
            tot_lockout = bool(c.tot_manager.is_locked()) if getattr(c, "tot_manager", None) else False

            tot_elapsed = 0.0
            lockout_remaining = 0.0
            if tot_enabled and getattr(c, "tot_manager", None):
                tm = c.tot_manager
                if tx and getattr(tm, "tx_start_time", None):
                    tot_elapsed = float(now - tm.tx_start_time)
                if tot_lockout and getattr(tm, "lockout_start", None):
                    lockout_time = float(tot_conf.get("tot_lockout_time", 0.0) or 0.0)
                    lockout_remaining = max(0.0, lockout_time - float(now - tm.lockout_start))

            # ID
            sending_id = bool(getattr(getattr(c, "schedule_id", None), "sending_id", False))
            next_id_in = None
            ident = cfg.get("identification", {})
            if bool(ident.get("cw_enabled", False)) and getattr(c, "schedule_id", None):
                interval = float(ident.get("interval_minutes", 10)) * 60.0
                last_id = float(getattr(c.schedule_id, "last_id_time", now) or now)
                next_id_in = max(0.0, interval - float(now - last_id))

            # clip/limit
            now = time.time()
            last_clip = float(getattr(c, "last_clip_time", 0.0) or 0.0)
            last_limit = float(getattr(c, "last_limit_time", 0.0) or 0.0)
            clipping = _recent_event(now, last_clip)
            limiting = _recent_event(now, last_limit)
            
            auto_start_err = getattr(c, "auto_start_error", None)

        return jsonify({
            "auto_start_error": state.auto_start_error,
            "running": True,
            "tx": tx,
            "rx": rx,
            "audio_db": float(round(current_db, 1)),
            "ptt_status_text": str(status_text),
            "ptt_status_color": str(status_color),
            "tot_enabled": tot_enabled,
            "tot_elapsed": float(round(tot_elapsed, 1)),
            "tot_limit": float(tot_limit),
            "tot_lockout": tot_lockout,
            "lockout_remaining": float(round(lockout_remaining, 1)),
            "sending_id": sending_id,
            "next_id_in": float(round(next_id_in, 1)) if next_id_in is not None else None,
            "clipping": clipping,
            "limiting": limiting,
        })
    except Exception as e:
        if debug_enabled():
            logging.debug("[WebUI] /status failed: %s", e)
        return jsonify(_status_default())
