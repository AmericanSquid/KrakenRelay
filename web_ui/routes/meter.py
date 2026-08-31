import logging
import time

from flask import Blueprint, jsonify

import web_ui.app as state
from runtime.logging_utils import debug_enabled

from .common import _recent_event

meter_bp = Blueprint("meter", __name__)


def _meter_default():
    return {
        "running": False,
        "rx_db": -60.0,
        "tx_db": -60.0,
        "rx_peak_db": -60.0,
        "tx_peak_db": -60.0,
        "clipping": False,
        "limiting": False,
    }


@meter_bp.route("/meter")
def meter():
    if state.lifecycle is None:
        return jsonify(_meter_default())

    lock = getattr(state.meter, "lock", state.state_lock)

    try:
        with lock:
            rx_db = state.meter.rx_db
            tx_db = state.meter.tx_db
            rx_peak_db = state.meter.rx_peak_db
            tx_peak_db = state.meter.tx_peak_db

            now = time.time()
            last_clip = float(state.repeater_state.last_clip_time or 0.0)
            last_limit = float(state.repeater_state.last_limit_time or 0.0)

            clipping = _recent_event(now, last_clip)
            limiting = _recent_event(now, last_limit)

        return jsonify(
            {
                "running": True,
                "rx_db": round(rx_db, 1),
                "tx_db": round(tx_db, 1),
                "rx_peak_db": round(rx_peak_db, 1),
                "tx_peak_db": round(tx_peak_db, 1),
                "clipping": clipping,
                "limiting": limiting,
            }
        )

    except Exception as e:
        if debug_enabled():
            logging.debug("[WebUI] /meter failed: %s", e)
        return jsonify(_meter_default())
