import logging

from flask import Blueprint, jsonify

import web_ui.app as state

from ..common import _error, _ok

stop_bp = Blueprint("stop", __name__)


@stop_bp.route("/stop", methods=["POST"])
def stop_repeater():
    if not state.lifecycle:
        return jsonify({"status": "not_running"})

    with state.state_lock:
        try:
            ok = state.lifecycle.cleanup()
        except Exception:
            logging.exception("[WebUI] Runtime cleanup failed.")
            ok = False
        state.lifecycle = None

    if not ok:
        return _error("Cleanup incomplete. Service restart may be required.")
    return _ok(status="stopped")
