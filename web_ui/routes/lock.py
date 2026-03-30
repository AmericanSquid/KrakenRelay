from flask import Blueprint, request, jsonify
import web_ui.app as state
from .common import _ok, _error, _locked

lock_bp = Blueprint("lock", __name__)

@lock_bp.route("/lock", methods=["GET", "POST"])
def lock_config():
    """
    GET -> {"locked": bool}
    POST -> {"status":"ok","locked": bool}
    """
    if request.method == "GET":
        with state._config_lock:
            return jsonify({"locked": bool(state.config_locked)})

    locked = bool((request.json or {}).get("locked", True))
    with state._config_lock:
        state.config_locked = locked
        return jsonify({"locked": bool(state.config_locked)})
