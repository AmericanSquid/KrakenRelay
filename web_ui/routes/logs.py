from flask import Blueprint, request, jsonify
from ..utils import ui_log as utils
import web_ui.app as state
from .common import _ok, _error, _locked

logs_bp = Blueprint("logs", __name__)

def _int_arg(name, default):
    try:
        return int(request.args.get(name, default) or default)
    except Exception:
        return default

@logs_bp.route("/logs")
def logs_tail():
    """Return recent log entries for the UI."""
    after = _int_arg("after", 0)
    limit = _int_arg("limit", 200)

    # clamp
    limit = max(1, min(limit, 500))

    with utils._ui_log_lock:
        entries = [e for e in utils._ui_log_buf if int(e.get("id", 0)) > after]

    if len(entries) > limit:
        entries = entries[-limit:]

    return jsonify({"entries": entries})

@logs_bp.route("/logs/clear", methods=["POST"])
def logs_clear():
    """Clear the in-memory UI log buffer."""
    with utils._ui_log_lock:
        utils._ui_log_buf.clear()
    return _ok()
