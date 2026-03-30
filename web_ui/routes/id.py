from flask import Blueprint, request, jsonify
import web_ui.app as state
from .common import _ok, _error

id_bp = Blueprint("id", __name__)

@id_bp.route("/manual_id", methods=["POST"])
def manual_id():
    if state.controller is None or not getattr(state.controller, "running", False):
        return _error("not_running", 400)

    state.controller.request_cw.request_manual_id()
    return _ok()
