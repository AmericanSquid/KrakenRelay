from flask import Blueprint

import web_ui.app as state

from .common import _error, _ok

id_bp = Blueprint("id", __name__)


@id_bp.route("/manual_id", methods=["POST"])
def manual_id():
    if state.lifecycle is None or not state.repeater_state.running:
        return _error("not_running", 400)

    state.lifecycle.request_manual_id()
    return _ok()
