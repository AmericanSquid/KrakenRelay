from flask import Blueprint, jsonify
import web_ui.app as state
from .common import _ok

maintenance_bp = Blueprint("maintenance", __name__)

def _set_flag(attr, response_key, value):
    setattr(state, attr, value)
    return _ok(**{response_key: value})

def maintenance_on():
    return _set_flag("maintenance_mode", "maintenance", True)

def maintenance_off():
    return _set_flag("maintenance_mode", "maintenance", False)

def maintenance_restart():
    return _set_flag("restarting", "restarting", True)

def restart_clear():
    return _set_flag("restarting", "restarting", False)

@maintenance_bp.route("/maintenance")
def maintenance_status():
    return jsonify({
        "maintenance": state.maintenance_mode,
        "restarting": state.restarting
    })
