from flask import jsonify
import web_ui.app as state

def _error(message, code=500):
    return jsonify({"status": "error", "message": message}), code
    
def _ok(**kwargs):
    payload = {"status": "ok"}
    payload.update(kwargs)
    return jsonify(payload)

def _locked():
    return jsonify({"status": "locked"}), 423

def _status_default():
    return {
        "running": False,
        "auto_start_error": state.auto_start_error
    }

def _recent_event(now, last_time, window=1.0):
    return bool((now - last_time) < window) if last_time > 0 else False
