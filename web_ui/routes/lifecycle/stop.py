from flask import Blueprint, jsonify
import web_ui.app as state
import logging
from ..common import _ok, _error, _locked

stop_bp = Blueprint("stop", __name__)

@stop_bp.route('/stop', methods=['POST'])
def stop_repeater():
    if not state.controller:
        return jsonify({"status": "not_running"})
    
    # Signal the controller to stop
    with state.state_lock:
        try:
            ok = state.controller.lifecycle.cleanup()
        except Exception as e:
            logging.exception("[WebUI] Controller cleanup failed.")
            ok = False
        state.controller = None

    if not ok:
        return _error("Cleanup incomplete. Service restart may be required.")
    return _ok(status="stopped")

   # Also close audio streams and reset controller
   # state.audio_manager.cleanup()
   # state.controller = None
   # return jsonify({"status": "stopped"})
