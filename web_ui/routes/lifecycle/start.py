from flask import Blueprint, request, jsonify
import web_ui.app as state
import yaml
import logging
from ..common import _ok, _error, _locked

start_bp = Blueprint("start", __name__)

@start_bp.route('/start', methods=['POST'])
def start_repeater():
    with state.state_lock:
        if state.controller and getattr(state.controller, "running", False):
            return jsonify({"status": "already_running"}), 400

         # Get selected device indices from form data (fallback to config)
        raw_in = request.form.get("input_index", None)
        raw_out = request.form.get("output_index", None)
        cfg = state.config.config
        audio_cfg = (cfg.get("audio", {}) or {})
 
        if raw_in is None:
            raw_in = audio_cfg.get("input_index", None)
        if raw_out is None:
            raw_out = audio_cfg.get("output_index", None)

        try:
            input_idx = int(raw_in)
            output_idx = int(raw_out)

            if input_idx < 0 or output_idx < 0:
                raise ValueError

        except Exception:
            return _error("Invalid device indices", 400)

        # Persist chosen devices so auto-start can work after "Start once"
        try:
           with state._config_lock:
                cfg_audio = (cfg.setdefault("audio", {}) or {})
                cfg_audio["input_index"] = int(input_idx)
                cfg_audio["output_index"] = int(output_idx)

                # write immediately so reboot auto-start has the saved indices
                path = state._config_path()
                with open(path, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
        except Exception as e:
            return _error(f"Failed to save device selection: {e}", 500)

        cfg['audio']['input_index'] = input_idx
        cfg['audio']['output_index'] = output_idx
        state.config.save_config()

        try:
            # Initialize and start the RepeaterController in a background thread
            state.controller = state.RepeaterController(input_idx, output_idx, state.config, state.audio_manager)
            state.controller.lifecycle.start()  # starts the audio thread
            state.auto_start_error = None
            return jsonify({"status": "running"})
        except state.AudioDeviceError as e:
            return _error(str(e), 500)
