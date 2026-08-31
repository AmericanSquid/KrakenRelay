import yaml
from flask import Blueprint, jsonify, request

import web_ui.app as state
from web_ui.utils.config import _config_path

from ..common import _error

start_bp = Blueprint("start", __name__)


@start_bp.route("/start", methods=["POST"])
def start_repeater():
    with state.state_lock:
        if state.lifecycle and state.repeater_state.running:
            return jsonify({"status": "already_running"}), 400

        # Get selected device indices from form data (fallback to config)
        raw_in = request.form.get("input_index", None)
        raw_out = request.form.get("output_index", None)
        raw_in2 = request.form.get("input_index_2", None)
        raw_out2 = request.form.get("output_device_2", None)
        cfg = state.config.config
        audio_cfg = cfg.get("audio", {}) or {}

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
                cfg_audio = cfg.setdefault("audio", {}) or {}
                cfg_audio["input_index"] = int(input_idx)
                cfg_audio["output_index"] = int(output_idx)

                if raw_in2 not in (None, ""):
                    try:
                        input_idx_2 = int(raw_in2)
                        cfg_audio["input_index_2"] = input_idx_2
                        cfg_audio["input_device_2"] = input_idx_2
                    except Exception:
                        pass

                if raw_out2 not in (None, ""):
                    try:
                        cfg_audio["output_device_2"] = int(raw_out2)
                    except Exception:
                        pass

                # write immediately so reboot auto-start has the saved indices
                path = _config_path()
                with open(path, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
        except Exception as e:
            return _error(f"Failed to save device selection: {e}", 500)

        cfg["audio"]["input_index"] = input_idx
        cfg["audio"]["output_index"] = output_idx
        state.config.save_config()

        try:
            state.lifecycle = state.build_repeater(
                input_idx,
                output_idx,
                state.config,
                state.audio_manager,
                audit=getattr(state, "audit", None),
                publish_services=state.publish_services,
            )
            state.lifecycle.start()
            state.auto_start_error = None
            return jsonify({"status": "running"})
        except state.AudioDeviceError as e:
            return _error(str(e), 500)
