from flask import Blueprint, request, jsonify
import yaml
import web_ui.app as state
from .common import _ok, _error, _locked

config_bp = Blueprint("config", __name__)

@config_bp.route("/config/live", methods=["POST"])
def config_live():
    data = request.json or {}
    key = data.get("key")
    value = data.get("value")

    cfg = state.config.config

    if not key or not isinstance(key, str):
        return _error("Missing key", 400)

    with state._config_lock:
        if state.config_locked:
            return _locked()
        state._set_path(cfg, key, state._coerce(value))
    return _ok()


@config_bp.route("/config/apply", methods=["POST"])
def config_apply():
    cfg = state.config.config
    try:
        with state._config_lock:
            path = state._config_path()
            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

        if state.controller:
            with state.state_lock:
                state.controller.lifecycle.cleanup()

            input_idx = state.controller.audio_io.input_device
            output_idx = state.controller.audio_io.output_device
            state.controller = None

            try:
                state.controller = state.RepeaterController(input_idx, output_idx, state.config, state.audio_manager)
                state.controller.lifecycle.start()

            except Exception as e:
                return _error(f"Failed to restart: {e}", 500)

        return _ok()
    except Exception as e:
        return _error(str(e), 500)
