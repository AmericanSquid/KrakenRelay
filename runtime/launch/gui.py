import argparse
import logging

from config import (
    ConfigManager,
    normalize_config_for_template, 
    _config_accepts_path
)

from .common import cleanup, cfg_bootstrap
from audio import AudioDeviceManager, AudioDeviceError
from core import RepeaterController
from runtime.logging_utils import debug_enabled

def run_web(args: argparse.Namespace) -> None:
    import web_ui.app as web_ui
    ui = web_ui

    # Inject deps web_ui.py currently uses but doesn't import
    ui.RepeaterController = RepeaterController
    ui.AudioDeviceError = AudioDeviceError

    try:
        import numpy as np  # type: ignore
        ui.np = np
    except Exception:
        logging.warning("NumPy not available — /status may fail (np.log10 used).")

    # Create config + audio manager and attach to web_ui globals
    config, cfg = cfg_bootstrap()

    am = AudioDeviceManager()

    # Helpers expected by the template/UI
    if not hasattr(am, "get_input_devices"):
        def get_input_devices():
            devs = am.list_devices()
            return [d for d in devs if d.get("maxInputChannels", 0) > 0]
        am.get_input_devices = get_input_devices  # type: ignore[attr-defined]

    if not hasattr(am, "get_output_devices"):
        def get_output_devices():
            devs = am.list_devices()
            return [d for d in devs if d.get("maxOutputChannels", 0) > 0]
        am.get_output_devices = get_output_devices  # type: ignore[attr-defined]

    # IMPORTANT:
    # Your web_ui.py /stop calls audio_manager.cleanup() and then keeps using the same instance.
    # If cleanup() terminates PortAudio, next start will fail.
    # So in web mode we "soft patch" cleanup to do nothing; on shutdown we call the real cleanup.
    real_cleanup = am.cleanup
    def soft_cleanup():
        if debug_enabled():
            logging.debug("[WebUI] Suppressing audio_manager.cleanup() to allow Start/Stop reuse.")
    am.cleanup = soft_cleanup  # type: ignore[assignment]

    # Publish globals for web_ui.py
    ui.config = config
    ui.audio_manager = am
    ui.controller = None

    set_error = lambda msg: setattr(ui, "auto_start_error", msg)

    def startup_error(msg):
        logging.error("[Startup] %s", msg)
        set_error(msg)
    # startup_error is used by /status (you’ll add it in web_ui.py)
    setattr(ui, "auto_start_error", None)

    # -----------------------------
    # AUTO START (web mode)
    # Only start if auto_start is true AND saved device indices exist AND are valid.
    # Otherwise: do NOT start; set startup_error for the UI.
    # -----------------------------
    repeater_cfg = cfg.get("repeater", {}) or {}
    audio_cfg = cfg.get("audio", {}) or {}

    auto = bool(repeater_cfg.get("auto_start", False))
    input_idx = audio_cfg.get("input_index")
    output_idx = audio_cfg.get("output_index")

    if auto:
        if input_idx is None or output_idx is None:
            msg = ("Auto-start is enabled, but no input/output device is saved yet. "
                   "Pick devices and press Start once (this will save them).")
            startup_error(msg) # type: ignore[attr-defined]
        else:
            try:
                input_idx = int(input_idx)
                output_idx = int(output_idx)

                devs = am.list_devices()
                by_index = {d.get("index"): d for d in devs}
                get_dev = by_index.get

                if input_idx not in by_index or output_idx not in by_index:
                    msg = (f"Auto-start is enabled, but saved device indices are not present: "
                           f"input_index={input_idx}, output_index={output_idx}. "
                           f"Open the UI, pick devices, and press Start once.")
                    startup_error(msg)  # type: ignore[attr-defined]
                else:
                    in_name = get_dev(input_idx).get("name", "Unknown")
                    out_name = get_dev(output_idx).get("name", "Unknown")
                    logging.info("[Startup] Auto-starting with input=%s (%s) output=%s (%s)",
                                 input_idx, in_name, output_idx, out_name)

                    ui.controller = RepeaterController(input_idx, output_idx, config, am)
                    ui.controller.lifecycle.start()
                    logging.info("[Startup] Auto-started repeater (web mode)")
            except Exception as e:
                logging.exception("[Startup] Auto-start failed")
                ui.auto_start_error = f"Auto-start failed: {e}"  # type: ignore[attr-defined]

    host = args.bind
    port = int(args.port)

    logging.info(f"Web UI listening on http://{host}:{port}")

    try:
        # Avoid reloader (forks a second process, breaks audio/PTT)
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        ui.app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
    finally:
        cleanup(ui.controller, am)
