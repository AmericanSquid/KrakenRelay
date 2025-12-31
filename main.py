import sys
import time
import argparse
import logging
import signal
import threading
from typing import Optional

from config_manager import ConfigManager
from audio_manager import AudioDeviceManager, AudioDeviceError
from repeater_core import RepeaterController

shutdown_event = threading.Event()


#--------------------#
# Shutdown Handling  #
#--------------------#
def signal_handler(sig, frame):
    print("\n[KrakenRelay] Signal received — shutting down gracefully…")
    shutdown_event.set()
    raise SystemExit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

#------------------#
# Logging Helpers  #
#------------------#
def setup_logging(debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

#------------------#
#   CLI Helpers    #
#------------------#
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KrakenRelay – Repeater Controller (Flask web UI default; use --headless for CLI)"
    )

    parser.add_argument("--headless", action="store_true", help="Run without web UI (CLI mode).")
    parser.add_argument("--input", help="(Headless) Input audio device (name substring or index).")
    parser.add_argument("--output", help="(Headless) Output audio device (name substring or index).")
    parser.add_argument("--debug", action="store_true", help="Verbose logging (DEBUG).")
    parser.add_argument("--id-now", action="store_true", help="(Headless) Send CW ID immediately after startup.")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit.")

    # Web UI bind options
    parser.add_argument("--bind", default="0.0.0.0", help="(Web UI) Bind address (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=5000, help="(Web UI) Port (default: 5000).")

    return parser.parse_args()


#---------------#
# List Devices  #
#---------------#
def list_audio_devices() -> None:
    am = AudioDeviceManager()
    try:
        devices = am.list_devices()
        print("Available audio devices:")
        for device in devices:
            print(
                f"  {device['index']}: {device['name']} "
                f"(in={device['maxInputChannels']} out={device['maxOutputChannels']})"
            )
    finally:
        am.cleanup()

#----------------------------#
# Device Resolution Helpers  #
#----------------------------#
def _resolve_device(arg: Optional[str], am: AudioDeviceManager, direction: str) -> int:
    if arg is None:
        default_info = (
            am.pa.get_default_input_device_info()
            if direction == "input"
            else am.pa.get_default_output_device_info()
        )
        return int(default_info["index"])

    if arg.isdigit():
        return int(arg)

    idx = am.find_device_by_name(arg)
    if idx is None:
        raise ValueError(f"No {direction} device matches '{arg}'.")
    return idx

#----------------------------#
# Config Normalization       #
#----------------------------#
def normalize_config_for_template(cfg: dict) -> None:
    """
    Keep your template happy and provide sensible defaults.
    Current index.html uses flat ptt keys: ptt.mode / ptt.device_path / ptt.gpio_pin
    """
    cfg.setdefault("ptt", {})
    ptt = cfg["ptt"]
    if not isinstance(ptt, dict):
        cfg["ptt"] = {}
        ptt = cfg["ptt"]

    # Ensure flat keys exist
    ptt.setdefault("mode", "VOX")
    ptt.setdefault("device_path", "")
    ptt.setdefault("gpio_pin", 3)

    # Ensure top-level sections exist
    cfg.setdefault("audio", {})
    cfg.setdefault("repeater", {})
    cfg.setdefault("identification", {})
    cfg.setdefault("tot", {})

    # Reasonable default: do NOT autostart unless configured
    cfg["repeater"].setdefault("auto_start", False)

def _config_accepts_path() -> bool:
    try:
        ConfigManager("config.yaml")  # type: ignore[arg-type]
        return True
    except TypeError:
        return False
    except Exception:
        # If it accepts path but file missing etc, still "accepts path"
        return True

#------------------#
# Headless runner  #
#------------------#
def run_headless(args: argparse.Namespace) -> None:
    config = ConfigManager("config.yaml") if _config_accepts_path() else ConfigManager()
    if not getattr(config, "config", None):
        logging.warning("Config failed to load. Falling back to default config.")

    normalize_config_for_template(config.config)

    am = AudioDeviceManager()
    controller: Optional[RepeaterController] = None

    if args.input is None or args.output is None:
        print("\nNo input/output device specified — entering setup...")
        devices = am.list_devices()
        print("\nAvailable Audio Devices:")
        for dev in devices:
            print(
                f"{dev['index']}: {dev['name']} "
                f"(in={dev['maxInputChannels']} out={dev['maxOutputChannels']})"
            )
        try:
            args.input = input("\nEnter INPUT device index: ").strip()
            args.output = input("Enter OUTPUT device index: ").strip()
        except KeyboardInterrupt:
            print("\nCancelled by user.")
            am.cleanup()
            sys.exit(0)

    try:
        input_idx = _resolve_device(args.input, am, "input")
        output_idx = _resolve_device(args.output, am, "output")

        controller = RepeaterController(input_idx, output_idx, config, am)
        if args.id_now:
            controller.schedule_id.send_id()

        controller.start()
        logging.info("Headless mode running – Ctrl+C to stop.")

        while not shutdown_event.is_set():
            time.sleep(1)

    except AudioDeviceError as e:
        logging.critical(f"[Startup] Audio device initialization failed: {e}")
        raise SystemExit(1)

    finally:
        try:
            if controller:
                controller.cleanup()
        finally:
            try:
                am.cleanup()
            except Exception:
                pass

#-------------------#
#  Web UI runner    #
#-------------------#
def run_web(args: argparse.Namespace) -> None:
    import web_ui

    # Inject deps web_ui.py currently uses but doesn't import
    web_ui.RepeaterController = RepeaterController
    web_ui.AudioDeviceError = AudioDeviceError
    try:
        import numpy as np  # type: ignore
        web_ui.np = np
    except Exception:
        logging.warning("NumPy not available — /status may fail (np.log10 used).")

    # Create config + audio manager and attach to web_ui globals
    config = ConfigManager("config.yaml") if _config_accepts_path() else ConfigManager()
    if not getattr(config, "config", None):
        logging.warning("Config failed to load. Falling back to default config.")

    normalize_config_for_template(config.config)

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
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("[WebUI] Suppressing audio_manager.cleanup() to allow Start/Stop reuse.")

    am.cleanup = soft_cleanup  # type: ignore[assignment]

    # Publish globals for web_ui.py
    web_ui.config = config
    web_ui.audio_manager = am
    web_ui.controller = None
    # startup_error is used by /status (you’ll add it in web_ui.py)
    setattr(web_ui, "auto_start_error", None)

    # -----------------------------
    # AUTO START (web mode)
    # Only start if auto_start is true AND saved device indices exist AND are valid.
    # Otherwise: do NOT start; set startup_error for the UI.
    # -----------------------------
    repeater_cfg = config.config.get("repeater", {}) or {}
    audio_cfg = config.config.get("audio", {}) or {}

    auto = bool(repeater_cfg.get("auto_start", False))
    input_idx = audio_cfg.get("input_index")
    output_idx = audio_cfg.get("output_index")

    if auto:
        if input_idx is None or output_idx is None:
            msg = ("Auto-start is enabled, but no input/output device is saved yet. "
                   "Pick devices and press Start once (this will save them).")
            logging.error("[Startup] %s", msg)
            web_ui.auto_start_error = msg  # type: ignore[attr-defined]
        else:
            try:
                input_idx = int(input_idx)
                output_idx = int(output_idx)

                devs = am.list_devices()
                by_index = {d.get("index"): d for d in devs}
                if input_idx not in by_index or output_idx not in by_index:
                    msg = (f"Auto-start is enabled, but saved device indices are not present: "
                           f"input_index={input_idx}, output_index={output_idx}. "
                           f"Open the UI, pick devices, and press Start once.")
                    logging.error("[Startup] %s", msg)
                    web_ui.auto_start_error = msg  # type: ignore[attr-defined]
                else:
                    in_name = by_index[input_idx].get("name", "Unknown")
                    out_name = by_index[output_idx].get("name", "Unknown")
                    logging.info("[Startup] Auto-starting with input=%s (%s) output=%s (%s)",
                                 input_idx, in_name, output_idx, out_name)

                    web_ui.controller = RepeaterController(input_idx, output_idx, config, am)
                    web_ui.controller.start()
                    logging.info("[Startup] Auto-started repeater (web mode)")
            except Exception as e:
                logging.exception("[Startup] Auto-start failed")
                web_ui.auto_start_error = f"Auto-start failed: {e}"  # type: ignore[attr-defined]

    host = args.bind
    port = int(args.port)

    logging.info(f"Web UI listening on http://{host}:{port}")

    try:
        # Avoid reloader (forks a second process, breaks audio/PTT)
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        web_ui.app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
    finally:
        # Best-effort shutdown cleanup
        try:
            ctl = getattr(web_ui, "controller", None)
            if ctl:
                ctl.cleanup()
        except Exception:
            logging.exception("[WebUI] Controller cleanup failed during shutdown.")

        try:
            real_cleanup()
        except Exception:
            logging.exception("[WebUI] Audio cleanup failed during shutdown.")


# -------------#
# Entry-point  #
# -------------#
def main() -> None:
    args = parse_args()
    setup_logging(args.debug)

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    if args.headless:
        run_headless(args)
    else:
        run_web(args)


if __name__ == "__main__":
    main()
