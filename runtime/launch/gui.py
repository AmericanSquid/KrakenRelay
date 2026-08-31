import argparse
import logging

from audio import AudioDeviceError, AudioDeviceManager
from audio.device_identity import resolve_saved_device
from core import build_repeater
from runtime.audit import AuditManager
from runtime.logging_utils import debug_enabled

from .common import cfg_bootstrap, cleanup


def run_web(args: argparse.Namespace) -> None:
    import web_ui.app as web_ui

    ui = web_ui

    # Inject deps web_ui routes currently use but do not import directly.
    ui.build_repeater = build_repeater
    ui.AudioDeviceError = AudioDeviceError

    try:
        import numpy as np  # type: ignore

        ui.np = np
    except Exception:
        logging.warning("NumPy not available - /status may fail if np.log10 is used.")

    # Create config + audio manager and attach to web_ui globals.
    config, cfg = cfg_bootstrap()

    am = AudioDeviceManager()

    # Helpers expected by the template/UI.
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
    # /stop calls audio_manager.cleanup() and then keeps using the same instance.
    # If cleanup() terminates PortAudio, the next Start can fail.
    # So in web mode we soft-patch cleanup to do nothing; on shutdown we call real cleanup.
    def soft_cleanup():
        if debug_enabled():
            logging.debug(
                "[WebUI] Suppressing audio_manager.cleanup() to allow Start/Stop reuse."
            )

    am.cleanup = soft_cleanup  # type: ignore[assignment]

    # Publish globals for route modules.
    ui.config = config
    ui.audio_manager = am
    ui.lifecycle = None
    ui.audit = AuditManager()
    ui.audit.start()

    def publish_services(**services):
        for name, service in services.items():
            setattr(ui, name, service)

    ui.publish_services = publish_services

    def set_error(msg):
        setattr(ui, "auto_start_error", msg)

    def startup_error(msg):
        logging.error("[Startup] %s", msg)
        set_error(msg)

    setattr(ui, "auto_start_error", None)

    # -----------------------------
    # AUTO START (web mode)
    # Resolve saved device descriptors first, with legacy index fallback.
    # -----------------------------
    repeater_cfg = cfg.get("repeater", {}) or {}
    audio_cfg = cfg.get("audio", {}) or {}

    auto = bool(repeater_cfg.get("auto_start", False))

    input_saved = audio_cfg.get("input_device")
    output_saved = audio_cfg.get("output_device")

    legacy_input_idx = audio_cfg.get("input_index")
    legacy_output_idx = audio_cfg.get("output_index")

    if auto:
        try:
            devs = am.list_devices()

            input_match = resolve_saved_device(
                saved=input_saved,
                devices=devs,
                direction="input",
                legacy_index=legacy_input_idx,
            )

            output_match = resolve_saved_device(
                saved=output_saved,
                devices=devs,
                direction="output",
                legacy_index=legacy_output_idx,
            )

            input_idx = int(input_match.index)
            output_idx = int(output_match.index)

            # Heal config in memory so anything later in this run sees the resolved form.
            audio_cfg["input_device"] = input_match.descriptor
            audio_cfg["output_device"] = output_match.descriptor
            audio_cfg["input_index"] = input_idx
            audio_cfg["output_index"] = output_idx

            by_index = {}
            for d in devs:
                try:
                    by_index[int(d.get("index"))] = d
                except Exception:
                    continue

            in_name = by_index.get(input_idx, {}).get("name", "Unknown")
            out_name = by_index.get(output_idx, {}).get("name", "Unknown")

            logging.info(
                "[Startup] Auto-starting with input=%s (%s) output=%s (%s)",
                input_idx,
                in_name,
                output_idx,
                out_name,
            )

            if input_match.warning:
                logging.warning("[Startup] %s", input_match.warning)

            if output_match.warning:
                logging.warning("[Startup] %s", output_match.warning)

            ui.lifecycle = build_repeater(
                input_idx,
                output_idx,
                config,
                am,
                audit=ui.audit,
                publish_services=publish_services,
            )
            ui.lifecycle.start()

            logging.info("[Startup] Auto-started repeater (web mode)")

        except ValueError as e:
            message = (
                f"Auto-start skipped: {e} "
                "Select currently connected devices in the web UI, then press Start."
            )
            logging.warning("[Startup] %s", message)
            set_error(message)
        except Exception as e:
            logging.exception("[Startup] Auto-start failed")
            set_error(f"Auto-start failed: {e}")

    host = args.bind
    port = int(args.port)

    logging.info(f"Web UI listening on http://{host}:{port}")

    try:
        # Avoid reloader, which forks a second process and breaks audio/PTT.
        logging.getLogger("werkzeug").setLevel(logging.WARNING)
        ui.app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
    finally:
        cleanup(ui.lifecycle, am)
