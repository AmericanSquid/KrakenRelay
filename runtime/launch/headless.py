import argparse
import logging
import sys
import time
from typing import Optional

from config import (
    ConfigManager,
    normalize_config_for_template,
    _config_accepts_path
)
from audio import (
    AudioDeviceManager, 
    AudioDeviceError, 
    resolve_device
)
from core import RepeaterController

from .common import cleanup, cfg_bootstrap
from runtime.sig_handler import shutdown_event

def run_headless(args: argparse.Namespace) -> None:
    config, cfg = cfg_bootstrap() 

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

        controller.lifecycle.start()
        logging.info("Headless mode running – Ctrl+C to stop.")

        while not shutdown_event.is_set():
            time.sleep(1)

    except AudioDeviceError as e:
        logging.critical(f"[Startup] Audio device initialization failed: {e}")
        raise SystemExit(1)

    finally:
        cleanup(controller, am)
