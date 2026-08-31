import argparse
import logging
import sys
import time

from audio import AudioDeviceError, AudioDeviceManager, resolve_device
from core import build_repeater
from runtime.sig_handler import shutdown_event

from .common import cfg_bootstrap, cleanup

log = logging.getLogger(__name__)


def run_headless(args: argparse.Namespace) -> None:
    config, _cfg = cfg_bootstrap()

    am = AudioDeviceManager()
    lifecycle = None

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
        input_idx = resolve_device(args.input, am, "input")
        output_idx = resolve_device(args.output, am, "output")

        lifecycle = build_repeater(input_idx, output_idx, config, am)
        if args.id_now:
            lifecycle.send_id()

        lifecycle.start()
        log.info("Headless mode running – Ctrl+C to stop.")

        while not shutdown_event.is_set():
            time.sleep(1)

    except AudioDeviceError as e:
        log.critical("[Startup] Audio device initialization failed: %s", e)
        raise SystemExit(1)

    finally:
        cleanup(lifecycle, am)
