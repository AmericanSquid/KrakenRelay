import sys
import time
import argparse
import logging
from typing import Optional
from PyQt5.QtWidgets import QApplication
from config_manager import ConfigManager
from audio_manager import AudioDeviceManager
from repeater_core import RepeaterController
from ui import RepeaterUI

#-----------------#
# Logging Helpers #
#-----------------#
def setup_logging(debug: bool = False) -> None:
    """Initialise root logger for the entire application."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
#-----------------#
#   CLI Helpers   #
#-----------------#
def parse_args() -> argparse.Namespace:
    """Parse CLI flags."""
    parser = argparse.ArgumentParser(
        description="KrakenRelay – An Open Source Repeater Controller by American Squid/K3AYV"
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the GUI.",
    )
    parser.add_argument(
        "--input",
        help="Input audio device (name substring or index).",
    )
    parser.add_argument(
        "--output",
        help="Output audio device (name substring or index).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Verbose logging (sets log level to DEBUG).",
    )
    parser.add_argument(
        "--id-now",
        action="store_true",
        help="Send a single CW ID immediately after startup.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit.",
    )
    return parser.parse_args()

#--------------#
# List Devices #
#--------------#

def list_audio_devices():
    audio_manager = AudioDeviceManager()
    devices = audio_manager.list_devices()
    print("Available audio devices:")
    for device in devices:
        print(f"  {device['index']}: {device['name']} (in={device['maxInputChannels']} out={device['maxOutputChannels']})")
    audio_manager.cleanup()

#---------------------------#
# Device Resolution Helpers #
#---------------------------#

def _resolve_device(arg: Optional[str], am: AudioDeviceManager, direction: str) -> int:
    """Convert a name/idx CLI arg into a PyAudio device index."""
    if arg is None:
        default_info = (
            am.pa.get_default_input_device_info()
            if direction == "input"
            else am.pa.get_default_output_device_info()
        )
        return int(default_info["index"])

    # If the arg is purely digits we assume it is already an index
    if arg.isdigit():
        return int(arg)

    # Otherwise search (case‑insensitively) for a device whose name contains arg
    idx = am.find_device_by_name(arg)
    if idx is None:
        raise ValueError(f"No {direction} device matches '{arg}'.")
    return idx

#-----------------#
# Headless runner #
#-----------------#

def run_headless(args: argparse.Namespace) -> None:
    """Run repeater controller loop without a GUI.

    Ctrl‑C cleans up streams and exits gracefully. Suitable for running under
    systemd or tmux where a clean shutdown is preferred over auto‑restart.
    """
    config = ConfigManager()
    if not config.config:
        logging.warning("Config failed to load. Falling back to default config.")
    audio_manager = AudioDeviceManager()

    try:
        input_idx = _resolve_device(args.input, audio_manager, "input")
        output_idx = _resolve_device(args.output, audio_manager, "output")
    except Exception as err:
        logging.error(err)
        audio_manager.cleanup()
        sys.exit(1)

    controller = RepeaterController(input_idx, output_idx, config, audio_manager)

    if args.id_now:
        controller.send_id()

    controller.start()
    logging.info("Headless mode running – Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Ctrl+C detected – shutting down…")
    finally:
        controller.cleanup()
        audio_manager.cleanup()

#------------#
# GUI runner #
#------------#

def run_gui() -> None:
    app = QApplication(sys.argv)
    config = ConfigManager()
    if not config.config:
        logging.warning("Config failed to load. Falling back to default config.")
    audio_manager = AudioDeviceManager()
    ui = RepeaterUI(config, audio_manager)
    ui.show()
    sys.exit(app.exec_())

#-------------#
# Entry‑point #
#-------------#

def main() -> None:
    args = parse_args()
    setup_logging(args.debug)

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)   # <- Make sure to exit before GUI or repeater code!

    if args.headless:
        run_headless(args)
    else:
        run_gui()


if __name__ == "__main__":
    main()
