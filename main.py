import sys
import time
import argparse
import logging
import signal
import threading
from typing import Optional
from config_manager import ConfigManager
from audio_manager import AudioDeviceManager
from repeater_core import RepeaterController

# GUI imports will be loaded conditionally in run_gui()
QApplication = None
QTimer = None
RepeaterUI = None

#-------------------#
# Shutdown Handling #
#-------------------#
app = None
shutdown_event = threading.Event()

def signal_handler(sig, frame):
    print("\n[KrakenRelay] Signal received — shutting down gracefully…")
    shutdown_event.set()
    global app
    if app is not None:
        app.quit()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
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

    # If no devices specified, run interactive prompt
    if args.input is None or args.output is None:
        print("\nNo input/output device specified — entering setup...")
        devices = audio_manager.list_devices()
        print("\nAvailable Audio Devices:")
        for dev in devices:
            print(f"{dev['index']}: {dev['name']} "
                  f"(in={dev['maxInputChannels']} out={dev['maxOutputChannels']})")
        try:
            args.input = input("\nEnter INPUT device index: ").strip()
            args.output = input("Enter OUTPUT device index: ").strip()
        except KeyboardInterrupt:
            print("\nCancelled by user.")
            audio_manager.cleanup()
            sys.exit(0)

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
        while not shutdown_event.is_set():
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
    global app, QApplication, RepeaterUI
    try:
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QTimer
        from ui import RepeaterUI
    except ImportError:
        logging.error("PyQt5 or UI components not installed. Cannot run GUI mode.")
        sys.exit(1)

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
