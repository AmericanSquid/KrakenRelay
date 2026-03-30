import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KrakenRelay – Repeater Controller (Web UI by default; use --headless for CLI)"
    )

    # Headless options
    parser.add_argument("--headless", action="store_true", help="Run without web UI (CLI mode).")
    parser.add_argument("--input", help="(Headless) Input audio device (name substring or index).")
    parser.add_argument("--output", help="(Headless) Output audio device (name substring or index).")
    parser.add_argument("--id-now", action="store_true", help="(Headless) Send CW ID immediately after startup.")

    # Web UI bind options
    parser.add_argument("--bind", default="0.0.0.0", help="(Web UI) Bind address (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=6973, help="(Web UI) Port (default: 6973).")

    # General optiosn
    parser.add_argument("--debug", action="store_true", help="Verbose logging (DEBUG).")
    parser.add_argument("--list-devices", action="store_true", help="List available audio devices and exit.")

    return parser.parse_args()
