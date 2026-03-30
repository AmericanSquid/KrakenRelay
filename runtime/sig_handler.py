import signal
import threading
import logging

shutdown_event = threading.Event()

def signal_handler(sig, frame):
    logging.info("[KrakenRelay] Signal received — shutting down gracefully…")
    shutdown_event.set()
    raise SystemExit(0)

for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, signal_handler)
