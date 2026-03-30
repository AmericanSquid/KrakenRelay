import logging
import threading
import time
from collections import deque

# ----- In-memory log capture for the Web UI ("Recent Events") -----
# Keeps a small ring buffer of recent log lines so the UI can poll /logs.
_UI_LOG_MAX = 300
_ui_log_lock = threading.Lock()
_ui_log_buf = deque(maxlen=_UI_LOG_MAX)
_ui_log_seq = 0
_ui_log_ready = False

class _UILogRingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        global _ui_log_seq
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()

        with _ui_log_lock:
            _ui_log_seq += 1
            _ui_log_buf.append({
                "id": _ui_log_seq,
                "ts": time.time(),
                "time": time.strftime("%H:%M:%S"),
                "level": record.levelname,
                "message": msg,
            })

def init_ui_log_capture() -> None:
    """Attach a ring-buffer log handler to the root logger (once)."""
    global _ui_log_ready
    if _ui_log_ready:
        return

    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, _UILogRingHandler):
            _ui_log_ready = True
            return

    h = _UILogRingHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(h)
    _ui_log_ready = True

# Enable UI log capture on import (safe if called multiple times)
init_ui_log_capture()
