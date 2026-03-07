import threading
import time
import logging

class BufferWatchdog:
    def __init__(
        self,
        get_inq,
        get_outq,
        get_kerchunk,
        get_backlog,
        get_squelch,
        get_transmitting,
        log_path="/tmp/krakenrelay_buffer.log",
        interval=1.0
    ):
        self.get_inq = get_inq
        self.get_outq = get_outq
        self.get_kerchunk = get_kerchunk
        self.get_backlog = get_backlog
        self.get_squelch = get_squelch
        self.get_transmitting = get_transmitting
        self.interval = interval
        self.stop_event = threading.Event()

        # Set up a dedicated file logger
        self.logger = logging.getLogger("buffer_logger")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler(log_path)
            handler.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
            self.logger.addHandler(handler)

    def start(self):
        self.thread = threading.Thread(target=self._run, name="BufferWatchdog", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=1.0)

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.logger.debug(
                    "inq=%s outq=%s ker=%s backlog=%s sq=%s tx=%s",
                    self.get_inq(),
                    self.get_outq(),
                    self.get_kerchunk(),
                    self.get_backlog(),
                    self.get_squelch(),
                    self.get_transmitting(),
                )
            except Exception as e:
                self.logger.error("BufferWatchdog error: %s", e)
            time.sleep(self.interval)
