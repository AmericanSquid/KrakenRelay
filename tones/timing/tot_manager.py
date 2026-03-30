import time
import logging
from runtime.logging_utils import debug_enabled

class TOTManager:
    def __init__(self, config, ptt_unkey_fn):
        self.config = config
        self.ptt_unkey_fn = ptt_unkey_fn
        tot_cfg = config.config['tot']
        self.tot_enabled = tot_cfg['tot_enabled']
        self.tot_limit = tot_cfg['tot_time']
        self.lockout_enabled = tot_cfg['tot_lockout_enabled']
        self.lockout_time = tot_cfg.get('tot_lockout_time', 180)
        self.tx_start_time = None
        self.timer_start = None
        self.lockout_start = None
        self._locked = False
        self._last_tot_log = 0.0

    def reset(self):
        self.timer_start = time.time()
        self.tx_start_time = None
    
    def update(self):
        """Returns elapsed time since TX start, or 0."""
        now = time.time()
        if self.timer_start is None:
            return 0
        return now - self.timer_start
    
    def check_timeout(self, is_transmitting):
        if (not self.tot_enabled) or (self.tx_start_time is None) or (not is_transmitting):
            return False

        now = time.time()
        elapsed = now - self.tx_start_time

        if now - self._last_tot_log >= 1.0:
            if debug_enabled:
                logging.debug(f"TOT: elapsed={elapsed:.1f}s / limit={self.tot_limit}s")
            self._last_tot_log = now
        if elapsed >= self.tot_limit:
            logging.warning(f"TOT limit reached at {elapsed:.1f} seconds")

            if self.lockout_enabled:
                self._locked = True
                self.lockout_start = now
                self.tx_start_time = None
                self.ptt_unkey_fn()
                logging.info("TOT lockout activated")

            return True
        return False

    def check_lockout_expired(self):
        now = time.time()

        if not self._locked:
            return False
        if now - self.lockout_start > self.lockout_time:
            self._locked = False
            self.lockout_start = None
            logging.info("🔓 TOT lockout released")
            return True
        return False

    def is_locked(self):
        if self._locked:
            logging.warning("Attempted to start transmission during TOT lockout. Blocking.")
        return self._locked

