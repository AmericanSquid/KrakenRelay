import time
import logging

class TOTManager:
    def __init__(self, config, ptt_unkey_fn):
        self.config = config

        self.ptt_unkey_fn = ptt_unkey_fn

        self.tot_enabled = config.config['tot']['tot_enabled']
        self.tot_limit = config.config['tot']['tot_time']
        self.lockout_enabled = config.config['tot']['tot_lockout_enabled']
        self.lockout_time = config.config['tot'].get('tot_lockout_time', 180)
        self.tx_start_time = None
        self.timer_start = None
        self.lockout_start = None
        self._locked = False

    def reset(self):
        self.timer_start = time.time()
        self.tx_start_time = None
    
    def update(self):
        """Returns elapsed time since TX start, or 0."""
        if self.timer_start is None:
            return 0
        return time.time() - self.timer_start
    
    def exceeded(self):
        return self.tot_enabled and self.update() >= self.tot_limit
    

    def check_timeout(self, is_transmitting):
        if (not self.tot_enabled) or (self.tx_start_time is None) or (not is_transmitting):
            return False

        elapsed = time.time() - self.tx_start_time
        logging.debug(f"TOT: elapsed={elapsed:.1f}s / limit={self.tot_limit}s")
        if elapsed >= self.tot_limit:
            logging.warning(f"TOT limit reached at {elapsed:.1f} seconds")

            if self.lockout_enabled:
                self._locked = True
                self.lockout_start = time.time()
                self.tx_start_time = None
                self.ptt_unkey_fn()
                logging.info("TOT lockout activated")

            return True
        return False

    def check_lockout_expired(self):
        if not self._locked:
            return False
        if time.time() - self.lockout_start > self.lockout_time:
            self._locked = False
            self.lockout_start = None
            logging.info("🔓 TOT lockout released")
            return True
        return False

    def is_locked(self):
        if self._locked:
            logging.warning("Attempted to start transmission during TOT lockout. Blocking.")
        return self._locked

