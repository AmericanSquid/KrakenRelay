import logging
import time


class ScheduleID:
    def __init__(
        self,
        config,
        start_cw_id,
        is_transmitting,
        is_cw_active,
    ):
        self.config = config
        self.start_cw_id = start_cw_id
        self.is_transmitting = is_transmitting
        self.is_cw_active = is_cw_active
        self.last_id_time = time.time()
        self.post_tx = False
        self.sending_id = False
        self.last_stop_time = time.time()
        self.cooldown = 0.25

    def mark_post_tx(self):
        now = time.time()
        self.post_tx = True
        self.last_stop_time = now

    def check_and_send(self):
        cfg = self.config.config
        id_cfg = cfg["identification"]

        now = time.time()

        if self.is_cw_active():
            return

        if id_cfg["cw_enabled"]:
            interval = float(id_cfg.get("interval_minutes", 10)) * 60.0
            should_id = now - self.last_id_time > interval

            if should_id and not self.sending_id:
                if self.is_transmitting() or now - self.last_stop_time > self.cooldown:
                    if self.post_tx and not self.is_transmitting():
                        logging.info("Sending CW ID after user transmission.")
                    elif self.is_transmitting():
                        logging.info("Sending CW ID under user audio.")
                    else:
                        logging.info("Sending CW ID while idle.")

                    self.post_tx = False
                    self.send_id()

    def send_id(self):
        cfg = self.config.config
        id_cfg = cfg["identification"]

        now = time.time()

        if not id_cfg.get("cw_enabled", False):
            logging.error("Manual ID requested but CW is disabled.")
            return

        if self.is_cw_active() or self.sending_id:
            logging.warning("Unable to Send ID: CW ID already in progress")
            return

        self.sending_id = True
        try:
            callsign = id_cfg["callsign"]
            if id_cfg["cw_enabled"]:
                self.start_cw_id(callsign)
                logging.info(f"Sent CW ID: {callsign}")
        except Exception as e:
            logging.error(f"ID failed: {e}")
        finally:
            self.last_id_time = now
            self.sending_id = False
