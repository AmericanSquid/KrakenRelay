import logging
import time
from ..tone_vol import _tone_vol
from ..engine import CWGenerator

class ScheduleID:
    def __init__(self, controller, config, tx_start_fn, tx_stop_fn, tx_state_fn, set_skip_courtesy_fn):
        self.config = config
        self.morse = CWGenerator(
            wpm=config.config['identification']['cw_wpm'],
            frequency=config.config['identification']['cw_pitch'],
            sample_rate=config.config['audio']['sample_rate'],
            volume=_tone_vol(config.config['identification'].get('cw_volume', 100))
            )
        self.tx_start = tx_start_fn
        self.tx_stop = tx_stop_fn
        self.is_transmitting = tx_state_fn
        self.set_skip_courtesy = set_skip_courtesy_fn
        self.last_id_time = time.time()
        self.post_tx = False
        self.sending_id = False
        self.last_stop_time = time.time()
        self.cooldown = 0.25
        self.controller = controller

    def mark_post_tx(self):
        now = time.time()
        self.post_tx = True
        self.last_stop_time = now
    
    def check_and_send(self):
        cfg = self.config.config
        id_cfg = cfg['identification']

        now = time.time()

        # === Idle & Post-TX CW ID === #
        if self.is_transmitting():
            return

        if id_cfg['cw_enabled']:
            interval = id_cfg['interval_minutes'] * 60
            should_id = now - self.last_id_time > interval

            if should_id and (self.post_tx or not self.sending_id):
                if now - self.last_stop_time > self.cooldown:
                    if self.post_tx:
                        logging.info("Sending CW ID after user transmission.")
                        self.post_tx = False
                    else:
                        logging.info("Sending CW ID while idle.")

                    self.send_id()

    def send_id(self):
        controller = self.controller
        cfg = self.config.config
        id_cfg = cfg['identification']

        now = time.time()

        if not id_cfg.get('cw_enabled', False):
            logging.error("Manual ID requested but CW is disabled.")
            return

        if self.is_transmitting() or self.sending_id:
            logging.warning("Unable to Send ID: already transmitting")
            return

        self.sending_id = True
        try:
            callsign = id_cfg['callsign']
            if id_cfg['cw_enabled']:
                controller.request_cw.start_cw_id(callsign)
                logging.info(f"Sent CW ID: {callsign}")
        except Exception as e:
            logging.error(f"ID failed: {e}")
        finally:
            self.last_id_time = now
            self.sending_id = False
