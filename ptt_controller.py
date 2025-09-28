import logging

class CM108PTT:
    def __init__(self, device="/dev/hidraw0", pin=3):
        self.device = device
        self.pin = pin
        self.working = True

    def _set_gpio(self, state):
        if not 1 <= self.pin <= 8:
            logging.error(f"CM108PTT: GPIO pin must be 1–8, got {self.pin}")
            self.working = False
            return
        iomask = 1 << (self.pin - 1)
        iodata = state << (self.pin - 1)
        buf = bytes([0x00, 0x00, iomask, iodata, 0x00])
        try:
            with open(self.device, "wb", buffering=0) as f:
                f.write(buf)
            self.working = True
        except Exception as e:
            logging.error(f"CM108PTT: FAILED to write to {self.device}: {e}")
            self.working = False 

    def key(self):
        try:
            self._set_gpio(1)
        except Exception as e:
            logging.error(f"CM108PTT: key() error: {e}")


    def unkey(self):
        try:
            self._set_gpio(0)
        except Exception as e:
            logging.error(f"CM108PTT: unkey() error: {e}")
