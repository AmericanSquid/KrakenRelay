class CM108PTT:
    def __init__(self, device="/dev/hidraw0", pin=3):
        self.device = device
        self.pin = pin

    def _set_gpio(self, state):
        if not 1 <= self.pin <= 8:
            raise ValueError("GPIO pin must be 1–8")
        iomask = 1 << (self.pin - 1)
        iodata = state << (self.pin - 1)
        buf = bytes([0x00, 0x00, iomask, iodata, 0x00])
        with open(self.device, "wb", buffering=0) as f:
            f.write(buf)

    def key(self):
        self._set_gpio(1)

    def unkey(self):
        self._set_gpio(0)
