class SignalGateState:
    def __init__(self):
        self._carrier_valid = False
        self._carrier_probe_start = None
        self._carrier_last_level_db = None

        # squelch
        self.squelch_open = False
        self.squelch_open_time = 0.0
        self._last_above_squelch = 0.0

        # kerchunk buffer
        self.kerchunk_buffer = []
        self.kerchunk_backlog_max = 50
