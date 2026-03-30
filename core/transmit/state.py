import numpy as np
from audio.primitives import silence

class TxState:
    def __init__(self, controller, config):
        self.controller = controller
        self.config = config
        chunk_size = self.config.config['audio']['chunk_size']

        self.transmitting = False
        self.tx_start_pending = False

        self.last_audio_time = 0.0
        self.last_transmission = 0.0

        self.skip_courtesy_tone = False

        # Silence Buffer
        self.silence_chunk = silence(chunk_size, np.int16)
        self.silence_bytes = self.silence_chunk.tobytes()

        self.vox_buffer = []
        self.vox_delay_active = False
        self.vox_delay_end_time = 0.0
