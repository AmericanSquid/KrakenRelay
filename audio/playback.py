import numpy as np

class AudioPlayback:
    def __init__(self, controller, config):
        self.controller = controller
        self.config = config

    def play_chunks(self, audio):
        controller = self.controller
        audio_cfg = self.config.config['audio']
        chunk_size = audio_cfg['chunk_size']
       
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            controller.tx_audio.send_chunk(chunk)
