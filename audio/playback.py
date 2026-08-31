import numpy as np


class AudioPlayback:
    def __init__(self, config, send_chunk):
        self.config = config
        self.send_chunk = send_chunk

    def play_chunks(self, audio):
        audio_cfg = self.config.config["audio"]
        chunk_size = audio_cfg["chunk_size"]

        for i in range(0, len(audio), chunk_size):
            chunk = audio[i : i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            self.send_chunk(chunk)
