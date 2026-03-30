import numpy as np
from audio import check_clipping

class TxAudio:
    def __init__(self, controller, config):
        self.controller = controller
        self.config = config
        
    def send_chunk(self, samples: np.ndarray) -> None:
        controller = self.controller
        cfg = self.config.config         
        audio_cfg = cfg["audio"]

        if (
            audio_cfg.get("limiter_enabled", False)
            or audio_cfg.get("compressor_enabled", False)
            or audio_cfg.get("highpass_enabled", False)
            or audio_cfg.get("notch_enabled", False)
        ):
            samples = controller.dsp_tx.process_int16_to_int16(samples)

        controller.meter.update(samples, "tx")
        check_clipping(samples)
        controller.audio_io.send_pcm(samples)
