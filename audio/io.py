import numpy as np
import logging

from .primitives import pcm_to_int16_bytes

class AudioIO:
    def __init__(self, controller, input_device, output_device):
        self.controller = controller
        self.input_device = input_device
        self.output_device = output_device

    def send_pcm(self, pcm: np.ndarray):
        controller = self.controller
        output_stream = controller.streams.output_stream
        output_stream_2 = controller.streams.output_stream_2

        data = pcm_to_int16_bytes(pcm)

        # Primary output
        try:
            if output_stream:
               output_stream.write(data)
        except Exception as e:
            logging.warning(f"[Output] Primary output write failed: {e}")

        # Secondary output (dual output) – best effort
        if output_stream_2:
            try:
                output_stream_2.write(data)
            except Exception as e:
                logging.warning(f"[Dual Output] Secondary output failed, disabling: {e}")
                try:
                    output_stream_2.stop_stream()
                    output_stream_2.close()
                except Exception:
                    pass
                output_stream_2 = None
