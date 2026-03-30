import logging
from .resolve_devices import resolve_device
from runtime.logging_utils import debug_enabled

class Streams:
    def __init__(self, controller, config):
        self.controller = controller
        self.config = config

        self.input_stream = None
        self.output_stream = None
        self.output_stream_2 = None

    def setup(self):
        controller = self.controller
        audio_manager = controller.audio_manager
 
        try:
            audio_cfg = self.config.config['audio']
            sample_rate = audio_cfg['sample_rate']
            chunk_size = audio_cfg['chunk_size']

            logging.info(f"Setting up audio streams with rate: {sample_rate}")

            if audio_cfg.get('dual_output', False):
                dev2_arg = audio_cfg.get('output_device_2', None)
                
                second_index = None
                
                second_index = resolve_device(dev2_arg, audio_manager, "output")

                if second_index is None:
                    logging.warning("Dual output requested but no valid output_device_2 resolved; continuing single-output.")
                elif int(second_index) == int(controller.audio_io.output_device):
                    logging.info("Dual output resolved to the same device as primary; skipping second stream.")
                else:
                    try:
                        self.output_stream_2 = audio_manager.create_output_stream(
                            device_index=second_index,
                            rate=sample_rate,
                            chunk=chunk_size
                        )
                        logging.info(f"Dual output enabled on device index {second_index}")
                    except Exception as e:
                        logging.warning(f"Dual output: failed to open second stream (idx={second_index}): {e}")
                        self.output_stream_2 = None
            else:
                if debug_enabled():
                    logging.debug("Dual output disabled in config")

            # Primary streams
            self.input_stream = audio_manager.create_input_stream(
                device_index=controller.audio_io.input_device,
                rate=sample_rate,
                chunk=chunk_size
            )

            self.output_stream = audio_manager.create_output_stream(
                device_index=controller.audio_io.output_device,
                rate=sample_rate,
                chunk=chunk_size
            )

            logging.info(f"Audio streams setup complete: rate={sample_rate}, chunk={chunk_size}")

        except Exception as e:
            logging.error(f"Failed to setup audio streams: {e}")
            raise
