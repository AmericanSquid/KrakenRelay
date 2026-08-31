import logging

from audio.device_identity import resolve_saved_device
from runtime.audit import AuditEvent
from runtime.logging_utils import debug_enabled


class Streams:
    def __init__(self, audio_manager, config, audit, input_device, output_device):
        self.audio_manager = audio_manager
        self.config = config
        self.audit = audit
        self.input_device = input_device
        self.output_device = output_device

        self.input_stream = None
        self.input_stream_2 = None
        self.output_stream = None
        self.output_stream_2 = None

    def setup(self):
        audio_manager = self.audio_manager

        try:
            audio_cfg = self.config.config["audio"]
            sample_rate = audio_cfg["sample_rate"]
            chunk_size = audio_cfg["chunk_size"]

            logging.info(f"Setting up audio streams with rate: {sample_rate}")

            # Primary input stream
            try:
                self.input_stream = audio_manager.create_input_stream(
                    device_index=self.input_device,
                    rate=sample_rate,
                    chunk=chunk_size,
                )
            except Exception as e:
                if self.audit:
                    self.audit.error(
                        event_type=AuditEvent.AUDIO_DEVICE_LOST,
                        source="audio_streams",
                        message="Failed to open primary input audio device",
                        metadata={
                            "direction": "input",
                            "device_index": self.input_device,
                            "sample_rate": sample_rate,
                            "chunk_size": chunk_size,
                            "error": repr(e),
                        },
                    )
                raise

            # Optional second input stream.
            # This is intentionally simple: when enabled, input 1 and input 2 are
            # read every audio loop. The engine decides what to mix/send.
            if audio_cfg.get("dual_input", False):
                saved_in2 = audio_cfg.get("input_device_2_info")
                legacy_in2 = audio_cfg.get(
                    "input_device_2", audio_cfg.get("input_index_2")
                )
                devices = audio_manager.list_devices()

                try:
                    match_in2 = resolve_saved_device(
                        saved=saved_in2,
                        devices=devices,
                        direction="input",
                        legacy_index=legacy_in2,
                    )

                    second_input_index = match_in2.index

                    # Keep config healed in memory if fallback/name match was used.
                    audio_cfg["input_device_2"] = int(second_input_index)
                    audio_cfg["input_index_2"] = int(second_input_index)
                    audio_cfg["input_device_2_info"] = match_in2.descriptor
                    audio_cfg["input_device_2_id"] = match_in2.descriptor.get("id", "")

                    if int(second_input_index) == int(self.input_device):
                        logging.info(
                            "Dual input resolved to the same device as primary; skipping second stream."
                        )
                    else:
                        self.input_stream_2 = audio_manager.create_input_stream(
                            device_index=second_input_index,
                            rate=sample_rate,
                            chunk=chunk_size,
                        )

                        logging.info(
                            "Dual input enabled on device index %s (%s match)",
                            second_input_index,
                            match_in2.method,
                        )

                        if match_in2.warning:
                            logging.warning("Dual input: %s", match_in2.warning)

                except Exception as e:
                    logging.warning(
                        "Dual input requested but input_device_2 could not be resolved/opened; "
                        "continuing single-input. Error: %s",
                        e,
                    )
                    if self.audit:
                        self.audit.warning(
                            event_type=AuditEvent.AUDIO_DEVICE_LOST,
                            source="audio_streams",
                            message="Failed to open secondary input audio device; continuing with single input",
                            metadata={
                                "direction": "input_2",
                                "saved_device": saved_in2,
                                "legacy_index": legacy_in2,
                                "sample_rate": sample_rate,
                                "chunk_size": chunk_size,
                                "error": repr(e),
                            },
                        )
                    self.input_stream_2 = None
            else:
                if debug_enabled():
                    logging.debug("Dual input disabled in config")

            # Primary output stream
            try:
                self.output_stream = audio_manager.create_output_stream(
                    device_index=self.output_device,
                    rate=sample_rate,
                    chunk=chunk_size,
                )
            except Exception as e:
                if self.audit:
                    self.audit.error(
                        event_type=AuditEvent.AUDIO_DEVICE_LOST,
                        source="audio_streams",
                        message="Failed to open primary output audio device",
                        metadata={
                            "direction": "output",
                            "device_index": self.output_device,
                            "sample_rate": sample_rate,
                            "chunk_size": chunk_size,
                            "error": repr(e),
                        },
                    )
                raise

            # Optional second output stream
            if audio_cfg.get("dual_output", False):
                saved_dev2 = audio_cfg.get("output_device_2_info")
                legacy_dev2 = audio_cfg.get("output_device_2")
                devices = audio_manager.list_devices()

                try:
                    match2 = resolve_saved_device(
                        saved=saved_dev2,
                        devices=devices,
                        direction="output",
                        legacy_index=legacy_dev2,
                    )

                    second_index = match2.index

                    # Keep config healed in memory if fallback/name match was used.
                    audio_cfg["output_device_2"] = int(second_index)
                    audio_cfg["output_device_2_info"] = match2.descriptor
                    audio_cfg["output_device_2_id"] = match2.descriptor.get("id", "")

                    if int(second_index) == int(self.output_device):
                        logging.info(
                            "Dual output resolved to the same device as primary; skipping second stream."
                        )
                    else:
                        self.output_stream_2 = audio_manager.create_output_stream(
                            device_index=second_index,
                            rate=sample_rate,
                            chunk=chunk_size,
                        )

                        logging.info(
                            "Dual output enabled on device index %s (%s match, mode=%s)",
                            second_index,
                            match2.method,
                            audio_cfg.get("output_2_mode", "simulcast"),
                        )

                        if match2.warning:
                            logging.warning("Dual output: %s", match2.warning)

                except Exception as e:
                    logging.warning(
                        "Dual output requested but output_device_2 could not be resolved/opened; "
                        "continuing single-output. Error: %s",
                        e,
                    )
                    if self.audit:
                        self.audit.warning(
                            event_type=AuditEvent.AUDIO_DEVICE_LOST,
                            source="audio_streams",
                            message="Failed to open secondary output audio device; continuing with single output",
                            metadata={
                                "direction": "output_2",
                                "saved_device": saved_dev2,
                                "legacy_index": legacy_dev2,
                                "sample_rate": sample_rate,
                                "chunk_size": chunk_size,
                                "error": repr(e),
                            },
                        )
                    self.output_stream_2 = None
            else:
                if debug_enabled():
                    logging.debug("Dual output disabled in config")

            if self.audit:
                self.audit.info(
                    event_type=AuditEvent.AUDIO_DEVICE_RECOVERED,
                    source="audio_streams",
                    message="Audio streams opened successfully",
                    metadata={
                        "input_device": self.input_device,
                        "input_device_2_active": self.input_stream_2 is not None,
                        "output_device": self.output_device,
                        "dual_input": bool(audio_cfg.get("dual_input", False)),
                        "dual_output": bool(audio_cfg.get("dual_output", False)),
                        "output_2_mode": audio_cfg.get("output_2_mode", "simulcast"),
                        "output_device_2_active": self.output_stream_2 is not None,
                        "sample_rate": sample_rate,
                        "chunk_size": chunk_size,
                    },
                )
            logging.info(
                f"Audio streams setup complete: rate={sample_rate}, chunk={chunk_size}"
            )

        except Exception as e:
            logging.error(f"Failed to setup audio streams: {e}")
            raise
