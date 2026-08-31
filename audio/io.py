import logging

import numpy as np

from runtime.audit import AuditEvent

from .primitives import pcm_to_int16_bytes


class AudioIO:
    def __init__(self, config, streams, plugins, audit, input_device, output_device):
        self.config = config
        self.streams = streams
        self.plugins = plugins
        self.audit = audit
        self.input_device = input_device
        self.output_device = output_device

    def send_pcm(self, pcm: np.ndarray, link_pcm: np.ndarray | bytes | None = None):
        output_stream = self.streams.output_stream
        output_stream_2 = self.streams.output_stream_2
        audio_cfg = self.config.config.get("audio", {})
        output_2_mode = str(
            audio_cfg.get("output_2_mode", "simulcast") or "simulcast"
        ).lower()

        data = pcm_to_int16_bytes(pcm)

        if self.plugins is not None:
            self.plugins.emit_audio_frame(pcm)

        # Primary output always receives the main TX program audio.
        try:
            if output_stream:
                output_stream.write(data)

        except Exception as e:
            error_text = repr(e)

            if self.audit:
                event_type = AuditEvent.AUDIO_STREAM_WRITE_FAILURE

                if "underflow" in error_text.lower():
                    event_type = AuditEvent.PYAUDIO_UNDERFLOW

                self.audit.error(
                    event_type=event_type,
                    source="audio_io",
                    message="Primary output audio stream write failed",
                    metadata={
                        "output": "primary",
                        "output_device": self.output_device,
                        "error": error_text,
                        "bytes": len(data),
                    },
                )

            logging.warning(
                "[Output] Primary output write failed: %s",
                e,
            )

        # Secondary output (dual output) - best effort.
        #   simulcast: output 2 gets the same main TX program audio.
        #   link:      output 2 only gets link_pcm when the live RX path supplies
        #              local/Input-1-only audio. Generated tones/CW/tail silence
        #              call send_pcm without link_pcm, so they do not go to the node.
        if output_stream_2:
            if output_2_mode == "link":
                if link_pcm is None:
                    return
                data_2 = pcm_to_int16_bytes(link_pcm)
            else:
                data_2 = data

            try:
                output_stream_2.write(data_2)

            except Exception as e:
                error_text = repr(e)

                if self.audit:
                    event_type = AuditEvent.AUDIO_STREAM_WRITE_FAILURE

                    if "underflow" in error_text.lower():
                        event_type = AuditEvent.PYAUDIO_UNDERFLOW

                    self.audit.error(
                        event_type=event_type,
                        source="audio_io",
                        message="Secondary output audio stream write failed; disabling secondary output",
                        metadata={
                            "output": "secondary",
                            "output_2_mode": output_2_mode,
                            "error": error_text,
                            "bytes": len(data_2),
                        },
                    )

                logging.warning(
                    "[Dual Output] Secondary output failed, disabling: %s",
                    e,
                )

                try:
                    output_stream_2.stop_stream()
                    output_stream_2.close()

                except Exception:
                    pass

                self.streams.output_stream_2 = None
