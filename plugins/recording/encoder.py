import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np


class MP3Encoder:
    """
    Thin ffmpeg wrapper.

    Input: mono signed 16-bit little-endian PCM.
    Output: MP3.
    """

    def __init__(self, sample_rate: int, bitrate: str = "64k"):
        self.sample_rate = int(sample_rate)
        self.bitrate = str(bitrate or "64k")
        self.process = None
        self.output_path = None

    def start(self, output_path: Path):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not installed or not available on PATH")

        self.output_path = Path(output_path)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-vn",
            "-codec:a",
            "libmp3lame",
            "-b:a",
            self.bitrate,
            str(self.output_path),
        ]

        logging.info("[Recording] Launching ffmpeg MP3 encoder: %s", self.output_path)
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write_pcm(self, samples):
        if self.process is None or self.process.stdin is None:
            return

        if self.process.poll() is not None:
            logging.error(
                "[Recording] ffmpeg exited unexpectedly with code %s",
                self.process.returncode,
            )
            return

        if isinstance(samples, bytes):
            data = samples
        elif isinstance(samples, np.ndarray) and samples.dtype == np.int16:
            data = samples.tobytes()
        else:
            data = (
                np.clip(np.asarray(samples), -32768, 32767).astype(np.int16).tobytes()
            )

        try:
            self.process.stdin.write(data)
        except BrokenPipeError:
            logging.error("[Recording] ffmpeg pipe broke while writing audio")
        except Exception:
            logging.exception("[Recording] Failed writing PCM to ffmpeg")

    def stop(self):
        if self.process is None:
            return

        proc = self.process
        self.process = None

        logging.info("[Recording] Finalizing MP3 file")

        try:
            if proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logging.warning("[Recording] ffmpeg did not exit cleanly; terminating")
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    logging.warning("[Recording] ffmpeg still did not exit; killing")
                    proc.kill()
                    proc.wait(timeout=2)

            if proc.returncode not in (0, None):
                logging.error("[Recording] ffmpeg exited with code %s", proc.returncode)
        except Exception:
            logging.exception("[Recording] Error while stopping MP3 encoder")
