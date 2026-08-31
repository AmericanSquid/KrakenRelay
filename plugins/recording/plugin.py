import logging
import queue
import threading
import time
from pathlib import Path

import numpy as np
from flask import send_file

from .encoder import MP3Encoder
from .naming import generate_recording_path


class RecordingPlugin:
    name = "recording"

    def __init__(self, config, audio_config):
        self.config = config or {}

        audio_cfg = audio_config or {}
        self.sample_rate = int(audio_cfg.get("sample_rate", 48000))
        self.chunk_size = int(audio_cfg.get("chunk_size", 1024))

        self.save_path = self.config.get(
            "save_path", "~/Documents/KrakenRelayRecordings"
        )
        self.bitrate = self.config.get("bitrate", "64k")
        self.filename_prefix = self.config.get("filename_prefix", "krakenrelay")

        self._lock = threading.RLock()
        self._queue = None
        self._writer_thread = None
        self._stop_event = threading.Event()
        self._encoder = None

        self._recording = False
        self._current_file = None
        self._started_at = None
        self._audio_seen_this_tick = False
        self._silence_chunk = np.zeros(self.chunk_size, dtype=np.int16)

    def start(self):
        logging.info("[Recording] Start requested")

        with self._lock:
            if self._recording:
                return {
                    "ok": True,
                    "already_recording": True,
                    "file": str(self._current_file) if self._current_file else None,
                }

            try:
                output_path = generate_recording_path(
                    self.save_path, self.filename_prefix
                )
            except Exception as exc:
                logging.exception("[Recording] Failed to prepare recording path")
                return {"ok": False, "error": str(exc)}

            self._queue = queue.Queue(maxsize=200)
            self._stop_event.clear()
            self._encoder = MP3Encoder(
                sample_rate=self.sample_rate, bitrate=self.bitrate
            )

            try:
                self._encoder.start(output_path)
            except Exception as exc:
                logging.exception("[Recording] Failed to start encoder")
                self._encoder = None
                self._queue = None
                return {"ok": False, "error": str(exc)}

            self._recording = True
            self._current_file = output_path
            self._started_at = time.time()
            self._audio_seen_this_tick = False

            self._writer_thread = threading.Thread(
                target=self._writer_loop,
                name="RecordingPluginWriter",
                daemon=True,
            )
            self._writer_thread.start()

            logging.info("[Recording] Started recording to %s", output_path)
            return {"ok": True, "file": str(output_path)}

    def stop(self):
        logging.info("[Recording] Stop requested")

        with self._lock:
            if not self._recording:
                return {"ok": True, "already_stopped": True}

            current_file = self._current_file
            self._recording = False
            self._stop_event.set()

            q = self._queue
            if q is not None:
                try:
                    q.put_nowait(None)
                except Exception:
                    pass

            writer_thread = self._writer_thread

        if writer_thread and writer_thread is not threading.current_thread():
            writer_thread.join(timeout=5)

        with self._lock:
            self._queue = None
            self._writer_thread = None
            self._encoder = None
            self._current_file = None
            self._started_at = None
            self._audio_seen_this_tick = False

        logging.info("[Recording] Saved recording: %s", current_file)
        return {"ok": True, "file": str(current_file) if current_file else None}

    def status(self):
        with self._lock:
            started_at = self._started_at
            elapsed = time.time() - started_at if started_at else 0.0
            return {
                "ok": True,
                "recording": bool(self._recording),
                "file": str(self._current_file) if self._current_file else None,
                "started_at": started_at,
                "elapsed_seconds": round(elapsed, 1),
                "save_path": self.save_path,
                "bitrate": self.bitrate,
                "filename_prefix": self.filename_prefix,
            }

    def on_audio_frame(self, samples):
        with self._lock:
            if not self._recording:
                return
            self._audio_seen_this_tick = True

        chunk = self._normalize_samples(samples)
        self._enqueue(chunk)

    def on_tick(self):
        with self._lock:
            if not self._recording:
                return
            saw_audio = self._audio_seen_this_tick
            self._audio_seen_this_tick = False

        if not saw_audio:
            self._enqueue(self._silence_chunk.copy())

    def on_shutdown(self):
        self.stop()

    def api_start(self, flask_request):
        if flask_request.method != "POST":
            return {"ok": False, "error": "Use POST to start recording."}, 405
        return self.start()

    def api_stop(self, flask_request):
        if flask_request.method != "POST":
            return {"ok": False, "error": "Use POST to stop recording."}, 405
        return self.stop()

    def api_status(self, flask_request):
        return self.status()

    def _recordings_dir(self):
        return Path(self.save_path).expanduser().resolve()

    def _recording_path_from_request(self, flask_request):
        filename = flask_request.args.get("file", "").strip()

        if not filename:
            return None, {"ok": False, "error": "Missing file parameter."}, 400

        recordings_dir = self._recordings_dir()
        requested = (recordings_dir / filename).resolve()

        if recordings_dir not in requested.parents and requested != recordings_dir:
            return None, {"ok": False, "error": "Invalid recording path."}, 403

        if not requested.exists() or not requested.is_file():
            return None, {"ok": False, "error": "Recording not found."}, 404

        return requested, None, None

    def api_download(self, flask_request):
        if flask_request.method != "GET":
            return {"ok": False, "error": "Use GET to download recording."}, 405

        path, error, status = self._recording_path_from_request(flask_request)
        if error is not None:
            return error, status

        return send_file(
            path,
            as_attachment=True,
            download_name=path.name,
            mimetype="audio/mpeg",
        )

    def _normalize_samples(self, samples):
        if isinstance(samples, bytes):
            return np.frombuffer(samples, dtype=np.int16).copy()
        if isinstance(samples, np.ndarray) and samples.dtype == np.int16:
            return samples.copy()
        return np.clip(np.asarray(samples), -32768, 32767).astype(np.int16)

    def _enqueue(self, chunk):
        with self._lock:
            q = self._queue
            active = self._recording

        if not active or q is None:
            return

        try:
            q.put_nowait(chunk)
        except queue.Full:
            logging.warning("[Recording] Queue full; dropped recording chunk")
        except Exception:
            logging.exception("[Recording] Failed to queue recording chunk")

    def _writer_loop(self):
        logging.info("[Recording] Writer thread started")
        encoder = self._encoder
        q = self._queue

        try:
            while not self._stop_event.is_set():
                try:
                    item = q.get(timeout=0.25)
                except queue.Empty:
                    continue
                if item is None:
                    break
                if encoder is not None:
                    encoder.write_pcm(item)

            while q is not None:
                try:
                    item = q.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    continue
                if encoder is not None:
                    encoder.write_pcm(item)
        except Exception:
            logging.exception("[Recording] Writer thread crashed")
        finally:
            if encoder is not None:
                encoder.stop()
            logging.info("[Recording] Writer thread exited")


def load_plugin(config, audio_config):
    return RecordingPlugin(config=config, audio_config=audio_config)
