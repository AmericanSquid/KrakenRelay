"""
mumble_interface.py  –  Opus/Mumble transport for KrakenRelay
──────────────────────────────────────────────────────────────
• LINK mode: fully implemented (bidirectional or TX-/RX-only).
• VOTER mode: public API + stubs in place for future work.
"""

from __future__ import annotations
import queue
import threading
import time
import logging
from typing import Dict, List
import numpy as np
from pymumble_py3 import Mumble          # ← if you switch to pymumble_py3, change import

_PCM      = np.int16
_FRAME_SZ = 960                      # 20 ms at 48 kHz
_SILENCE = np.zeros(_FRAME_SZ, dtype=_PCM)

# ──────────────────────────────────────────────────────────────
#  Link wrapper
# ──────────────────────────────────────────────────────────────
class MumbleLink:
    """Single-channel Mumble client that behaves like an audio device."""

    def __init__(self, cfg: Dict):
        """
        cfg keys (see config.yaml):
            enabled, mode, direction, host, port, user, password, channel
        """
        self.cfg  = cfg
        self.mode = cfg["mode"]          # "link" | "voter"

        self._running = True

        # Queues for LINK mode
        self._rx_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=50)
        self._tx_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=50)

        # Start client
        self._m = Mumble(
            cfg["host"],
            cfg["user"],
            port     = cfg["port"],
            password = cfg["password"],
            reconnect= True
        )
        self._m.set_receive_sound(True)
        self._m.callbacks.set_callback("sound_received", self._on_sound)
        self._m.start()
        self._m.is_ready()

        # Join target channel
        chan = self._m.channels.find_by_name(cfg["channel"])
        if chan is None:                          # channel not found
            logging.warning("Mumble: channel %s not found, staying in root", cfg["channel"])
        else:
            chan.move_in()

        # Background TX thread
        self._tx_thread = threading.Thread(
            target = self._tx_worker,
            name   = "MumbleTX",
            daemon = True
        )

        self._ping_thread = threading.Thread(
            target = self._keepalive_worker,
            name   = "MumblePing",
            daemon = True
        )

        self._tx_thread.start()
        self._ping_thread.start()

        # Place-holders for voter
        self._sources: Dict[int, _Source] = {}           # session-id ➜ _Source

    # ─── Public API (Link mode) ───────────────────────────────
    def read_frame(self) -> np.ndarray | None:
        """Return next 20 ms PCM (mono 48 k), or silence if none pending."""
        try:
            return self._rx_q.get_nowait()
        except queue.Empty:
            return None

    def write_frame(self, pcm: np.ndarray) -> None:
        """Queue a 20 ms PCM chunk for transmit to the channel."""
        try:
            self._tx_q.put_nowait(pcm.copy())
        except queue.Full:
            pass      # drop on overload

    def disconnect(self) -> None:
        self._running = False
        self._m.stop()

        if hasattr(self, "_tx_thread") and self._tx_thread.is_alive():
            logging.debug("Joining MumbleTX thread...")
            self._tx_thread.join(timeout=1.0)

        if hasattr(self, "_ping_thread") and self._ping_thread.is_alive():
            logging.debug("Joining MumblePing thread...")
            self._ping_thread.join(timeout=1.0)

    # ─── Internal callbacks / workers ─────────────────────────
    def _on_sound(self, user, chunk) -> None:
        print(f"Mumble: received audio, mode={self.mode}")  # DEBUG
        if self.mode == "link":
            pcm = np.frombuffer(chunk.pcm, dtype=_PCM)
            print("PCM size:", pcm.size)  # DEBUG
            if pcm.size == _FRAME_SZ:
                try:
                    self._rx_q.put_nowait(pcm)
                    print("Queued RX audio")  # DEBUG
                except queue.Full:
                    print("RX queue - dropping frame")  # DEBUG

        elif self.mode == "voter":
            # TODO: implement per-user demux + jitter buffer
            # 1) src = self._sources.get(user.session) or create _Source(...)
            # 2) src.enqueue(chunk.pcm)
            pass

    def _tx_worker(self) -> None:
        while self._running:
            try:
                pcm = self._tx_q.get(timeout=0.1)
                try:
                    self._m.sound_output.add_sound(pcm.tobytes())
                except Exception as e:
                    logging.warning(f"[MumbleTX] add_sound failed: {e}")
                    break
            except queue.Empty:
                continue
                
    def _keepalive_worker(self) -> None:
        """Send a UDP Ping every 5s so Murmur never reaps us."""
        while self._running:
            try:
                self._m.ping()
            except Exception as e:
                logging.warning(f"Mumble ping failed: {e}")
            time.sleep(5)


    # ─── Voter scaffolding (stub) ─────────────────────────────
    def voter_sources(self) -> List["_Source"]:
        """Return iterable of _Source objects (empty until voter logic added)."""
        return list(self._sources.values())


# ──────────────────────────────────────────────────────────────
#  Stub class for future VoterEngine integration
# ──────────────────────────────────────────────────────────────
class _Source:
    """Per-remote-site audio queue; ready for future quality scoring."""

    def __init__(self, session_id: int, username: str):
        self.session_id = session_id
        self.username   = username
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=50)
        self._last_pk   = 0.0            # last packet timestamp

    def enqueue(self, pcm_bytes: bytes) -> None:
        try:
            pcm = np.frombuffer(pcm_bytes, dtype=_PCM)
            if pcm.size == _FRAME_SZ:
                self._q.put_nowait(pcm)
                self._last_pk = time.time()
        except queue.Full:
            pass

    def read_frame(self) -> np.ndarray | None:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def id(self) -> str:
        return f"{self.username}({self.session_id})"
