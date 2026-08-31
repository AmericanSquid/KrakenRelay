"""Small mutable state shared by repeater services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RepeaterState:
    """Flags and timestamps shared by the audio, lifecycle, and TX services."""

    running: bool = False
    current_rms: float = 0.0
    transmission_start_time: float | None = None
    cw_gen: Any = None
    cw_next_t: float | None = None
    last_clip_time: float = 0.0
    last_limit_time: float = 0.0
