"""Stable-ish audio device identity helpers.

KrakenRelay should not persist raw PyAudio indices as the primary device
selection. PortAudio/ALSA device indices can move after reboot, USB replug, or
hardware reorder. These helpers store a small descriptor instead:

    {"id": "usb-audio-codec", "name": "USB Audio CODEC"}

The legacy index is still accepted as a fallback so existing configs keep
working.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional


@dataclass(frozen=True)
class DeviceMatch:
    index: int
    descriptor: dict[str, Any]
    method: str
    warning: Optional[str] = None


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(value: Any) -> str:
    text = str(value or "audio-device").strip().lower()
    text = _SLUG_RE.sub("-", text).strip("-")
    return text or "audio-device"


def _device_name(device: dict[str, Any]) -> str:
    return str(device.get("name") or "Unknown Audio Device").strip()


def _direction_ok(device: dict[str, Any], direction: str) -> bool:
    if direction == "input":
        return int(device.get("maxInputChannels", 0) or 0) > 0
    if direction == "output":
        return int(device.get("maxOutputChannels", 0) or 0) > 0
    raise ValueError(f"Unknown audio device direction: {direction}")


def device_descriptor(device: dict[str, Any], direction: str) -> dict[str, Any]:
    """Return the minimal persisted descriptor for a PyAudio device dict."""
    name = _device_name(device)

    # Keep this intentionally simple: no API, no channel count in config. The id
    # is hidden/internal and generated from the visible device name plus the
    # direction so input/output selections cannot collide in the UI/config path.
    return {
        "id": f"{direction}-{_slug(name)}",
        "name": name,
    }


def selectable_devices(
    devices: Iterable[dict[str, Any]], direction: str
) -> list[dict[str, Any]]:
    """Build compact UI/config device records for one direction."""
    out: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for device in devices:
        if not _direction_ok(device, direction):
            continue

        descriptor = device_descriptor(device, direction)
        base_id = descriptor["id"]
        seen[base_id] = seen.get(base_id, 0) + 1

        # Duplicate names can happen with multiple USB audio dongles. We still
        # keep the saved descriptor small, but make current-session ids unique
        # so the UI selection is not ambiguous while the page is open.
        if seen[base_id] > 1:
            descriptor = {
                "id": f"{base_id}-{seen[base_id]}",
                "name": descriptor["name"],
            }

        out.append(
            {
                "index": int(device.get("index")),
                "id": descriptor["id"],
                "name": descriptor["name"],
            }
        )

    return out


def descriptor_for_index(
    devices: Iterable[dict[str, Any]], index: int, direction: str
) -> Optional[dict[str, Any]]:
    """Return the descriptor for a current runtime index, if present."""
    for device in selectable_devices(devices, direction):
        if int(device["index"]) == int(index):
            return {"id": device["id"], "name": device["name"]}
    return None


def resolve_saved_device(
    saved: Any,
    devices: Iterable[dict[str, Any]],
    direction: str,
    legacy_index: Any = None,
) -> DeviceMatch:
    """Resolve a saved descriptor/id/name to the current PyAudio index.

    Match order:
      1. exact saved id
      2. unique exact saved name
      3. legacy index
    """
    current = selectable_devices(devices, direction)

    saved_id: Optional[str] = None
    saved_name: Optional[str] = None

    if isinstance(saved, dict):
        saved_id = saved.get("id")
        saved_name = saved.get("name")
    elif saved not in (None, ""):
        text = str(saved)
        if text.isdigit():
            legacy_index = text
        else:
            saved_id = text
            saved_name = text

    if saved_id:
        for device in current:
            if device["id"] == saved_id:
                return DeviceMatch(
                    index=int(device["index"]),
                    descriptor={"id": device["id"], "name": device["name"]},
                    method="id",
                )

    if saved_name:
        name_matches = [d for d in current if d["name"] == saved_name]
        if len(name_matches) == 1:
            d = name_matches[0]
            return DeviceMatch(
                index=int(d["index"]),
                descriptor={"id": d["id"], "name": d["name"]},
                method="name",
                warning=f"{direction} device matched by name; saved id was not present.",
            )
        if len(name_matches) > 1:
            raise ValueError(
                f"Multiple {direction} devices are named '{saved_name}'. "
                f"Please reselect the {direction} device in the UI."
            )

    if legacy_index not in (None, ""):
        try:
            idx = int(legacy_index)
        except Exception as exc:
            raise ValueError(
                f"Invalid legacy {direction} device index: {legacy_index}"
            ) from exc

        for d in current:
            if int(d["index"]) == idx:
                return DeviceMatch(
                    index=idx,
                    descriptor={"id": d["id"], "name": d["name"]},
                    method="legacy_index",
                    warning=f"{direction} device matched by legacy index; config should be resaved.",
                )

    raise ValueError(f"No saved {direction} device could be resolved.")
