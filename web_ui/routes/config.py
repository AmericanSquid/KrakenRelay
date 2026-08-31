import yaml
from flask import Blueprint, request

import web_ui.app as state
from audio.device_identity import descriptor_for_index, selectable_devices
from config.common import sync_primary_ptt_legacy_keys
from config.primitives import compressor_settings
from runtime.audit import AuditEvent
from web_ui.utils.config import _coerce, _config_path, _set_path

from .common import _error, _locked, _ok

config_bp = Blueprint("config", __name__)


def _get_path(cfg: dict, key: str):
    cur = cfg
    for part in key.split("."):
        cur = cur[part]
    return cur


def _sync_audio_device_keys(cfg: dict, key: str, coerced):
    """
    Keep the newer stable device descriptors and the legacy index fields in sync.

    The UI still sends:
      audio.input_index
      audio.output_index
      audio.output_device_2_id

    Runtime code expects:
      audio.input_index + audio.input_device
      audio.input_device_2/input_index_2 + audio.input_device_2_info
      audio.output_index + audio.output_device
      audio.output_device_2 + audio.output_device_2_info
      audio.output_2_mode
    """
    audio_cfg = cfg.setdefault("audio", {})

    if not state.audio_manager:
        return

    try:
        devices = state.audio_manager.list_devices()
    except Exception:
        return

    if key == "audio.input_index":
        try:
            idx = int(coerced)
            desc = descriptor_for_index(devices, idx, "input")
            audio_cfg["input_index"] = idx
            if desc:
                audio_cfg["input_device"] = desc
        except Exception:
            pass

    elif key == "audio.input_device_2_id":
        wanted_id = str(coerced or "")
        inputs = selectable_devices(devices, "input")

        for dev in inputs:
            if str(dev.get("id")) == wanted_id:
                idx = int(dev["index"])
                audio_cfg["input_device_2"] = idx
                audio_cfg["input_index_2"] = idx
                audio_cfg["input_device_2_id"] = dev["id"]
                audio_cfg["input_device_2_info"] = {
                    "id": dev["id"],
                    "name": dev["name"],
                }
                break

    elif key in ("audio.input_device_2", "audio.input_index_2"):
        try:
            idx = int(coerced)
            desc = descriptor_for_index(devices, idx, "input")
            audio_cfg["input_device_2"] = idx
            audio_cfg["input_index_2"] = idx
            if desc:
                audio_cfg["input_device_2_id"] = desc["id"]
                audio_cfg["input_device_2_info"] = desc
        except Exception:
            pass

    elif key == "audio.output_index":
        try:
            idx = int(coerced)
            desc = descriptor_for_index(devices, idx, "output")
            audio_cfg["output_index"] = idx
            if desc:
                audio_cfg["output_device"] = desc
        except Exception:
            pass

    elif key == "audio.output_device_2_id":
        wanted_id = str(coerced or "")
        outputs = selectable_devices(devices, "output")

        for dev in outputs:
            if str(dev.get("id")) == wanted_id:
                audio_cfg["output_device_2"] = int(dev["index"])
                audio_cfg["output_device_2_id"] = dev["id"]
                audio_cfg["output_device_2_info"] = {
                    "id": dev["id"],
                    "name": dev["name"],
                }
                break

    elif key == "audio.output_device_2":
        try:
            idx = int(coerced)
            desc = descriptor_for_index(devices, idx, "output")
            audio_cfg["output_device_2"] = idx
            if desc:
                audio_cfg["output_device_2_id"] = desc["id"]
                audio_cfg["output_device_2_info"] = desc
        except Exception:
            pass

    elif key == "audio.output_2_mode":
        mode = str(coerced or "simulcast").lower()
        audio_cfg["output_2_mode"] = "link" if mode == "link" else "simulcast"


def _apply_config_side_effects(cfg: dict, key: str, coerced):
    if key == "audio.compressor_strength":
        audio_cfg = cfg.setdefault("audio", {})

        threshold_db, ratio, makeup_db = compressor_settings(
            audio_cfg.get("compressor_strength", 50)
        )

        audio_cfg["compressor_threshold_db"] = round(threshold_db, 1)
        audio_cfg["compressor_ratio"] = round(ratio, 1)
        audio_cfg["compressor_makeup_db"] = round(makeup_db, 1)

        if audio_cfg.get("compressor_attack_ms") in (None, 10, 10.0):
            audio_cfg["compressor_attack_ms"] = 8.0

        if audio_cfg.get("compressor_release_ms") in (None, 200, 200.0):
            audio_cfg["compressor_release_ms"] = 160.0

    if key in (
        "ptt.mode",
        "ptt.device_path",
        "ptt.gpio_pin",
    ):
        suffix = key.split(".", 1)[1]

        _set_path(
            cfg,
            f"ptt.primary.{suffix}",
            cfg["ptt"][suffix],
        )

        sync_primary_ptt_legacy_keys(cfg)

    elif (
        key.startswith("ptt.primary.")
        or key.startswith("ptt.secondary.")
        or key == "ptt.dual_ptt"
    ):
        sync_primary_ptt_legacy_keys(cfg)

    if key.startswith("audio."):
        _sync_audio_device_keys(cfg, key, coerced)


@config_bp.route("/config/live", methods=["POST"])
def config_live():
    data = request.json or {}
    key = data.get("key")
    value = data.get("value")

    if not key or not isinstance(key, str):
        return _error("Missing key", 400)

    cfg = state.config.config

    try:
        old_value = _get_path(cfg, key)
    except Exception:
        old_value = None

    with state._config_lock:
        if state.config_locked:
            return _locked()

        coerced = _coerce(value)
        _set_path(cfg, key, coerced)
        _apply_config_side_effects(cfg, key, coerced)

    if hasattr(state, "audit") and state.audit:
        state.audit.info(
            event_type=AuditEvent.CONFIG_CHANGED,
            source="web_ui",
            username="local_admin",
            message="Configuration changed",
            metadata={
                "path": key,
                "old": old_value,
                "new": coerced,
            },
        )

    return _ok()


@config_bp.route("/config/apply", methods=["POST"])
def config_apply():
    cfg = state.config.config

    try:
        with state._config_lock:
            if state.config_locked:
                return _locked()

            path = _config_path()

            with open(path, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            if hasattr(state, "audit") and state.audit:
                state.audit.info(
                    event_type=AuditEvent.CONFIG_APPLIED,
                    source="web_ui",
                    username="local_admin",
                    message="Configuration applied",
                    metadata={
                        "path": str(path),
                    },
                )

        if state.lifecycle:
            with state.state_lock:
                old_lifecycle = state.lifecycle

                try:
                    old_lifecycle.cleanup()
                finally:
                    state.lifecycle = None

                audio_cfg = cfg.get("audio", {}) or {}

                try:
                    input_idx = int(audio_cfg.get("input_index"))
                    output_idx = int(audio_cfg.get("output_index"))
                except Exception:
                    return _error(
                        "Saved config has invalid input/output device indices", 500
                    )

                try:
                    state.lifecycle = state.build_repeater(
                        input_idx,
                        output_idx,
                        state.config,
                        state.audio_manager,
                        audit=getattr(state, "audit", None),
                        publish_services=state.publish_services,
                    )
                    state.lifecycle.start()
                except Exception as e:
                    return _error(f"Failed to restart: {e}", 500)

        return _ok()

    except Exception as e:
        return _error(str(e), 500)
