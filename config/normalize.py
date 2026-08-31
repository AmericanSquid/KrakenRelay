import copy

from .manager import ConfigManager
from .primitives import compressor_settings
from .template import DEFAULT_CONFIG


def _merge_defaults(target: dict, defaults: dict) -> None:
    for key, value in defaults.items():
        if isinstance(value, dict):
            if not isinstance(target.get(key), dict):
                target[key] = {}
            _merge_defaults(target[key], value)
        else:
            target.setdefault(key, copy.deepcopy(value))


def _normalize_legacy_compressor(audio_cfg: dict) -> None:
    defaults = DEFAULT_CONFIG["audio"]
    advanced_keys = (
        "compressor_threshold_db",
        "compressor_ratio",
        "compressor_makeup_db",
        "compressor_attack_ms",
        "compressor_release_ms",
    )
    untouched = all(
        audio_cfg.get(key, defaults[key]) == defaults[key] for key in advanced_keys
    )
    if not untouched:
        return
    threshold_db, ratio, makeup_db = compressor_settings(
        audio_cfg.get("compressor_strength", defaults["compressor_strength"])
    )
    audio_cfg["compressor_threshold_db"] = round(threshold_db, 1)
    audio_cfg["compressor_ratio"] = round(ratio, 1)
    audio_cfg["compressor_makeup_db"] = round(makeup_db, 1)
    audio_cfg["compressor_attack_ms"] = 8.0
    audio_cfg["compressor_release_ms"] = 160.0


def normalize_config_for_template(cfg: dict) -> None:
    if not isinstance(cfg, dict):
        return
    _merge_defaults(cfg, DEFAULT_CONFIG)

    ptt = cfg.setdefault("ptt", {})
    if not isinstance(ptt, dict):
        cfg["ptt"] = copy.deepcopy(DEFAULT_CONFIG["ptt"])
        ptt = cfg["ptt"]

    primary_defaults = DEFAULT_CONFIG["ptt"]["primary"]
    secondary_defaults = DEFAULT_CONFIG["ptt"]["secondary"]
    flat_mode = ptt.get("mode", primary_defaults["mode"])
    flat_device = ptt.get("device_path", primary_defaults["device_path"])
    flat_pin = ptt.get("gpio_pin", primary_defaults["gpio_pin"])

    primary = ptt.get("primary")
    if not isinstance(primary, dict):
        primary = {}
        ptt["primary"] = primary
    primary.setdefault("mode", flat_mode)
    primary.setdefault("device_path", flat_device)
    primary.setdefault("gpio_pin", flat_pin)

    secondary = ptt.get("secondary")
    if not isinstance(secondary, dict):
        secondary = {}
        ptt["secondary"] = secondary
    secondary.setdefault("mode", secondary_defaults["mode"])
    secondary.setdefault("device_path", secondary_defaults["device_path"])
    secondary.setdefault("gpio_pin", secondary_defaults["gpio_pin"])

    ptt.setdefault("dual_ptt", DEFAULT_CONFIG["ptt"]["dual_ptt"])
    ptt["mode"] = primary.get("mode", flat_mode)
    ptt["device_path"] = primary.get("device_path", flat_device)
    ptt["gpio_pin"] = primary.get("gpio_pin", flat_pin)

    cfg["repeater"].setdefault("auto_start", False)
    _normalize_legacy_compressor(cfg["audio"])


def _config_accepts_path() -> bool:
    try:
        ConfigManager("config.yaml")  # type: ignore[arg-type]
    except TypeError:
        return False
    return True
