from .manager import ConfigManager

def normalize_config_for_template(cfg: dict) -> None:
    """
    Keep your template happy and provide sensible defaults.
    Current index.html uses flat ptt keys: ptt.mode / ptt.device_path / ptt.gpio_pin
    """
    cfg.setdefault("ptt", {})
    ptt = cfg["ptt"]
    if not isinstance(ptt, dict):
        cfg["ptt"] = {}
        ptt = cfg["ptt"]

    # Ensure flat keys exist
    ptt.setdefault("mode", "VOX")
    ptt.setdefault("device_path", "")
    ptt.setdefault("gpio_pin", 3)

    # Ensure top-level sections exist
    for section in ("audio", "repeater", "identification", "tot"):
        cfg.setdefault(section, {})

    # Reasonable default: do NOT autostart unless configured
    cfg["repeater"].setdefault("auto_start", False)

def _config_accepts_path() -> bool:
    try:
        ConfigManager("config.yaml")  # type: ignore[arg-type]
    except TypeError:
        return False
    return True
