"""Shared configuration updates that preserve legacy settings."""


def sync_primary_ptt_legacy_keys(cfg: dict) -> None:
    """Mirror the nested primary PTT settings to legacy flat keys."""
    ptt_cfg = cfg.setdefault("ptt", {})
    primary_cfg = ptt_cfg.get("primary", {})

    if not isinstance(primary_cfg, dict):
        primary_cfg = {}
        ptt_cfg["primary"] = primary_cfg

    ptt_cfg["mode"] = primary_cfg.get("mode", ptt_cfg.get("mode", "VOX"))
    ptt_cfg["device_path"] = primary_cfg.get(
        "device_path", ptt_cfg.get("device_path", "")
    )
    ptt_cfg["gpio_pin"] = primary_cfg.get("gpio_pin", ptt_cfg.get("gpio_pin", 3))
