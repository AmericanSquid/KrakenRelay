import logging
from typing import Any, Dict, Tuple

from .uvk5d import UVK5DPTT

UVK5D_ALIASES = {"UVK5D", "UV-K5D", "UV_K5D"}


def _normalize_mode(mode: Any) -> str:
    text = str(mode or "").strip().upper()
    if text in UVK5D_ALIASES:
        return "UVK5D"
    return text


def _cfg_get(cfg: Dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in cfg:
            return cfg.get(name)
    return default


def _ptt_device_and_mode(cfg: Dict[str, Any], default_device: str) -> Tuple[str, str]:
    mode = _normalize_mode(cfg.get("mode", "NONE"))

    if mode == "UVK5D":
        host = _cfg_get(cfg, "uvk5d_host", "host", default="127.0.0.1")
        port = _cfg_get(cfg, "uvk5d_port", "port", default=7355)
        return f"{host}:{port}", "UVK5D"

    return cfg.get("device_path", default_device), mode


def _audit_error(
    manager: Any, event_type: Any, message: str, metadata: Dict[str, Any]
) -> None:
    audit = getattr(manager, "audit", None)
    if not audit:
        return

    try:
        audit.error(
            event_type=event_type,
            source="ptt_manager",
            message=message,
            metadata=metadata or {},
        )
    except Exception:
        logging.exception("[UVK5DPTTPlugin] Failed to write PTT audit event")


def _install_ptt_manager_patch() -> bool:
    """
    Install UVK5D support into PTTManager at runtime.

    KrakenRelay's current plugin API exposes audio/tick/shutdown/http hooks, not a
    formal PTT-backend registry. This plugin is loaded early by core.initialize,
    before PTTManager is instantiated, so a small runtime patch keeps UVK5D support
    plugin-contained without editing ptt/manager.py or ptt/modes/__init__.py.
    """

    from ptt.manager import PTTManager
    from runtime.audit import AuditEvent

    if getattr(PTTManager, "_uvk5d_ptt_plugin_patched", False):
        return False

    original_build_ptt = getattr(PTTManager, "_build_ptt")
    original_is_hardware_ptt = getattr(PTTManager, "_is_hardware_ptt", None)
    original_get_device_status = getattr(PTTManager, "_get_device_status", None)

    PTTManager._uvk5d_ptt_plugin_originals = {
        "_build_ptt": original_build_ptt,
        "_is_hardware_ptt": original_is_hardware_ptt,
        "_get_device_status": original_get_device_status,
        "safe_ptt_key": getattr(PTTManager, "safe_ptt_key", None),
        "safe_ptt_unkey": getattr(PTTManager, "safe_ptt_unkey", None),
        "get_ptt_status": getattr(PTTManager, "get_ptt_status", None),
    }

    def _is_hardware_ptt(self, mode):
        normalized = _normalize_mode(mode)
        if normalized == "UVK5D":
            return True
        if original_is_hardware_ptt is not None:
            return original_is_hardware_ptt(self, mode)
        return normalized == "CM108"

    def _build_ptt(self, cfg, default_device):
        cfg = cfg or {}
        mode = _normalize_mode(cfg.get("mode", ""))

        if mode == "UVK5D":
            ptt = UVK5DPTT(
                host=_cfg_get(cfg, "uvk5d_host", "host", default="127.0.0.1"),
                port=int(_cfg_get(cfg, "uvk5d_port", "port", default=7355)),
                timeout=float(_cfg_get(cfg, "uvk5d_timeout", "timeout", default=0.5)),
            )
            return ptt, "UVK5D"

        return original_build_ptt(self, cfg, default_device)

    def safe_ptt_key(self):
        primary_success = False
        secondary_success = False
        event_type = getattr(AuditEvent, "PTT_FAILURE", "ptt_failure")

        if self._is_hardware_ptt(getattr(self, "ptt_mode", None)) and getattr(
            self, "ptt", None
        ):
            try:
                if getattr(self.ptt, "working", True):
                    self.ptt.key()
                    primary_success = True
            except Exception as exc:
                logging.error("Primary PTT key error: %s", exc)
                _audit_error(
                    self,
                    event_type,
                    "Primary PTT key failed; falling back to VOX",
                    {
                        "ptt": "primary",
                        "action": "key",
                        "error": repr(exc),
                        "previous_mode": getattr(self, "ptt_mode", None),
                        "device": getattr(getattr(self, "ptt", None), "device", None),
                        "pin": getattr(getattr(self, "ptt", None), "pin", None),
                    },
                )
                self.ptt = None
                self.ptt_mode = "VOX"
                self.ptt_fallback = True
                logging.warning("Primary PTT failed - fallback to VOX")

        if self._is_hardware_ptt(getattr(self, "ptt_2_mode", None)) and getattr(
            self, "ptt_2", None
        ):
            try:
                if getattr(self.ptt_2, "working", True):
                    self.ptt_2.key()
                    logging.info("Secondary PTT keyed")
                    secondary_success = True
            except Exception as exc:
                logging.error("Secondary PTT key error: %s", exc)
                _audit_error(
                    self,
                    event_type,
                    "Secondary PTT key failed; falling back to VOX",
                    {
                        "ptt": "secondary",
                        "action": "key",
                        "error": repr(exc),
                        "previous_mode": getattr(self, "ptt_2_mode", None),
                        "device": getattr(getattr(self, "ptt_2", None), "device", None),
                        "pin": getattr(getattr(self, "ptt_2", None), "pin", None),
                    },
                )
                self.ptt_2 = None
                self.ptt_2_mode = "VOX"
                self.ptt_2_fallback = True
                logging.warning("Secondary PTT failed - fallback to VOX")

        return primary_success or secondary_success

    def safe_ptt_unkey(self):
        primary_success = False
        secondary_success = False
        event_type = getattr(AuditEvent, "PTT_FAILURE", "ptt_failure")

        if self._is_hardware_ptt(getattr(self, "ptt_mode", None)) and getattr(
            self, "ptt", None
        ):
            try:
                self.ptt.unkey()
                primary_success = True
            except Exception as exc:
                logging.error("Primary PTT unkey error: %s", exc)
                _audit_error(
                    self,
                    event_type,
                    "Primary PTT unkey failed; falling back to VOX",
                    {
                        "ptt": "primary",
                        "action": "unkey",
                        "error": repr(exc),
                        "previous_mode": getattr(self, "ptt_mode", None),
                        "device": getattr(getattr(self, "ptt", None), "device", None),
                        "pin": getattr(getattr(self, "ptt", None), "pin", None),
                    },
                )
                self.ptt = None
                self.ptt_mode = "VOX"
                self.ptt_fallback = True
                logging.warning("Primary PTT unkey failed - fallback to VOX")

        if self._is_hardware_ptt(getattr(self, "ptt_2_mode", None)) and getattr(
            self, "ptt_2", None
        ):
            try:
                if getattr(self.ptt_2, "working", True):
                    self.ptt_2.unkey()
                    secondary_success = True
            except Exception as exc:
                logging.error("Secondary PTT unkey error: %s", exc)
                _audit_error(
                    self,
                    event_type,
                    "Secondary PTT unkey failed; falling back to VOX",
                    {
                        "ptt": "secondary",
                        "action": "unkey",
                        "error": repr(exc),
                        "previous_mode": getattr(self, "ptt_2_mode", None),
                        "device": getattr(getattr(self, "ptt_2", None), "device", None),
                        "pin": getattr(getattr(self, "ptt_2", None), "pin", None),
                    },
                )
                self.ptt_2 = None
                self.ptt_2_mode = "VOX"
                self.ptt_2_fallback = True
                logging.warning("Secondary PTT unkey failed - fallback to VOX")

        return primary_success or secondary_success

    def _get_device_status(self, label, device, ptt, mode):
        normalized = _normalize_mode(mode)

        if normalized == "UVK5D":
            if ptt is None:
                return (f"{label}: UVK5D Not Configured", "red")

            status = getattr(ptt, "status", None)
            if status is not None:
                message, color = status()
                if not str(message).startswith(f"{label}:"):
                    message = f"{label}: {message}"
                return (message, color)

            if getattr(ptt, "working", True):
                return (f"{label}: UVK5D OK", "green")

            return (f"{label}: UVK5D Last Op Failed", "orange")

        if original_get_device_status is not None:
            return original_get_device_status(self, label, device, ptt, mode)

        if ptt and getattr(ptt, "working", True):
            return (f"{label}: OK ({device})", "green")
        return (f"{label}: VOX Mode", "blue")

    def get_ptt_status(self):
        ptt_cfg = self.config.config.get("ptt", {})
        dual_mode = ptt_cfg.get("dual_ptt", False)
        statuses = []

        primary_cfg = ptt_cfg.get("primary", ptt_cfg)
        device1, mode1 = _ptt_device_and_mode(primary_cfg, "/dev/hidraw0")
        statuses.append(self._get_device_status("Primary", device1, self.ptt, mode1))

        if not dual_mode:
            return statuses[0]

        secondary_cfg = ptt_cfg.get("secondary", {})
        device2, mode2 = _ptt_device_and_mode(secondary_cfg, "/dev/hidraw3")
        statuses.append(
            self._get_device_status("Secondary", device2, self.ptt_2, mode2)
        )

        combined_status = " | ".join([status[0] for status in statuses])
        color = "green"
        if any(status[1] == "red" for status in statuses):
            color = "red"
        elif any(status[1] == "orange" for status in statuses):
            color = "orange"
        elif all(status[1] == "blue" for status in statuses):
            color = "blue"

        return (combined_status, color)

    PTTManager._is_hardware_ptt = _is_hardware_ptt
    PTTManager._build_ptt = _build_ptt
    PTTManager.safe_ptt_key = safe_ptt_key
    PTTManager.safe_ptt_unkey = safe_ptt_unkey
    PTTManager._get_device_status = _get_device_status
    PTTManager.get_ptt_status = get_ptt_status
    PTTManager._uvk5d_ptt_plugin_patched = True

    logging.info("[UVK5DPTTPlugin] Patched PTTManager for UVK5D backend support")
    return True


def _install_control_patch() -> bool:
    """Prevent the VOX carrier-delay wake burst when UVK5D hardware PTT is active."""

    try:
        from core.transmit.control import Control
    except Exception:
        logging.exception(
            "[UVK5DPTTPlugin] Could not patch TX control VOX-delay behavior"
        )
        return False

    if getattr(Control, "_uvk5d_ptt_plugin_patched", False):
        return False

    original_handle_vox_delay = Control._handle_vox_delay
    Control._uvk5d_ptt_original_handle_vox_delay = original_handle_vox_delay

    def _handle_vox_delay(self, repeater_cfg):
        ptt_manager = getattr(self, "ptt_manager", None)
        if ptt_manager is None:
            return original_handle_vox_delay(self, repeater_cfg)

        is_hardware = getattr(ptt_manager, "_is_hardware_ptt", None)
        if callable(is_hardware):
            if is_hardware(getattr(ptt_manager, "ptt_mode", "VOX")):
                return
            if is_hardware(getattr(ptt_manager, "ptt_2_mode", "VOX")):
                return
        elif _normalize_mode(getattr(ptt_manager, "ptt_mode", "VOX")) != "VOX":
            return

        return original_handle_vox_delay(self, repeater_cfg)

    Control._handle_vox_delay = _handle_vox_delay
    Control._uvk5d_ptt_plugin_patched = True

    logging.info("[UVK5DPTTPlugin] Patched TX control VOX-delay behavior")
    return True


class UVK5DPTTPlugin:
    name = "uvk5d_ptt"

    def __init__(self, config=None, audio_config=None):
        self.config = config or {}
        self.audio_config = audio_config or {}
        self.ptt_patch_installed = _install_ptt_manager_patch()
        self.control_patch_installed = _install_control_patch()

    def status(self):
        return {
            "plugin": self.name,
            "ptt_patch_installed": self.ptt_patch_installed,
            "control_patch_installed": self.control_patch_installed,
        }

    def api_status(self, flask_request=None):
        return self.status()

    def on_shutdown(self):
        # Intentionally keep runtime patches installed for the life of this process.
        return None


def load_plugin(config=None, audio_config=None):
    return UVK5DPTTPlugin(config=config, audio_config=audio_config)
