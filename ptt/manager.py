import logging
import os

from runtime.audit import AuditEvent

from .modes import CM108PTT


class PTTManager:
    def __init__(self, config, audit=None):
        # PTT Setup (dual + single + legacy)
        self.config = config
        self.audit = audit

        self.ptt = None
        self.ptt_2 = None
        self.ptt_mode = "VOX"
        self.ptt_2_mode = "VOX"

        self.ptt_fallback = False
        self.ptt_2_fallback = False

        # Prevent audit spam if get_ptt_status() is polled by the UI.
        self._missing_device_alerted = set()

        ptt_cfg = self.config.config.get("ptt", {})

        if ptt_cfg.get("dual_ptt", False):
            # Dual PTT mode: read both
            primary_cfg = ptt_cfg.get("primary", {})
            secondary_cfg = ptt_cfg.get("secondary", {})

            self.ptt, self.ptt_mode = self._build_ptt(
                primary_cfg,
                "/dev/hidraw0",
            )

            if self.ptt_mode == "CM108":
                logging.info(
                    "Primary PTT (CM108) on %s, GPIO %s",
                    self.ptt.device,
                    self.ptt.pin,
                )
            else:
                logging.info("Primary PTT mode set to VOX (no GPIO control)")

            self.ptt_2, self.ptt_2_mode = self._build_ptt(
                secondary_cfg,
                "/dev/hidraw3",
            )

            if self.ptt_2_mode == "CM108":
                logging.info(
                    "Secondary PTT (CM108) on %s, GPIO %s",
                    self.ptt_2.device,
                    self.ptt_2.pin,
                )
            else:
                logging.info("Secondary PTT mode set to VOX (no GPIO control)")

            logging.info(
                "[PTT] init secondary: mode=%s, dev=%s, pin=%s",
                self.ptt_2_mode,
                getattr(self.ptt_2, "device", None),
                getattr(self.ptt_2, "pin", None),
            )

        else:
            # Single PTT mode: check for new-style `primary` first, fallback to legacy
            primary_cfg = ptt_cfg.get("primary", ptt_cfg)

            self.ptt, self.ptt_mode = self._build_ptt(
                primary_cfg,
                "/dev/hidraw0",
            )

            if self.ptt_mode == "CM108":
                logging.info(
                    "PTT mode set to CM108 (device=%s, pin=%s)",
                    self.ptt.device,
                    self.ptt.pin,
                )
            else:
                logging.info("PTT mode set to VOX (no GPIO control)")

    def _audit_error(self, event_type, message, metadata=None):
        if not self.audit:
            return

        self.audit.error(
            event_type=event_type,
            source="ptt_manager",
            message=message,
            metadata=metadata or {},
        )

    def _build_ptt(self, cfg, default_device):
        if cfg.get("mode", "").upper() == "CM108":
            ptt = CM108PTT(
                device=cfg.get("device_path", default_device),
                pin=int(cfg.get("gpio_pin", 3)),
            )
            return ptt, "CM108"

        return None, "VOX"

    def safe_ptt_key(self):
        primary_success = False
        secondary_success = False

        # PRIMARY PTT
        if self.ptt_mode == "CM108" and self.ptt:
            try:
                if getattr(self.ptt, "working", True):
                    self.ptt.key()
                    primary_success = True

            except Exception as e:
                logging.error("Primary PTT key error: %s", e)

                self._audit_error(
                    event_type=AuditEvent.PTT_FAILURE,
                    message="Primary PTT key failed; falling back to VOX",
                    metadata={
                        "ptt": "primary",
                        "action": "key",
                        "error": repr(e),
                        "previous_mode": "CM108",
                        "device": getattr(self.ptt, "device", None),
                        "pin": getattr(self.ptt, "pin", None),
                    },
                )

                self.ptt = None
                self.ptt_mode = "VOX"
                self.ptt_fallback = True

                logging.warning("Primary PTT failed — fallback to VOX")

        # SECONDARY PTT
        if self.ptt_2_mode == "CM108" and self.ptt_2:
            try:
                if getattr(self.ptt_2, "working", True):
                    self.ptt_2.key()
                    logging.info("Secondary PTT keyed")
                    secondary_success = True

            except Exception as e:
                logging.error("Secondary PTT key error: %s", e)

                self._audit_error(
                    event_type=AuditEvent.PTT_FAILURE,
                    message="Secondary PTT key failed; falling back to VOX",
                    metadata={
                        "ptt": "secondary",
                        "action": "key",
                        "error": repr(e),
                        "previous_mode": "CM108",
                        "device": getattr(self.ptt_2, "device", None),
                        "pin": getattr(self.ptt_2, "pin", None),
                    },
                )

                self.ptt_2 = None
                self.ptt_2_mode = "VOX"
                self.ptt_2_fallback = True

                logging.warning("Secondary PTT failed — fallback to VOX")

        return primary_success or secondary_success

    def safe_ptt_unkey(self):
        primary_success = False
        secondary_success = False

        # PRIMARY PTT
        if self.ptt_mode == "CM108" and self.ptt:
            try:
                if getattr(self.ptt, "working", True):
                    self.ptt.unkey()
                    primary_success = True

            except Exception as e:
                logging.error("Primary PTT unkey error: %s", e)

                self._audit_error(
                    event_type=AuditEvent.PTT_FAILURE,
                    message="Primary PTT unkey failed; falling back to VOX",
                    metadata={
                        "ptt": "primary",
                        "action": "unkey",
                        "error": repr(e),
                        "previous_mode": "CM108",
                        "device": getattr(self.ptt, "device", None),
                        "pin": getattr(self.ptt, "pin", None),
                    },
                )

                self.ptt = None
                self.ptt_mode = "VOX"
                self.ptt_fallback = True

                logging.warning("Primary PTT unkey failed — fallback to VOX")

        # SECONDARY PTT
        if self.ptt_2_mode == "CM108" and self.ptt_2:
            try:
                if getattr(self.ptt_2, "working", True):
                    self.ptt_2.unkey()
                    secondary_success = True

            except Exception as e:
                logging.error("Secondary PTT unkey error: %s", e)

                self._audit_error(
                    event_type=AuditEvent.PTT_FAILURE,
                    message="Secondary PTT unkey failed; falling back to VOX",
                    metadata={
                        "ptt": "secondary",
                        "action": "unkey",
                        "error": repr(e),
                        "previous_mode": "CM108",
                        "device": getattr(self.ptt_2, "device", None),
                        "pin": getattr(self.ptt_2, "pin", None),
                    },
                )

                self.ptt_2 = None
                self.ptt_2_mode = "VOX"
                self.ptt_2_fallback = True

                logging.warning("Secondary PTT unkey failed — fallback to VOX")

        return primary_success or secondary_success

    def _audit_missing_device_once(self, label, device, mode):
        key = (label, device, mode)

        if key in self._missing_device_alerted:
            return

        self._missing_device_alerted.add(key)

        self._audit_error(
            event_type=AuditEvent.PTT_DEVICE_MISSING,
            message=f"{label} PTT device missing",
            metadata={
                "ptt": label.lower(),
                "device": device,
                "mode": mode,
            },
        )

    def _clear_missing_device_alert(self, label, device, mode):
        key = (label, device, mode)
        self._missing_device_alerted.discard(key)

    def _get_device_status(self, label, device, ptt, mode):
        if ptt:
            if not os.path.exists(device):
                self._audit_missing_device_once(label, device, mode)
                return (f"{label}: Device Missing ({device})", "red")

            self._clear_missing_device_alert(label, device, mode)

            try:
                with open(device, "wb", buffering=0):
                    pass

                if getattr(ptt, "working", True):
                    return (f"{label}: OK ({device})", "green")

                return (f"{label}: Detected, Last Op Failed ({device})", "orange")

            except PermissionError:
                return (f"{label}: Permission Denied ({device})", "orange")

            except Exception:
                return (f"{label}: Unusable ({device})", "orange")

        else:
            if mode == "CM108":
                if not os.path.exists(device):
                    self._audit_missing_device_once(label, device, mode)
                    return (f"{label}: Device Missing ({device})", "red")

                return (f"{label}: Not Configured ({device})", "red")

            return (f"{label}: VOX Mode", "blue")

    def get_ptt_status(self):
        ptt_cfg = self.config.config.get("ptt", {})
        dual_mode = ptt_cfg.get("dual_ptt", False)

        statuses = []

        # Determine where to look for primary PTT config
        primary_cfg = ptt_cfg.get("primary", ptt_cfg)
        device1 = primary_cfg.get("device_path", "/dev/hidraw0")
        mode1 = primary_cfg.get("mode", "NONE").upper()

        statuses.append(
            self._get_device_status(
                "Primary",
                device1,
                self.ptt,
                mode1,
            )
        )

        # If not dual PTT, return now
        if not dual_mode:
            return statuses[0]

        # Check secondary
        secondary_cfg = ptt_cfg.get("secondary", {})
        device2 = secondary_cfg.get("device_path", "/dev/hidraw3")
        mode2 = secondary_cfg.get("mode", "NONE").upper()

        statuses.append(
            self._get_device_status(
                "Secondary",
                device2,
                self.ptt_2,
                mode2,
            )
        )

        # Combine both into one string for GUI
        combined_status = " | ".join([s[0] for s in statuses])

        color = "green"

        if any(c == "red" for _, c in statuses):
            color = "red"

        elif any(c == "orange" for _, c in statuses):
            color = "orange"

        elif all(c == "blue" for _, c in statuses):
            color = "blue"

        return (combined_status, color)
