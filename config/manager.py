import copy
import logging

import yaml

from runtime.audit import AuditEvent

from .template import DEFAULT_CONFIG


class ConfigManager:
    def __init__(self, config_path="config.yaml", audit=None):
        self.config_path = config_path
        self.audit = audit

        self.config = self.load_config()

        if self.config is None:
            logging.warning(
                "No config found at '%s' — generating default.",
                self.config_path,
            )

            self.config = self.get_default_config()
            self.save_config()

            logging.info(
                "Default config saved to '%s'",
                self.config_path,
            )
        else:
            logging.info("Config loaded successfully.")

    def _audit_info(self, event_type, message, metadata=None):
        if not self.audit:
            return

        self.audit.info(
            event_type=event_type,
            source="config_manager",
            message=message,
            metadata=metadata or {},
        )

    def _audit_error(self, event_type, message, metadata=None):
        if not self.audit:
            return

        self.audit.error(
            event_type=event_type,
            source="config_manager",
            message=message,
            metadata=metadata or {},
        )

    def load_config(self):
        try:
            with open(self.config_path, "r") as file:
                config = yaml.safe_load(file)

            logging.info("Configuration loaded successfully")

            if config is not None:
                self._audit_info(
                    event_type=AuditEvent.CONFIG_RELOADED,
                    message="Configuration loaded from disk",
                    metadata={
                        "path": self.config_path,
                    },
                )

            return config

        except Exception as e:
            logging.error("Error loading configuration: %s", e)

            return self.get_default_config()

    def save_config(self):
        try:
            with open(self.config_path, "w") as file:
                yaml.dump(self.config, file)

            logging.info("Configuration saved successfully")

        except Exception as e:
            logging.error("Error saving configuration: %s", e)

            self._audit_error(
                event_type=AuditEvent.CONFIG_SAVE_FAILED,
                message="Configuration save failed",
                metadata={
                    "path": self.config_path,
                    "error": repr(e),
                },
            )

    def get_default_config(self):
        return copy.deepcopy(DEFAULT_CONFIG)
