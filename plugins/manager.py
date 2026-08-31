import importlib
import logging
from pathlib import Path

import yaml
from flask import Response


class PluginManager:
    """
    Small generic plugin manager for KrakenRelay.

    Config:
        plugins/config.yaml

    Plugin module convention:
        plugins/<name>/plugin.py must expose load_plugin(config, audio_config)

    Optional plugin hooks:
        on_audio_frame(samples)
        on_tick()
        on_shutdown()

    Optional HTTP actions:
        api_<action>(flask_request)
    """

    def __init__(self, repeater_config, config_path=None):
        self.repeater_config = repeater_config
        self.plugins = []

        if config_path is None:
            config_path = Path(__file__).resolve().parent / "config.yaml"

        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self):
        if not self.config_path.exists():
            logging.warning("[Plugins] Plugin config not found: %s", self.config_path)
            return {"enabled": [], "plugins": {}}

        try:
            with self.config_path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            logging.exception(
                "[Plugins] Failed to load plugin config: %s", self.config_path
            )
            return {"enabled": [], "plugins": {}}

        data.setdefault("enabled", [])
        data.setdefault("plugins", {})
        return data

    def load_enabled(self):
        enabled = self.config.get("enabled", []) or []
        plugin_configs = self.config.get("plugins", {}) or {}

        for plugin_name in enabled:
            self._load_plugin(plugin_name, plugin_configs.get(plugin_name, {}) or {})

    def _load_plugin(self, plugin_name, plugin_config):
        module_name = f"plugins.{plugin_name}.plugin"

        try:
            module = importlib.import_module(module_name)
        except Exception:
            logging.exception("[Plugins] Failed to import plugin: %s", plugin_name)
            return

        load_plugin = getattr(module, "load_plugin", None)
        if load_plugin is None:
            logging.error(
                "[Plugins] Plugin %s does not define load_plugin()", plugin_name
            )
            return

        try:
            plugin = load_plugin(
                config=plugin_config,
                audio_config=self.repeater_config.config.get("audio", {}),
            )
        except Exception:
            logging.exception("[Plugins] Failed to initialize plugin: %s", plugin_name)
            return

        self.plugins.append(plugin)
        logging.info("[Plugins] Loaded plugin: %s", plugin_name)

    def get_plugin(self, plugin_name):
        for plugin in self.plugins:
            if getattr(plugin, "name", None) == plugin_name:
                return plugin
        return None

    def emit_audio_frame(self, samples):
        for plugin in list(self.plugins):
            hook = getattr(plugin, "on_audio_frame", None)
            if hook is None:
                continue
            try:
                hook(samples)
            except Exception:
                logging.exception(
                    "[Plugins] on_audio_frame failed for plugin: %s",
                    getattr(plugin, "name", plugin),
                )

    def emit_tick(self):
        for plugin in list(self.plugins):
            hook = getattr(plugin, "on_tick", None)
            if hook is None:
                continue
            try:
                hook()
            except Exception:
                logging.exception(
                    "[Plugins] on_tick failed for plugin: %s",
                    getattr(plugin, "name", plugin),
                )

    def emit_shutdown(self):
        for plugin in list(self.plugins):
            hook = getattr(plugin, "on_shutdown", None)
            if hook is None:
                continue
            try:
                hook()
            except Exception:
                logging.exception(
                    "[Plugins] on_shutdown failed for plugin: %s",
                    getattr(plugin, "name", plugin),
                )

    def dispatch_http(self, plugin_name, action, flask_request):
        plugin = self.get_plugin(plugin_name)
        if plugin is None:
            return {"ok": False, "error": f"Plugin not loaded: {plugin_name}"}, 404

        handler = getattr(plugin, f"api_{action}", None)
        if handler is None:
            return {
                "ok": False,
                "error": f"Plugin action not available: {plugin_name}/{action}",
            }, 404

        try:
            result = handler(flask_request)
        except Exception as exc:
            logging.exception(
                "[Plugins] HTTP dispatch failed: %s/%s", plugin_name, action
            )
            return {"ok": False, "error": str(exc)}, 500

        if isinstance(result, Response):
            return result, 200
        if isinstance(result, tuple):
            return result
        return result, 200
