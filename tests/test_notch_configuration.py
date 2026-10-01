"""Regression coverage for notch mode migration, API saves, and DSP selection."""

import copy
import importlib.util
import math
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import yaml
from flask import Flask
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def package(name):
    module = ModuleType(name)
    module.__path__ = []
    return module


class FakeDSP:
    def __init__(self):
        self.calls = {}

    def __getattr__(self, method):
        return lambda **kwargs: self.calls.update({method: kwargs})


class NotchConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.modules = patch.dict(sys.modules)
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        for name in ("config", "runtime", "audio", "web_ui", "web_ui.routes", "web_ui.utils"):
            sys.modules[name] = package(name)
        audit = ModuleType("runtime.audit")
        audit.AuditEvent = SimpleNamespace(CONFIG_CHANGED="changed", CONFIG_APPLIED="applied")
        sys.modules["runtime.audit"] = audit
        devices = ModuleType("audio.device_identity")
        devices.descriptor_for_index = lambda *_: None
        devices.selectable_devices = lambda *_: []
        sys.modules["audio.device_identity"] = devices

        self.primitives = load("config.primitives", "config/primitives.py")
        defaults = load("config.template", "config/template.py").DEFAULT_CONFIG
        load("config.manager", "config/manager.py")
        load("config.common", "config/common.py")
        self.normalize = load("config.normalize", "config/normalize.py").normalize_config_for_template
        self.configure = load("notch_test_dsp", "dsp/configure.py").configure_dsp
        self.cfg = copy.deepcopy(defaults)
        self.cfg["audio"].update(input_index=0, output_index=0, notch_enabled=True)

        self.state = ModuleType("web_ui.app")
        self.state.config = SimpleNamespace(config=self.cfg, config_path=Path(self.temp.name) / "config.yaml")
        self.state._config_lock = threading.RLock()
        self.state.state_lock = threading.RLock()
        self.state.config_locked = False
        self.state.audio_manager = None
        self.state.audit = None
        self.state.lifecycle = None
        sys.modules["web_ui.app"] = self.state
        sys.modules["web_ui"].app = self.state
        load("web_ui.utils.config", "web_ui/utils/config.py")
        load("web_ui.routes.common", "web_ui/routes/common.py")
        routes = load("web_ui.routes.config", "web_ui/routes/config.py")
        app = Flask(__name__)
        app.register_blueprint(routes.config_bp)
        self.client = app.test_client()

    def update(self, key, value):
        return self.client.post("/config/live", json={"key": f"audio.{key}", "value": value})

    def test_legacy_mode_is_inferred_before_defaults_are_merged(self):
        for values, expected in (([], "harmonics"), ([60, 123], "frequencies"), (["bad", -1], "harmonics")):
            with self.subTest(values=values):
                cfg = {"audio": {"notch_frequencies_hz": values}}
                self.normalize(cfg)
                self.assertEqual(cfg["audio"]["notch_mode"], expected)
                self.assertEqual(cfg["audio"]["notch_frequencies_hz"], values)
        cfg = {"audio": {"notch_mode": "harmonics", "notch_frequencies_hz": [77]}}
        self.normalize(cfg)
        self.assertEqual(cfg["audio"]["notch_mode"], "harmonics")

    def test_mode_switch_save_reload_preserves_both_modes_and_restarts_dsp(self):
        self.assertEqual(self.update("notch_frequencies_hz", [77, 131, 77]).status_code, 200)
        self.assertEqual(self.update("notch_frequency_hz", 50).status_code, 200)
        self.assertEqual(self.update("notch_harmonics", 3).status_code, 200)
        self.assertEqual(self.update("notch_mode", "frequencies").status_code, 200)

        rebuilt = []
        def build(*_args, **_kwargs):
            rx, tx = FakeDSP(), FakeDSP()
            self.configure(self.state.config, rx, tx)
            rebuilt.append((rx, tx))
            return SimpleNamespace(start=lambda: None, cleanup=lambda: None)

        self.state.build_repeater = build
        self.state.publish_services = lambda **_kwargs: None
        self.state.lifecycle = SimpleNamespace(cleanup=lambda: None)
        for mode in ("harmonics", "frequencies"):
            self.assertEqual(self.update("notch_mode", mode).status_code, 200)
            self.assertEqual(self.client.post("/config/apply").status_code, 200)
            saved = yaml.safe_load(self.state.config.config_path.read_text())
            self.normalize(saved)
            self.assertEqual(saved["audio"]["notch_mode"], mode)
            self.assertEqual(saved["audio"]["notch_frequencies_hz"], [77.0, 131.0])
            self.assertEqual(saved["audio"]["notch_frequency_hz"], 50)
            self.assertEqual(saved["audio"]["notch_harmonics"], 3)
            selected = "configure_notch" if mode == "harmonics" else "configure_notch_frequencies"
            self.assertIn(selected, rebuilt[-1][0].calls)
            self.assertIn(selected, rebuilt[-1][1].calls)

    def test_invalid_api_updates_do_not_replace_saved_settings(self):
        self.cfg["audio"]["notch_frequencies_hz"] = [60]
        invalid = ("60, 120", [True], ["60"], [4], [24000], list(range(10, 19)), [None])
        for value in invalid:
            with self.subTest(value=value):
                self.assertEqual(self.update("notch_frequencies_hz", value).status_code, 400)
                self.assertEqual(self.cfg["audio"]["notch_frequencies_hz"], [60])
        self.assertEqual(self.update("notch_mode", "unknown").status_code, 400)
        self.assertEqual(self.cfg["audio"]["notch_mode"], "harmonics")
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                self.primitives.validate_notch_frequencies([value], 48000)
        self.cfg["audio"]["sample_rate"] = 8000
        self.assertEqual(self.update("notch_frequencies_hz", [3920]).status_code, 200)
        self.assertEqual(self.update("notch_frequencies_hz", [3921]).status_code, 400)

    def test_configuration_lock_prevents_notch_changes(self):
        self.state.config_locked = True
        self.assertEqual(self.update("notch_mode", "frequencies").status_code, 423)
        self.assertEqual(self.update("notch_frequencies_hz", [77]).status_code, 423)
        self.assertEqual(self.cfg["audio"]["notch_mode"], "harmonics")
        self.assertEqual(self.cfg["audio"]["notch_frequencies_hz"], [])

    def test_empty_individual_mode_and_yaml_tx_switch(self):
        self.cfg["audio"].update(notch_mode="frequencies", notch_frequencies_hz=[])
        rx, tx = FakeDSP(), FakeDSP()
        self.configure(self.state.config, rx, tx)
        self.assertNotIn("configure_notch", rx.calls)
        self.assertFalse(rx.calls["configure_notch_frequencies"]["enabled"])
        self.cfg["audio"].update(notch_frequencies_hz=[77], notch_apply_to_tx=False)
        self.configure(self.state.config, rx, tx)
        self.assertTrue(rx.calls["configure_notch_frequencies"]["enabled"])
        self.assertFalse(tx.calls["configure_notch_frequencies"]["enabled"])

    def test_template_renders_retained_values_and_excludes_hidden_settings(self):
        self.cfg["audio"].update(notch_mode="harmonics", notch_frequencies_hz=[77, 131])
        env = Environment(loader=FileSystemLoader(ROOT / "web_ui/templates"), autoescape=True)
        html = env.get_template("index.html").render(config=self.cfg, input_devices=[], output_devices=[])
        self.assertIn('value="harmonics" selected', html)
        self.assertIn('value="77, 131"', html)
        for key in ("notch_apply_to_tx", "rx_fade_in_ms", "rx_fade_in_start_gain"):
            self.assertNotIn(f'data-key="audio.{key}"', html)


if __name__ == "__main__":
    unittest.main()
