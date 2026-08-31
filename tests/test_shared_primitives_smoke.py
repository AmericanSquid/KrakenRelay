"""Dependency-free smoke tests for extracted shared helpers."""

import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(relative_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SharedPrimitivesSmokeTests(unittest.TestCase):
    def test_compressor_settings_clamps_and_maps_strength(self):
        primitives = load_module("config/primitives.py", "smoke_config_primitives")

        self.assertEqual(primitives.compressor_settings(-1), (-15.0, 1.8, 2.5))
        self.assertEqual(primitives.compressor_settings(50), (-20.0, 3.0, 3.75))
        self.assertEqual(primitives.compressor_settings(101), (-25.0, 4.2, 5.0))

    def test_ptt_legacy_keys_follow_primary_configuration(self):
        common = load_module("config/common.py", "smoke_config_common")
        cfg = {
            "ptt": {
                "mode": "VOX",
                "device_path": "",
                "gpio_pin": 3,
                "primary": {
                    "mode": "CM108",
                    "device_path": "/dev/hidraw1",
                    "gpio_pin": 5,
                },
            }
        }

        common.sync_primary_ptt_legacy_keys(cfg)

        self.assertEqual(
            cfg["ptt"],
            {
                "mode": "CM108",
                "device_path": "/dev/hidraw1",
                "gpio_pin": 5,
                "primary": {
                    "mode": "CM108",
                    "device_path": "/dev/hidraw1",
                    "gpio_pin": 5,
                },
            },
        )

    def test_signal_gate_primitives_reset_only_probe_state(self):
        primitives = load_module("core/primitives.py", "smoke_core_primitives")
        gate = SimpleNamespace(
            _carrier_valid=True,
            _carrier_probe_start=10.0,
            _carrier_last_level_db=-25.0,
            squelch_open=True,
        )

        primitives.reset_carrier_probe(gate)

        self.assertFalse(gate._carrier_valid)
        self.assertIsNone(gate._carrier_probe_start)
        self.assertIsNone(gate._carrier_last_level_db)
        self.assertTrue(gate.squelch_open)
        self.assertTrue(primitives.is_squelch_open_edge(True, False))
        self.assertTrue(primitives.is_squelch_close_edge(False, True))

    def test_shutdown_helper_falls_back_to_unkey(self):
        common = load_module("core/common.py", "smoke_core_common")
        calls = []

        def stop_transmission():
            calls.append("stop")
            raise RuntimeError("stop failed")

        def unkey_transmitter():
            calls.append("unkey")

        common.shutdown_transmitter(
            SimpleNamespace(transmitting=True),
            stop_transmission,
            unkey_transmitter,
        )

        self.assertEqual(calls, ["stop", "unkey"])

    def test_plugin_file_helpers_preserve_existing_backup(self):
        common = load_module("plugins/common.py", "smoke_plugin_common")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "target.txt"
            common.write_text(path, "first")
            with redirect_stdout(io.StringIO()):
                common.backup_file(path, suffix=".snapshot")
            common.write_text(path, "second")
            with redirect_stdout(io.StringIO()):
                common.backup_file(path, suffix=".snapshot")

            self.assertEqual(common.read_text(path), "second")
            self.assertEqual(
                common.read_text(path.with_suffix(".txt.snapshot")), "first"
            )


if __name__ == "__main__":
    unittest.main()
