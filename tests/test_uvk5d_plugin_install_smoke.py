"""Smoke coverage for UVK5D's generic-installer integration."""

import sys
import tempfile
import types
import unittest
from pathlib import Path
from shutil import copy2
from unittest.mock import patch


def _safe_load(stream):
    enabled = []
    in_enabled = False
    for raw_line in stream.read().splitlines():
        if raw_line == "enabled:":
            in_enabled = True
        elif raw_line and not raw_line.startswith((" ", "\t")):
            in_enabled = False
        elif in_enabled and raw_line.strip().startswith("- "):
            enabled.append(raw_line.strip()[2:])
    return {"enabled": enabled, "plugins": {}}


def _safe_dump(data, stream, **_kwargs):
    stream.write("enabled:\n")
    for name in data["enabled"]:
        stream.write(f"  - {name}\n")
    stream.write("plugins:\n")
    for name in data["plugins"]:
        stream.write(f"  {name}: {{}}\n")


class UVK5DPluginInstallSmokeTest(unittest.TestCase):
    def test_generic_installer_adds_uvk5d_to_both_ptt_dropdowns(self):
        repo_root = Path(__file__).resolve().parents[1]
        yaml_stub = types.SimpleNamespace(safe_load=_safe_load, safe_dump=_safe_dump)

        with patch.dict(sys.modules, {"yaml": yaml_stub}):
            from plugins import install as plugin_installer

            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_root = Path(tmp_dir)
                files = (
                    "plugins/uvk5d_ptt/install.py",
                    "web_ui/templates/index.html",
                    "web_ui/static/js/index.js",
                    "web_ui/routes/config.py",
                    "config/normalize.py",
                    "config/template.py",
                )
                for relative_path in files:
                    source = repo_root / relative_path
                    destination = temp_root / relative_path
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    copy2(source, destination)

                config_path = temp_root / "plugins/config.yaml"
                config_path.parent.mkdir(parents=True, exist_ok=True)
                config_path.write_text("enabled: []\nplugins: {}\n", encoding="utf-8")

                with (
                    patch.object(plugin_installer, "REPO_ROOT", temp_root),
                    patch.object(
                        plugin_installer, "PLUGINS_DIR", temp_root / "plugins"
                    ),
                    patch.object(plugin_installer, "PLUGIN_CONFIG", config_path),
                ):
                    plugin_installer.enable_plugin("uvk5d_ptt")

                template = (temp_root / "web_ui/templates/index.html").read_text(
                    encoding="utf-8"
                )
                self.assertIn('id="ptt-mode-primary"', template)
                self.assertIn('id="ptt-mode-secondary"', template)
                self.assertEqual(template.count('option value="UVK5D"'), 2)
                self.assertIn("- uvk5d_ptt", config_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
