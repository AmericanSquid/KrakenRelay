"""Dependency-free regression coverage for web startup and configuration routes."""

import importlib.util
import sys
import tempfile
import threading
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()


def _package(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []
    return module


def _load_module(module_name: str, relative_path: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        module_name, REPO_ROOT / relative_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def _isolated_modules(*module_names: str):
    originals = {name: sys.modules.get(name, _MISSING) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)

    try:
        yield
    finally:
        for name, original in originals.items():
            if original is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class _Blueprint:
    def __init__(self, *_args, **_kwargs):
        pass

    def route(self, *_args, **_kwargs):
        return lambda function: function


class WebRuntimeSmokeTests(unittest.TestCase):
    def test_config_live_uses_config_utility_not_app_globals(self):
        module_names = (
            "flask",
            "yaml",
            "audio",
            "audio.device_identity",
            "config",
            "config.common",
            "config.primitives",
            "runtime",
            "runtime.audit",
            "web_ui",
            "web_ui.app",
            "web_ui.routes",
            "web_ui.routes.common",
            "web_ui.routes.config",
            "web_ui.utils",
            "web_ui.utils.config",
        )

        with _isolated_modules(*module_names):
            flask = ModuleType("flask")
            flask.Blueprint = _Blueprint
            flask.request = SimpleNamespace(
                json={"key": "audio.squelch_threshold", "value": "-35"}
            )
            sys.modules["flask"] = flask

            sys.modules["yaml"] = ModuleType("yaml")

            state = ModuleType("web_ui.app")
            state.config = SimpleNamespace(config={"audio": {"squelch_threshold": -40}})
            state._config_lock = threading.RLock()
            state.config_locked = False
            state.audio_manager = None
            state.audit = None

            web_ui = _package("web_ui")
            web_ui.app = state
            sys.modules["web_ui"] = web_ui
            sys.modules["web_ui.app"] = state

            routes = _package("web_ui.routes")
            routes_common = ModuleType("web_ui.routes.common")
            routes_common._error = lambda message, code=500: (message, code)
            routes_common._locked = lambda: {"status": "locked"}
            routes_common._ok = lambda **_kwargs: {"status": "ok"}
            routes.common = routes_common
            sys.modules["web_ui.routes"] = routes
            sys.modules["web_ui.routes.common"] = routes_common

            utils = _package("web_ui.utils")
            sys.modules["web_ui.utils"] = utils
            utils.config = _load_module("web_ui.utils.config", "web_ui/utils/config.py")

            config = _package("config")
            common = ModuleType("config.common")
            common.sync_primary_ptt_legacy_keys = lambda _cfg: None
            primitives = ModuleType("config.primitives")
            primitives.compressor_settings = lambda _strength: (-20.0, 3.0, 3.75)
            config.common = common
            config.primitives = primitives
            sys.modules["config"] = config
            sys.modules["config.common"] = common
            sys.modules["config.primitives"] = primitives

            audio = _package("audio")
            device_identity = ModuleType("audio.device_identity")
            device_identity.descriptor_for_index = lambda *_args: None
            device_identity.selectable_devices = lambda *_args: []
            audio.device_identity = device_identity
            sys.modules["audio"] = audio
            sys.modules["audio.device_identity"] = device_identity

            runtime = _package("runtime")
            audit = ModuleType("runtime.audit")
            audit.AuditEvent = SimpleNamespace(CONFIG_CHANGED="config_changed")
            runtime.audit = audit
            sys.modules["runtime"] = runtime
            sys.modules["runtime.audit"] = audit

            route = _load_module("web_ui.routes.config", "web_ui/routes/config.py")

            self.assertEqual(route.config_live(), {"status": "ok"})
            self.assertEqual(state.config.config["audio"]["squelch_threshold"], -35)

    def test_start_route_saves_selected_devices_via_config_utility(self):
        module_names = (
            "flask",
            "yaml",
            "web_ui",
            "web_ui.app",
            "web_ui.routes",
            "web_ui.routes.common",
            "web_ui.routes.lifecycle",
            "web_ui.routes.lifecycle.start",
            "web_ui.utils",
            "web_ui.utils.config",
        )

        with (
            _isolated_modules(*module_names),
            tempfile.TemporaryDirectory() as temp_dir,
        ):
            flask = ModuleType("flask")
            flask.Blueprint = _Blueprint
            flask.jsonify = lambda payload: payload
            flask.request = SimpleNamespace(
                form={"input_index": "3", "output_index": "4"}
            )
            sys.modules["flask"] = flask

            yaml = ModuleType("yaml")
            yaml.safe_dump = lambda _cfg, stream, **_kwargs: stream.write("saved")
            sys.modules["yaml"] = yaml

            saved = []

            class FakeLifecycle:
                def start(self):
                    saved.append("started")

            def build_repeater(*_args, **kwargs):
                lifecycle = FakeLifecycle()
                kwargs["publish_services"](
                    lifecycle=lifecycle,
                    repeater_state=SimpleNamespace(running=False),
                )
                return lifecycle

            state = ModuleType("web_ui.app")
            state.config = SimpleNamespace(
                config={"audio": {}},
                config_path=Path(temp_dir) / "config.yaml",
                save_config=lambda: saved.append("config"),
            )
            state.lifecycle = None
            state.repeater_state = None
            state.state_lock = threading.RLock()
            state._config_lock = threading.RLock()
            state.build_repeater = build_repeater
            state.publish_services = lambda **services: [
                setattr(state, name, service) for name, service in services.items()
            ]
            state.AudioDeviceError = RuntimeError
            state.audio_manager = SimpleNamespace()
            state.audit = None
            state.auto_start_error = "stale error"

            web_ui = _package("web_ui")
            web_ui.app = state
            sys.modules["web_ui"] = web_ui
            sys.modules["web_ui.app"] = state

            routes = _package("web_ui.routes")
            routes_common = ModuleType("web_ui.routes.common")
            routes_common._error = lambda message, code=500: (message, code)
            routes.common = routes_common
            sys.modules["web_ui.routes"] = routes
            sys.modules["web_ui.routes.common"] = routes_common
            sys.modules["web_ui.routes.lifecycle"] = _package("web_ui.routes.lifecycle")

            utils = _package("web_ui.utils")
            sys.modules["web_ui.utils"] = utils
            utils.config = _load_module("web_ui.utils.config", "web_ui/utils/config.py")

            route = _load_module(
                "web_ui.routes.lifecycle.start", "web_ui/routes/lifecycle/start.py"
            )

            self.assertEqual(route.start_repeater(), {"status": "running"})
            self.assertEqual(
                state.config.config["audio"],
                {
                    "input_index": 3,
                    "output_index": 4,
                },
            )
            self.assertEqual((Path(temp_dir) / "config.yaml").read_text(), "saved")
            self.assertEqual(saved, ["config", "started"])

    def test_web_auto_start_skips_unavailable_saved_device_cleanly(self):
        module_names = (
            "audio",
            "audio.device_identity",
            "core",
            "runtime",
            "runtime.audit",
            "runtime.logging_utils",
            "runtime.launch",
            "runtime.launch.common",
            "runtime.launch.gui",
            "web_ui",
            "web_ui.app",
        )

        with _isolated_modules(*module_names):

            class AudioDeviceError(Exception):
                pass

            class AudioDeviceManager:
                def list_devices(self):
                    return []

            def resolve_saved_device(**_kwargs):
                raise ValueError("No saved input device could be resolved.")

            audio = _package("audio")
            audio.AudioDeviceError = AudioDeviceError
            audio.AudioDeviceManager = AudioDeviceManager
            device_identity = ModuleType("audio.device_identity")
            device_identity.resolve_saved_device = resolve_saved_device
            audio.device_identity = device_identity
            sys.modules["audio"] = audio
            sys.modules["audio.device_identity"] = device_identity

            core = _package("core")
            core.build_repeater = lambda *_args, **_kwargs: None
            sys.modules["core"] = core

            runtime = _package("runtime")
            audit = ModuleType("runtime.audit")

            class AuditManager:
                def start(self):
                    pass

            audit.AuditManager = AuditManager
            logging_utils = ModuleType("runtime.logging_utils")
            logging_utils.debug_enabled = lambda: False
            runtime.audit = audit
            runtime.logging_utils = logging_utils
            sys.modules["runtime"] = runtime
            sys.modules["runtime.audit"] = audit
            sys.modules["runtime.logging_utils"] = logging_utils

            launch = _package("runtime.launch")
            common = ModuleType("runtime.launch.common")
            cfg = {
                "repeater": {"auto_start": True},
                "audio": {"input_device": {"id": "unplugged"}},
            }
            common.cfg_bootstrap = lambda: (SimpleNamespace(config=cfg), cfg)
            common.cleanup = lambda *_args: None
            launch.common = common
            runtime.launch = launch
            sys.modules["runtime.launch"] = launch
            sys.modules["runtime.launch.common"] = common

            state = ModuleType("web_ui.app")
            state.app = SimpleNamespace(run=lambda **_kwargs: None)
            web_ui = _package("web_ui")
            web_ui.app = state
            sys.modules["web_ui"] = web_ui
            sys.modules["web_ui.app"] = state

            gui = _load_module("runtime.launch.gui", "runtime/launch/gui.py")
            gui.run_web(SimpleNamespace(bind="127.0.0.1", port=6973))

            self.assertEqual(
                state.auto_start_error,
                "Auto-start skipped: No saved input device could be resolved. "
                "Select currently connected devices in the web UI, then press Start.",
            )


if __name__ == "__main__":
    unittest.main()
