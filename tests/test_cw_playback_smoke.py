"""Hardware-free smoke tests for paced CW ID playback."""

import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
LOOP_PATH = REPO_ROOT / "core" / "engine" / "audio_loop.py"
CHUNK_SIZE = 1024
SAMPLE_RATE = 48000
FRAME_SEC = CHUNK_SIZE / SAMPLE_RATE
_MISSING = object()


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeAudioIO:
    def __init__(self):
        self.frames = []

    def send_pcm(self, frame):
        self.frames.append(frame)


class BusyNetAudioIO(FakeAudioIO):
    """Simulate an output write delayed by a busy repeater host."""

    def __init__(self, clock):
        super().__init__()
        self.clock = clock

    def send_pcm(self, frame):
        super().send_pcm(frame)
        if len(self.frames) == 1:
            self.clock.now += 0.5


class FakePlugins:
    def __init__(self):
        self.ticks = 0

    def emit_tick(self):
        self.ticks += 1


def load_audio_loop():
    """Load AudioLoop without requiring the optional PyAudio package."""
    module_names = (
        "audio",
        "audio.health",
        "core",
        "core.common",
        "runtime",
        "runtime.audit",
    )
    original_modules = {name: sys.modules.get(name, _MISSING) for name in module_names}

    audio_module = ModuleType("audio")
    audio_health_module = ModuleType("audio.health")

    class AudioStreamFailure(RuntimeError):
        pass

    audio_health_module.AudioStreamFailure = AudioStreamFailure
    audio_module.health = audio_health_module
    sys.modules["audio"] = audio_module
    sys.modules["audio.health"] = audio_health_module

    core_module = ModuleType("core")
    core_common_module = ModuleType("core.common")
    core_common_module.shutdown_transmitter = lambda *_args: None
    core_module.common = core_common_module
    sys.modules["core"] = core_module
    sys.modules["core.common"] = core_common_module

    runtime_module = ModuleType("runtime")
    runtime_audit_module = ModuleType("runtime.audit")
    runtime_audit_module.AuditEvent = SimpleNamespace()
    runtime_module.audit = runtime_audit_module
    sys.modules["runtime"] = runtime_module
    sys.modules["runtime.audit"] = runtime_audit_module

    try:
        spec = importlib.util.spec_from_file_location("smoke_audio_loop", LOOP_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original_module in original_modules.items():
            if original_module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def make_loop(audio_loop_module, frames, clock):
    stopped = []
    state = SimpleNamespace(cw_gen=iter(frames), cw_next_t=None, running=True)
    tx_state = SimpleNamespace(skip_courtesy_tone=False, transmitting=True)
    audio_io = FakeAudioIO()
    config = SimpleNamespace(
        config={"audio": {"chunk_size": CHUNK_SIZE, "sample_rate": SAMPLE_RATE}}
    )

    def stop_transmission():
        stopped.append(True)
        state.running = False

    audio_loop_module.time = clock
    loop = audio_loop_module.AudioLoop(
        config,
        state,
        tx_state,
        audio_io.send_pcm,
        stop_transmission,
        lambda: None,
        SimpleNamespace(process_audio=lambda: None),
        SimpleNamespace(manual_id_event=threading.Event()),
        SimpleNamespace(send_id=lambda: None, check_and_send=lambda: None),
        SimpleNamespace(check_lockout_expired=lambda: False),
    )
    return loop, state, tx_state, audio_io, stopped


class CWPlaybackSmokeTests(unittest.TestCase):
    def test_cw_playback_is_idle_without_a_generator(self):
        audio_loop_module = load_audio_loop()
        clock = FakeClock()
        loop, state, _tx_state, audio_io, stopped = make_loop(
            audio_loop_module, [], clock
        )
        state.cw_gen = None

        self.assertFalse(loop._handle_cw_playback())
        self.assertEqual(audio_io.frames, [])
        self.assertEqual(stopped, [])

    def test_cw_playback_keeps_normal_frame_pacing(self):
        audio_loop_module = load_audio_loop()
        clock = FakeClock()
        loop, _state, tx_state, audio_io, stopped = make_loop(
            audio_loop_module, [b"first", b"last"], clock
        )

        self.assertTrue(loop._handle_cw_playback())
        self.assertTrue(loop._handle_cw_playback())
        self.assertTrue(loop._handle_cw_playback())

        self.assertEqual(audio_io.frames, [b"first", b"last"])
        self.assertEqual(
            [round(delay, 6) for delay in clock.sleeps],
            [round(FRAME_SEC, 6), round(FRAME_SEC, 6)],
        )
        self.assertEqual(stopped, [True])
        self.assertTrue(tx_state.skip_courtesy_tone)

    def test_busy_net_id_keeps_pacing_after_a_late_output_frame(self):
        audio_loop_module = load_audio_loop()
        clock = FakeClock()
        stopped = []
        state = SimpleNamespace(cw_gen=None, cw_next_t=None, running=True)
        tx_state = SimpleNamespace(skip_courtesy_tone=False, transmitting=False)
        audio_io = BusyNetAudioIO(clock)

        class TrafficProcessor:
            def __init__(self):
                self.frames = 0

            def process_audio(self):
                self.frames += 1
                # Normal RX/DSP work for an active net consumes loop time.
                clock.now += 0.02

        class ScheduleAfterTraffic:
            def __init__(self):
                self.checks = 0

            def check_and_send(self):
                self.checks += 1
                if self.checks == 6:
                    state.cw_gen = iter([b"first", b"middle", b"last"])
                    tx_state.transmitting = True

            def send_id(self):
                pass

        processor = TrafficProcessor()
        schedule = ScheduleAfterTraffic()
        plugins = FakePlugins()

        def stop_transmission():
            stopped.append(True)
            tx_state.transmitting = False
            state.running = False

        config = SimpleNamespace(
            config={"audio": {"chunk_size": CHUNK_SIZE, "sample_rate": SAMPLE_RATE}}
        )
        audio_loop_module.time = clock
        loop = audio_loop_module.AudioLoop(
            config,
            state,
            tx_state,
            audio_io.send_pcm,
            stop_transmission,
            lambda: None,
            processor,
            SimpleNamespace(manual_id_event=threading.Event()),
            schedule,
            SimpleNamespace(check_lockout_expired=lambda: False),
            plugins=plugins,
        )
        loop.audio_loop()

        self.assertEqual(audio_io.frames, [b"first", b"middle", b"last"])
        self.assertEqual(processor.frames, 6)
        self.assertEqual(schedule.checks, 6)
        self.assertEqual(
            [round(delay, 6) for delay in clock.sleeps],
            [round(FRAME_SEC, 6), round(FRAME_SEC, 6)],
        )
        self.assertEqual(stopped, [True])
        self.assertEqual(plugins.ticks, 9)


if __name__ == "__main__":
    unittest.main()
