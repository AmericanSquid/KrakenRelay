"""Hardware-free checks for CW mixed with blocking RX/TX frames."""

import importlib.util
import sys
import threading
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_MISSING = object()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_audio_loop():
    """Load AudioLoop without importing the optional PyAudio package."""
    names = ("audio", "audio.health", "core", "core.common", "runtime", "runtime.audit")
    original = {name: sys.modules.get(name, _MISSING) for name in names}
    audio = ModuleType("audio")
    health = ModuleType("audio.health")
    health.AudioStreamFailure = type("AudioStreamFailure", (RuntimeError,), {})
    audio.health = health
    core = ModuleType("core")
    common = ModuleType("core.common")
    common.shutdown_transmitter = lambda *_args: None
    core.common = common
    runtime = ModuleType("runtime")
    audit = ModuleType("runtime.audit")
    audit.AuditEvent = SimpleNamespace()
    runtime.audit = audit
    sys.modules.update({
        "audio": audio,
        "audio.health": health,
        "core": core,
        "core.common": common,
        "runtime": runtime,
        "runtime.audit": audit,
    })
    try:
        return load_module("core/engine/audio_loop.py", "smoke_audio_loop")
    finally:
        for name, module in original.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


CWPlayback = load_module("tones/cw_playback.py", "smoke_cw_playback").CWPlayback


def make_playback(state, ratio=100, attenuation_db=0):
    config = SimpleNamespace(config={
        "identification": {
            "cw_mix_ratio": ratio,
            "cw_mix_attenuation_db": attenuation_db,
        },
        "audio": {"sample_rate": 48000},
    })
    return CWPlayback(state, config)


def load_tx_audio():
    names = ("audio", "runtime", "runtime.audit")
    original = {name: sys.modules.get(name, _MISSING) for name in names}
    audio = ModuleType("audio")
    audio.check_clipping = lambda _samples: None
    runtime = ModuleType("runtime")
    audit = ModuleType("runtime.audit")
    audit.AuditEvent = SimpleNamespace()
    runtime.audit = audit
    sys.modules.update({"audio": audio, "runtime": runtime, "runtime.audit": audit})
    try:
        return load_module("core/transmit/audio.py", "smoke_tx_audio")
    finally:
        for name, module in original.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def load_pipeline():
    names = ("runtime", "runtime.audit", "runtime.logging_utils")
    original = {name: sys.modules.get(name, _MISSING) for name in names}
    runtime = ModuleType("runtime")
    audit = ModuleType("runtime.audit")
    audit.AuditEvent = SimpleNamespace()
    logging_utils = ModuleType("runtime.logging_utils")
    logging_utils.debug_enabled = lambda: False
    runtime.audit = audit
    runtime.logging_utils = logging_utils
    sys.modules.update({
        "runtime": runtime,
        "runtime.audit": audit,
        "runtime.logging_utils": logging_utils,
    })
    try:
        return load_module("core/transmit/pipeline.py", "smoke_tx_pipeline")
    finally:
        for name, module in original.items():
            if module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class CWPlaybackSmokeTests(unittest.TestCase):
    def test_overloaded_mix_scales_whole_waveform_without_changing_input(self):
        voice = np.array([20000, -30000, 100], dtype=np.float32)
        state = SimpleNamespace(
            cw_gen=iter([np.array([20000, -10000, 50], dtype=np.int16)]),
            cw_next_t=None,
        )
        playback = make_playback(state)

        mixed = playback.mix_voice(voice)

        np.testing.assert_allclose(mixed, np.array([40000, -40000, 150]) * (32767 / 40000), atol=0.001)
        np.testing.assert_array_equal(voice, [20000, -30000, 100])
        self.assertTrue(playback.voice_emitted)

    def test_tx_path_mixes_primary_but_keeps_link_audio_clean(self):
        tx_audio_cls = load_tx_audio().TxAudio
        state = SimpleNamespace(
            cw_gen=iter([np.array([10, 20], dtype=np.int16)]), cw_next_t=None
        )
        output = []
        tx_audio = tx_audio_cls(
            SimpleNamespace(config={"audio": {}}),
            SimpleNamespace(),
            SimpleNamespace(update=lambda *_args: None),
            lambda samples, link_pcm=None: output.append((samples, link_pcm)),
            cw_playback=make_playback(state),
        )
        link = np.array([7, 8], dtype=np.float32)

        tx_audio.send_chunk(
            np.array([100, 200], dtype=np.float32),
            link_samples=link,
            overlay_cw=True,
        )

        np.testing.assert_array_equal(output[0][0], [110, 220])
        np.testing.assert_array_equal(output[0][1], [7, 8])

    def test_tx_protects_combined_signal_after_voice_dsp(self):
        tx_audio_cls = load_tx_audio().TxAudio
        voice = np.array([32000, -32000, 100], dtype=np.int16)
        cw = np.array([4000, -4000, 50], dtype=np.int16)
        state = SimpleNamespace(cw_gen=iter([cw]), cw_next_t=None)
        output = []
        calls = []

        def process(samples):
            calls.append(samples.copy())
            return samples

        tx = tx_audio_cls(
            SimpleNamespace(config={"audio": {"limiter_enabled": True}}),
            SimpleNamespace(process_int16_to_int16=process),
            SimpleNamespace(update=lambda *_args: None),
            lambda samples, **_kwargs: output.append(samples),
            cw_playback=make_playback(state),
        )
        tx.send_chunk(voice, overlay_cw=True)
        np.testing.assert_array_equal(calls[0], voice)
        np.testing.assert_allclose(output[0], np.array([36000, -36000, 150]) * (32767 / 36000), atol=0.001)

    def test_voice_without_cw_or_overload_release_is_untouched(self):
        state = SimpleNamespace(cw_gen=None, cw_next_t=None)
        voice = np.array([-32768, 123, 32767], dtype=np.int16)
        self.assertIs(make_playback(state).mix_voice(voice), voice)

    def test_loop_mixes_live_frames_then_finishes_cw_alone(self):
        loop_module = load_audio_loop()
        cw_frames = [
            np.array([10, 20], dtype=np.int16),
            np.array([30, 40], dtype=np.int16),
            np.array([50, 60], dtype=np.int16),
        ]
        state = SimpleNamespace(cw_gen=iter(cw_frames), cw_next_t=None, running=True)
        tx = SimpleNamespace(transmitting=True, skip_courtesy_tone=False)
        playback = make_playback(state)
        output = []
        stopped = []

        class Processor:
            reads = 0

            def process_audio(self):
                self.reads += 1
                if self.reads <= 2:
                    voice = np.array([100, 200], dtype=np.float32)
                    output.append(playback.mix_voice(voice))
                return True

        processor = Processor()

        def stop_transmission():
            stopped.append(True)
            tx.transmitting = False
            state.running = False

        loop = loop_module.AudioLoop(
            SimpleNamespace(config={}), state, tx, output.append,
            stop_transmission, lambda: None, processor,
            SimpleNamespace(manual_id_event=threading.Event()),
            SimpleNamespace(send_id=lambda: None, check_and_send=lambda: None),
            SimpleNamespace(check_lockout_expired=lambda: None),
            cw_playback=playback,
        )
        loop.audio_loop()

        self.assertEqual(processor.reads, 4)
        self.assertEqual(len(output), 3)
        np.testing.assert_array_equal(output[0], [110, 220])
        np.testing.assert_array_equal(output[1], [130, 240])
        np.testing.assert_array_equal(output[2], [50, 60])
        self.assertEqual(stopped, [True])
        self.assertTrue(tx.skip_courtesy_tone)

    def test_cw_finishing_during_voice_does_not_stop_transmission(self):
        state = SimpleNamespace(
            cw_gen=iter([np.array([10], dtype=np.int16)]), cw_next_t=None
        )
        playback = make_playback(state)
        playback.begin_iteration()
        np.testing.assert_array_equal(playback.mix_voice(np.array([100])), [110])
        playback.begin_iteration()
        np.testing.assert_array_equal(playback.mix_voice(np.array([200])), [200])
        self.assertIsNone(state.cw_gen)
        self.assertTrue(playback.voice_emitted)

    def test_mix_ratio_sets_fixed_fraction_of_cw_level(self):
        voice = np.full(4, 1000, dtype=np.float32)

        half_state = SimpleNamespace(
            cw_gen=iter([np.full(4, 1000, dtype=np.int16)]), cw_next_t=None
        )
        half = make_playback(half_state, ratio=50, attenuation_db=0)
        np.testing.assert_allclose(half.mix_voice(voice), np.full(4, 1500))

        full_state = SimpleNamespace(
            cw_gen=iter([np.full(4, 1000, dtype=np.int16)]), cw_next_t=None
        )
        full = make_playback(full_state, ratio=100, attenuation_db=0)
        np.testing.assert_allclose(full.mix_voice(voice), np.full(4, 2000))

        muted_state = SimpleNamespace(
            cw_gen=iter([np.full(4, 1000, dtype=np.int16)]), cw_next_t=None
        )
        muted = make_playback(muted_state, ratio=0, attenuation_db=0)
        np.testing.assert_array_equal(muted.mix_voice(voice), voice)

    def test_attenuation_reduces_fixed_cw_mix_gain(self):
        state = SimpleNamespace(
            cw_gen=iter([np.full(4, 1000, dtype=np.int16)]), cw_next_t=None
        )
        playback = make_playback(state, ratio=100, attenuation_db=6)

        mixed = playback.mix_voice(np.full(4, 10000, dtype=np.float32))

        self.assertAlmostEqual(float(mixed[0] - 10000), 1000 * 10 ** (-6 / 20), delta=1)

    def test_cw_mix_gain_stays_fixed_when_voice_and_cw_chunks_vary(self):
        state = SimpleNamespace(cw_gen=iter([
            np.full(4, 1000, dtype=np.int16),
            np.full(4, 1000, dtype=np.int16),
            np.full(4, 200, dtype=np.int16),
        ]), cw_next_t=None)
        playback = make_playback(state, ratio=50, attenuation_db=6)
        gain = 0.5 * 10 ** (-6 / 20)
        for voice_level, cw_level in [(100, 1000), (10000, 1000), (0, 200)]:
            voice = np.full(4, voice_level, dtype=np.float32)
            mixed = playback.mix_voice(voice)
            np.testing.assert_allclose(mixed - voice, cw_level * gain, atol=0.001)

    def test_standalone_cw_is_not_scaled_by_mix_controls(self):
        original = np.array([100, -100, 0], dtype=np.int16)
        state = SimpleNamespace(cw_gen=iter([original]), cw_next_t=None)
        playback = make_playback(state, ratio=0, attenuation_db=30)

        np.testing.assert_array_equal(playback.take(), original)

    def test_live_mix_ratio_change_ramps_then_settles(self):
        state = SimpleNamespace(cw_gen=iter([
            np.full(2400, 1000, dtype=np.int16) for _ in range(3)
        ]), cw_next_t=None)
        playback = make_playback(state, ratio=50, attenuation_db=0)
        voice = np.full(2400, 1000, dtype=np.float32)
        np.testing.assert_allclose(playback.mix_voice(voice), 1500)
        playback.config.config["identification"]["cw_mix_ratio"] = 25
        ramped = playback.mix_voice(voice)
        self.assertGreater(float(ramped[0]), 1499)
        self.assertAlmostEqual(float(ramped[-1]), 1250, places=3)
        np.testing.assert_allclose(playback.mix_voice(voice), 1250)

    def test_cw_ramps_between_mixed_and_standalone_across_chunks(self):
        state = SimpleNamespace(cw_gen=iter([
            np.full(800, 1000, dtype=np.int16) for _ in range(9)
        ]), cw_next_t=None)
        playback = make_playback(state, ratio=50, attenuation_db=0)
        voice = np.zeros(800, dtype=np.float32)
        np.testing.assert_allclose(playback.mix_voice(voice), 500)
        rising = np.concatenate([playback.take() for _ in range(3)])
        self.assertLess(abs(float(rising[0]) - 500), 1)
        self.assertAlmostEqual(float(rising[-1]), 1000, places=3)
        self.assertTrue(np.all(np.diff(rising) >= 0))
        np.testing.assert_allclose(playback.take(), 1000)
        falling = np.concatenate([playback.mix_voice(voice) for _ in range(3)])
        self.assertLess(abs(float(falling[0]) - 1000), 1)
        self.assertAlmostEqual(float(falling[-1]), 500, places=3)
        self.assertTrue(np.all(np.diff(falling) <= 0))
        np.testing.assert_allclose(playback.mix_voice(voice), 500)

    def test_overload_protection_recovers_smoothly_after_cw_ends(self):
        state = SimpleNamespace(cw_gen=iter([
            np.full(1024, 4000, dtype=np.int16)
        ]), cw_next_t=None)
        playback = make_playback(state)
        overloaded = playback.mix_voice(np.full(1024, 32000, dtype=np.float32))
        self.assertLessEqual(float(np.max(np.abs(overloaded))), 32767)
        previous = 1000 * 32767 / 36000
        for _ in range(10):
            recovered = playback.mix_voice(np.full(1024, 1000, dtype=np.float32))
            self.assertGreater(float(recovered[0]), previous)
            self.assertLess(float(recovered[0]), 1000)
            previous = float(recovered[0])

    def test_due_id_starts_during_active_transmission_once(self):
        schedule_cls = load_module(
            "tones/timing/schedule_id.py", "smoke_schedule_id"
        ).ScheduleID
        state = SimpleNamespace(cw_active=False)
        started = []

        def start_cw_id(callsign):
            started.append(callsign)
            state.cw_active = True

        schedule = schedule_cls(
            SimpleNamespace(config={"identification": {
                "cw_enabled": True, "interval_minutes": 1, "callsign": "TEST"
            }}),
            start_cw_id,
            is_transmitting=lambda: True,
            is_cw_active=lambda: state.cw_active,
        )
        schedule.last_id_time = 0
        schedule.check_and_send()
        schedule.check_and_send()
        self.assertEqual(started, ["TEST"])

    def test_scheduler_accepts_interval_minutes_as_string_after_cw_finishes(self):
        schedule_cls = load_module(
            "tones/timing/schedule_id.py", "smoke_schedule_id_string_interval"
        ).ScheduleID
        state = SimpleNamespace(cw_active=True)
        started = []
        schedule = schedule_cls(
            SimpleNamespace(config={"identification": {
                "cw_enabled": True, "interval_minutes": "1", "callsign": "TEST"
            }}),
            lambda callsign: started.append(callsign),
            is_transmitting=lambda: False,
            is_cw_active=lambda: state.cw_active,
        )
        schedule.last_id_time = 0
        schedule.last_stop_time = 0

        schedule.check_and_send()  # Active CW defers scheduling.
        state.cw_active = False  # CW completed; the next loop checks the interval.
        schedule.check_and_send()

        self.assertEqual(started, ["TEST"])

    def test_tail_expiry_accepts_numeric_strings(self):
        tail_expired = load_module(
            "core/primitives.py", "smoke_tail_expiry"
        ).tail_expired

        self.assertFalse(tail_expired(11.0, 10.0, "2.0"))
        self.assertTrue(tail_expired(12.1, 10.0, "2.0"))

    def test_id_during_kerchunk_holdoff_keeps_buffer_until_gate_passes(self):
        pipeline_cls = load_pipeline().Pipeline
        gate = SimpleNamespace(
            squelch_open=True,
            squelch_open_time=time.time(),
            kerchunk_buffer=[np.array([1], dtype=np.float32)],
        )
        tx = SimpleNamespace(transmitting=True, last_audio_time=0)
        sent = []
        pipeline = pipeline_cls(
            SimpleNamespace(config={"repeater": {"anti_kerchunk_time": 1}}),
            gate,
            tx,
            lambda chunk, **_kwargs: sent.append(chunk.copy()),
            SimpleNamespace(check_timeout=lambda _transmitting: False),
            lambda: None,
            lambda: None,
            lambda: None,
        )

        pipeline.feed(np.array([2], dtype=np.float32))
        self.assertEqual(sent, [])
        self.assertEqual(len(gate.kerchunk_buffer), 2)

        gate.squelch_open_time -= 2
        pipeline.feed(np.array([3], dtype=np.float32))
        self.assertEqual([int(chunk[0]) for chunk in sent], [1, 2, 3])
        self.assertEqual(gate.kerchunk_buffer, [])


if __name__ == "__main__":
    unittest.main()
