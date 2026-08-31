import logging

from audio import AudioIO, Metering, Streams
from audio.health import AudioHealthMonitor
from dsp import DSPChain, configure_dsp
from dsp.modules.speex import SpeexNSStage
from dsp.python_chain import PythonDSPChain
from plugins.manager import PluginManager
from ptt import PTTManager
from tones import RequestID, ScheduleID, ToneGenerator, TonePlayer, TOTManager

from .engine import AudioLoop, ProcessAudio
from .lifecycle import Lifecycle
from .signal_gate import SignalGateState
from .state import RepeaterState
from .transmit import Control, Pipeline, TxAudio, TxState

log = logging.getLogger(__name__)


class Initialization:
    """Composition root for one repeater runtime."""

    def run(
        self,
        input_device,
        output_device,
        config,
        audio_manager,
        audit=None,
        publish_services=None,
    ):
        cfg = config.config
        audio_cfg = cfg["audio"]
        state = RepeaterState()
        plugins = PluginManager(config)
        plugins.load_enabled()

        ptt_manager = PTTManager(config)
        tot_manager = TOTManager(config, ptt_manager.safe_ptt_unkey)
        signal_gate = SignalGateState()
        tx_state = TxState(config)

        dsp_rx = DSPChain()
        dsp_tx = DSPChain()
        configure_dsp(config, dsp_rx, dsp_tx)

        py_dsp_rx = PythonDSPChain()
        if bool(audio_cfg.get("speex_ns_enabled", False)):
            py_dsp_rx.add(
                SpeexNSStage(
                    config=config,
                    frame_size=audio_cfg["chunk_size"],
                    is_transmitting=lambda: tx_state.transmitting,
                    sample_rate=audio_cfg["sample_rate"],
                    enabled=True,
                )
            )

        streams = Streams(
            audio_manager,
            config,
            audit,
            input_device=input_device,
            output_device=output_device,
        )
        audio_io = AudioIO(
            config,
            streams,
            plugins,
            audit,
            input_device=input_device,
            output_device=output_device,
        )
        meter = Metering()
        tx_audio = TxAudio(config, dsp_tx, meter, audio_io.send_pcm, audit=audit)

        control_ref = {}

        def start_transmission():
            return control_ref["service"].start()

        def stop_transmission():
            return control_ref["service"].stop()

        request_cw = RequestID(
            config,
            start_transmission=start_transmission,
            set_cw_generator=lambda generator: setattr(state, "cw_gen", generator),
        )
        schedule_id = ScheduleID(
            config=config,
            start_cw_id=request_cw.start_cw_id,
            is_transmitting=lambda: tx_state.transmitting,
        )
        tone_player = TonePlayer(
            config=config,
            send_pcm_callable=audio_io.send_pcm,
            tx_state_callable=lambda: tx_state.transmitting,
            tone_generator=ToneGenerator(config),
        )
        tx_control = Control(
            config,
            state,
            tx_state,
            ptt_manager,
            tot_manager,
            audio_io.send_pcm,
            tx_audio.send_chunk,
            tone_player.play_courtesy_tone,
            schedule_id.mark_post_tx,
            audit=audit,
        )
        control_ref["service"] = tx_control

        tx_pipeline = Pipeline(
            config,
            signal_gate,
            tx_state,
            tx_audio.send_chunk,
            tot_manager,
            tone_player.play_tot_tone,
            start_transmission,
            stop_transmission,
            audit=audit,
        )
        audio_health = AudioHealthMonitor(
            max_failures=3,
            reset_window=10.0,
            stale_timeout=5.0,
            audit=audit,
        )
        process_audio = ProcessAudio(
            config,
            dsp_rx,
            tx_pipeline,
            py_dsp_rx,
            streams,
            signal_gate,
            tx_state,
            audio_health,
            meter,
            state,
            audio_io.send_pcm,
            stop_transmission,
            audit=audit,
        )
        audio_loop = AudioLoop(
            config,
            state,
            tx_state,
            audio_io.send_pcm,
            stop_transmission,
            ptt_manager.safe_ptt_unkey,
            process_audio,
            request_cw,
            schedule_id,
            tot_manager,
            plugins=plugins,
            audit=audit,
        )
        lifecycle = Lifecycle(
            state,
            tx_state,
            signal_gate,
            audio_loop,
            process_audio,
            audio_health,
            streams,
            ptt_manager.safe_ptt_unkey,
            plugins,
            request_cw,
            schedule_id,
            audit=audit,
        )

        streams.setup()

        if publish_services:
            publish_services(
                lifecycle=lifecycle,
                repeater_state=state,
                ptt_manager=ptt_manager,
                tot_manager=tot_manager,
                signal_gate=signal_gate,
                tx_state=tx_state,
                streams=streams,
                meter=meter,
                request_cw=request_cw,
                schedule_id=schedule_id,
                plugins=plugins,
            )

        log.info("Repeater initialized")
        return lifecycle
