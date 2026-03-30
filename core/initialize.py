import numpy as np
import time
import threading
import logging

from tones import (
    ScheduleID, 
    CWGenerator,
    ToneGenerator,
    TonePlayer,
    TOTManager,
    RequestID
)

from ptt import PTTManager
from dsp import DSPChain, configure_dsp
from audio import Metering, Streams, AudioIO

from .signal_gate import SignalGateState
from .transmit import TxState
from .lifecycle import Lifecycle
from .transmit import Control, Pipeline, TxAudio
from .engine import AudioLoop, ProcessAudio

class Initialization:
    def __init__(self, controller):
        self.controller = controller

    def run(self, input_device, output_device, config, audio_manager):
        # Aliases
        controller = self.controller
        cfg = config.config
        audio_cfg = cfg['audio']
        id_cfg = cfg['identification']

        # Core Configuration
        controller.chunk_size = audio_cfg['chunk_size']
        controller.audio_manager = audio_manager
        controller.ptt_manager = PTTManager(config)
        controller.tot_manager = TOTManager(config, controller.ptt_manager.safe_ptt_unkey)

        # Subsystem Construction
        controller.lifecycle = Lifecycle(controller)
        controller.tx_control = Control(controller, config)
        controller.tx_pipeline = Pipeline(controller, config)
        controller.tx_audio = TxAudio(controller, config)
        
        controller.dsp_rx = DSPChain()
        controller.dsp_tx = DSPChain()

        controller.request_cw = RequestID(controller)

        configure_dsp(controller, config)
        
        controller.audio_loop = AudioLoop(controller, config)
        controller.process_audio = ProcessAudio(controller, config)
        controller.audio_io = AudioIO(
            controller,
            input_device=input_device,
            output_device=output_device
        )
        controller.signal_gate = SignalGateState()
        controller.tx_state = TxState(controller, config)

        # State Tracking & Timing
        controller.running = False

        # CW Init
        controller.morse = CWGenerator(
            wpm=id_cfg['cw_wpm'],
            frequency=id_cfg['cw_pitch'],
            sample_rate=audio_cfg['sample_rate'],
            volume=id_cfg['cw_volume'] / 100.0
        )

        # Stats
        controller.meter = Metering(controller)

        # CW IDer Setup
        controller.schedule_id = ScheduleID(
            controller,
            config=config,
            tx_start_fn=controller.tx_control.start,
            tx_stop_fn=controller.tx_control.stop,
            tx_state_fn=lambda: controller.tx_state.transmitting,
            set_skip_courtesy_fn=lambda: setattr(controller.tx_state, "skip_courtesy_tone", True)
        )

        # Tone Control Setup
        controller.tone_generator = ToneGenerator(config)
        controller.tone_player = TonePlayer(
            config=config,
            send_pcm_callable=controller.audio_io.send_pcm,
            tx_state_callable=lambda: controller.tx_state.transmitting,
            tone_generator=controller.tone_generator
        )

        controller.streams = Streams(controller, config)
        controller.streams.setup()
        logging.info("RepeaterController initialized")
