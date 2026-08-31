import logging
import time

from runtime.audit import AuditEvent
from runtime.logging_utils import debug_enabled


class Pipeline:
    def __init__(
        self,
        config,
        signal_gate,
        tx_state,
        send_chunk,
        tot_manager,
        play_tot_tone,
        start_transmission,
        stop_transmission,
        audit=None,
    ):
        self.config = config
        self.signal_gate = signal_gate
        self.tx_state = tx_state
        self.send_chunk = send_chunk
        self.tot_manager = tot_manager
        self.play_tot_tone = play_tot_tone
        self.start_transmission = start_transmission
        self.stop_transmission = stop_transmission
        self.audit = audit

    def _handle_anti_kerchunk(self, gate, repeater_cfg, debug_on, now, samples):
        anti_s = float(repeater_cfg.get("anti_kerchunk_time", 0) or 0)

        if anti_s <= 0 or not gate.squelch_open:
            return False

        squelch_open_time = float(gate.squelch_open_time or 0.0)
        held = now - squelch_open_time

        if held < anti_s:
            remaining = anti_s - held
            gate.kerchunk_buffer.append(samples.copy())

            if len(gate.kerchunk_buffer) == 1 or len(gate.kerchunk_buffer) % 10 == 0:
                if debug_on:
                    logging.debug(
                        "[AntiKerchunk] Holdoff: held=%.3fs remaining=%.3fs buffered=%d",
                        held,
                        remaining,
                        len(gate.kerchunk_buffer),
                    )
            return True

        if debug_on:
            logging.debug(
                "[AntiKerchunk] Gate passed: held=%.3fs >= %.3fs. Starting TX. buffered=%d",
                held,
                anti_s,
                len(gate.kerchunk_buffer),
            )

        return False

    def feed(self, samples, link_samples=None):
        gate = self.signal_gate
        tx = self.tx_state
        cfg = self.config.config
        repeater_cfg = cfg.get("repeater", {})

        debug_on = debug_enabled()
        now = time.time()
        send_chunk = self.send_chunk

        if self.tot_manager.check_timeout(tx.transmitting):
            tx_start_time = getattr(self.tot_manager, "tx_start_time", None)
            tx_duration = None

            if tx_start_time is not None:
                tx_duration = now - tx_start_time

            if self.audit:
                self.audit.warning(
                    event_type=AuditEvent.TOT_TRIGGERED,
                    source="tx_pipeline",
                    message="Transmit timeout timer triggered",
                    metadata={
                        "tx_duration": round(tx_duration, 2)
                        if tx_duration is not None
                        else None,
                        "transmitting": bool(tx.transmitting),
                    },
                )

            self.play_tot_tone()
            self.stop_transmission()
            return

        if not tx.transmitting:
            if self._handle_anti_kerchunk(gate, repeater_cfg, debug_on, now, samples):
                return

            self.start_transmission()

            if gate.kerchunk_buffer:
                if debug_on:
                    logging.debug(
                        "[AntiKerchunk] Flushing %d buffered chunks",
                        len(gate.kerchunk_buffer),
                    )
                for chunk in gate.kerchunk_buffer:
                    send_chunk(chunk)
                gate.kerchunk_buffer = []

        send_chunk(samples, link_samples=link_samples)

        tx.last_audio_time = now
