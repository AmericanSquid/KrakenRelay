def _cfg(self):
    cfg = self.config.config
    return cfg, cfg['audio'], cfg['repeater']
    
def _now():
    return time.time()
    
def _chunk_size(self):
    cfg = self.config.config
    audio_cfg = cfg['audio']
    return audio_cfg['chunk_size']
    
def _sample_rate(self):
    cfg = self.config.config
    audio_cfg = cfg['audio']
    return audio_cfg['sample_rate']
    
def clear_buffer(state):
    state.kerchunk_buffer = []

def _shutdown_cleanup(controller, tx):
    try:
        if getattr(tx, "transmitting", False):
            try:
                controller.tx_control.stop()
            except Exception:
                controller.ptt_manager.safe_ptt_unkey()
        else:
            controller.ptt_manager.safe_ptt_unkey()
    except Exception:
        logging.exception("[Repeater] Error while unkeying during shutdown.")

    def _update_rms(self, samples):
        controller = self.controller
        controller.current_rms = compute_rms(samples)

    def _normalize_samples(self, samples):
        samples = ensure_float32(samples)
        samples = sanitize_audio(samples)
        return samples

    def _compute_levels(self, samples, raw_samples):
        return (
            float(calculate_db_level(samples)),
            float(calculate_db_level(raw_samples))
        )

    def _reset_carrier_probe(self):
        controller = self.controller
        gate = controller.signal_gate

        _carrier_valid = False
        _carrier_probe_start = None
        _carrier_last_level_db = None

    def is_squelch_open_edge(self, squelch_open_now, prev_open):
        return squelch_open_now and not prev_open

    def is_squelch_close_edge(self, squelch_open_now, prev_open):
        return not squelch_open_now and prev_open
