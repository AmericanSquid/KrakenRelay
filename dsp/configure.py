import logging

from config.primitives import compressor_settings


def configure_dsp(config, dsp_rx, dsp_tx):
    cfg = config.config
    a = cfg["audio"]

    sr = float(a.get("sample_rate", 48000))
    chunk = int(a.get("chunk_size", 1920))

    notch_enabled = bool(a.get("notch_enabled", False))
    notch_freq = float(a.get("notch_frequency_hz", 60.0))
    notch_q = float(a.get("notch_q", 30.0))
    notch_harmonics = int(a.get("notch_harmonics", 1))
    notch_frequencies = a.get("notch_frequencies_hz", [])
    notch_apply_to_tx = bool(a.get("notch_apply_to_tx", True))

    if not isinstance(notch_frequencies, (list, tuple)):
        notch_frequencies = []

    parsed_frequencies = []
    for value in notch_frequencies:
        try:
            freq = float(value)
        except (TypeError, ValueError):
            logging.warning("Ignoring invalid notch frequency: %r", value)
            continue
        if freq > 0.0 and freq not in parsed_frequencies:
            parsed_frequencies.append(freq)
    notch_frequencies = parsed_frequencies[:8]

    tx_tone_enabled = bool(a.get("tx_tone_eq_enabled", True))
    tx_tone = float(a.get("tx_tone", 0.0))

    hpf_enabled = bool(a.get("highpass_enabled", True))
    hpf_cutoff = float(a.get("highpass_cutoff", 300))

    limiter_enabled = bool(a.get("limiter_enabled", True))
    limiter_threshold = float(a.get("limiter_threshold", 0.85))

    compressor_enabled = bool(a.get("compressor_enabled", False))
    compressor_strength = float(a.get("compressor_strength", 50))

    if notch_frequencies:
        logging.info(
            "Notch config: enabled=%s frequencies=%s q=%s apply_to_tx=%s",
            notch_enabled,
            notch_frequencies,
            notch_q,
            notch_apply_to_tx,
        )
    else:
        logging.info(
            "Notch config: enabled=%s freq=%s q=%s harmonics=%s apply_to_tx=%s",
            notch_enabled,
            notch_freq,
            notch_q,
            notch_harmonics,
            notch_apply_to_tx,
        )
    logging.info("TX tone config: enabled=%s tone=%.3f", tx_tone_enabled, tx_tone)

    def apply_notch(dsp, enabled):
        if notch_frequencies:
            dsp.configure_notch_frequencies(
                enabled=enabled,
                freqs_hz=notch_frequencies,
                q=notch_q,
                sample_rate=sr,
            )
        else:
            dsp.configure_notch(
                enabled=enabled,
                freq_hz=notch_freq,
                q=notch_q,
                harmonics=notch_harmonics,
                sample_rate=sr,
            )

    def apply_hpf(dsp, enabled, cutoff):
        dsp.configure_hpf(enabled=enabled, order=4, cutoff_hz=cutoff, sample_rate=sr)

    def apply_tone(dsp, enabled, tone):
        dsp.configure_tone(enabled=enabled, tone=tone, sample_rate=sr)

    def apply_limiter(dsp, enabled):
        dsp.configure_limiter(
            enabled=enabled,
            threshold=limiter_threshold,
            sample_rate=sr,
            chunk_len=chunk,
        )

    threshold_default, ratio_default, makeup_default = compressor_settings(
        compressor_strength
    )
    compressor_threshold_db = float(a.get("compressor_threshold_db", threshold_default))
    compressor_ratio = float(a.get("compressor_ratio", ratio_default))
    compressor_attack_ms = float(a.get("compressor_attack_ms", 8.0))
    compressor_release_ms = float(a.get("compressor_release_ms", 160.0))
    compressor_makeup_db = float(a.get("compressor_makeup_db", makeup_default))

    def apply_compressor(
        dsp, enabled, threshold_db, ratio_value, attack_ms, release_ms, makeup_db
    ):
        dsp.configure_compressor(
            enabled=enabled,
            threshold_db=threshold_db,
            ratio=ratio_value,
            sample_rate=sr,
            chunk_len=chunk,
            attack_ms=attack_ms,
            release_ms=release_ms,
            makeup_db=makeup_db,
        )

    apply_hpf(dsp_rx, hpf_enabled, hpf_cutoff)
    apply_notch(dsp_rx, notch_enabled)
    apply_tone(dsp_rx, False, 0.0)
    apply_compressor(dsp_rx, False, 0.0, 1.0, 8.0, 160.0, 0.0)
    apply_limiter(dsp_rx, False)

    apply_hpf(dsp_tx, False, 300)
    apply_notch(dsp_tx, notch_enabled and notch_apply_to_tx)
    apply_tone(dsp_tx, tx_tone_enabled, tx_tone)
    apply_compressor(
        dsp_tx,
        compressor_enabled,
        compressor_threshold_db,
        compressor_ratio,
        compressor_attack_ms,
        compressor_release_ms,
        compressor_makeup_db,
    )
    apply_limiter(dsp_tx, limiter_enabled)
