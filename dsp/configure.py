import logging

def configure_dsp(controller, config):
    # -----------------------------
    # DSP CHAIN CONFIGURATION
    # -----------------------------
    cfg = config.config
    a = cfg["audio"]

    sr = float(a.get("sample_rate", 48000))
    chunk = int(a.get("chunk_size", 1920))

    # --- NOTCH FILTER CONFIG ---
    notch_enabled = bool(a.get("notch_enabled", False))
    notch_freq = float(a.get("notch_frequency_hz", 60.0))
    notch_q = float(a.get("notch_q", 30.0))
    notch_harmonics = int(a.get("notch_harmonics", 1))

    # --- HPF CONFIG ---
    hpf_enabled = bool(a.get("highpass_enabled", True))
    hpf_cutoff = float(a.get("highpass_cutoff", 300))

    # --- LIMITER CONFIG ---
    limiter_enabled = bool(a.get("limiter_enabled", True))
    limiter_threshold = float(a.get("limiter_threshold", 0.85))

    logging.info(
        "Notch config: enabled=%s freq=%s q=%s harmonics=%s",
        notch_enabled,
        notch_freq,
        notch_q,
        notch_harmonics
    )

    # --- Macros --- #

    def apply_notch(dsp):
        dsp.configure_notch(
            enabled=notch_enabled,
            freq_hz=notch_freq,
            q=notch_q,
            harmonics=notch_harmonics,
            sample_rate=sr,
        )

    def apply_hpf(dsp, enabled, cutoff):
        dsp.configure_hpf(
            enabled=enabled,
            order=4,
            cutoff_hz=cutoff,
            sample_rate=sr,
        )

    def apply_limiter(dsp, enabled):
        dsp.configure_limiter(
            enabled=enabled,
            threshold=limiter_threshold,
            sample_rate=sr,
            chunk_len=chunk,
        )

    def compressor_macro(percent: float):
        """
        percent: 0–100 (UI-facing)
        Returns (threshold_db, ratio, makeup_db)
        Tuned for FM repeater TX audio.
        """
        p = max(0.0, min(100.0, percent))
        s = p / 100.0

        threshold_db = -15.0 - (10.0 * s)   # -15 → -25 dB
        ratio = 1.8 + (2.4 * s)             # 1.8 → 4.2
        makeup_db = 2.5 + (2.5 * s)          # +2.5 → +5.0 dB

        return threshold_db, ratio, makeup_db

    strength_pct = float(a.get("compressor_strength", 50))
    threshold_db, ratio, makeup_db = compressor_macro(strength_pct)

    compressor_enabled = bool(a.get("compressor_enabled", False))

    def apply_compressor(dsp, enabled, threshold, ratio_value, makeup):
        dsp.configure_compressor(
            enabled=compressor_enabled,
            threshold_db=threshold_db,
            ratio=ratio,
            sample_rate=sr,
            chunk_len=chunk,
            attack_ms=8.0,
            release_ms=160.0,
            makeup_db=makeup_db,
        )

    # ---------- RX DSP CHAIN ----------
    # RX: HPF only by default (compressor optional, limiter off)
    apply_hpf(controller.dsp_rx, hpf_enabled, hpf_cutoff)
    apply_notch(controller.dsp_rx)
    apply_compressor(controller.dsp_rx, False, 0.0, 1.0, 0.0)
    apply_limiter(controller.dsp_rx, False)

    # ---------- TX DSP CHAIN ----------
    # TX: HPF off (or enable if you want), compressor + limiter
    apply_hpf(controller.dsp_tx, False, 300)
    apply_notch(controller.dsp_tx)
    apply_compressor(controller.dsp_tx, compressor_enabled, threshold_db, ratio, makeup_db)
    apply_limiter(controller.dsp_tx, limiter_enabled)
