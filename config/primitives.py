"""Pure transformations used by configuration and DSP setup."""

import math


def parsed_notch_frequencies(values) -> list[float]:
    """Read legacy YAML lists using the DSP's existing frequency selection rules."""
    if not isinstance(values, (list, tuple)):
        return []
    frequencies = []
    for value in values:
        try:
            frequency = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(frequency) and frequency > 0 and frequency not in frequencies:
            frequencies.append(frequency)
    return frequencies[:8]


def resolve_notch_mode(audio_cfg: dict) -> str:
    """Honor an explicit mode, or retain legacy list-based mode selection."""
    mode = audio_cfg.get("notch_mode")
    if mode in ("harmonics", "frequencies"):
        return mode
    return (
        "frequencies"
        if parsed_notch_frequencies(audio_cfg.get("notch_frequencies_hz", []))
        else "harmonics"
    )


def validate_notch_frequencies(values, sample_rate: float) -> list[float]:
    """Validate an explicit UI/API list without silently dropping frequencies."""
    if not isinstance(values, list):
        raise ValueError("Notch frequencies must be a numeric array")
    maximum = 0.49 * float(sample_rate)
    if not math.isfinite(maximum) or maximum < 5:
        raise ValueError("Invalid sample rate for notch frequencies")
    frequencies = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Notch frequencies must be numbers")
        frequency = float(value)
        if not math.isfinite(frequency) or not 5 <= frequency <= maximum:
            raise ValueError(f"Notch frequencies must be between 5 and {maximum:g} Hz")
        if frequency not in frequencies:
            frequencies.append(frequency)
    if len(frequencies) > 8:
        raise ValueError("Use at most eight distinct notch frequencies")
    return frequencies


def compressor_settings(percent: float) -> tuple[float, float, float]:
    """Map a user-facing compressor strength to DSP settings."""
    strength = max(0.0, min(100.0, float(percent))) / 100.0
    return (
        -15.0 - (10.0 * strength),
        1.8 + (2.4 * strength),
        2.5 + (2.5 * strength),
    )
