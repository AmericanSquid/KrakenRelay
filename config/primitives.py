"""Pure transformations used by configuration and DSP setup."""


def compressor_settings(percent: float) -> tuple[float, float, float]:
    """Map a user-facing compressor strength to DSP settings."""
    strength = max(0.0, min(100.0, float(percent))) / 100.0
    return (
        -15.0 - (10.0 * strength),
        1.8 + (2.4 * strength),
        2.5 + (2.5 * strength),
    )
