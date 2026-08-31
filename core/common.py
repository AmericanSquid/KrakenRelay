"""Shared core orchestration helpers."""

import logging
from collections.abc import Callable


def shutdown_transmitter(
    tx_state,
    stop_transmission: Callable[[], None],
    unkey_transmitter: Callable[[], None],
) -> None:
    """Stop or unkey the transmitter during audio-loop shutdown."""
    try:
        if getattr(tx_state, "transmitting", False):
            try:
                stop_transmission()
            except Exception:
                unkey_transmitter()
        else:
            unkey_transmitter()
    except Exception:
        logging.exception("[Repeater] Error while unkeying during shutdown.")
