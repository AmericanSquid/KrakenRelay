"""Small state transformations shared by core signal processing."""


def reset_carrier_probe(gate) -> None:
    gate._carrier_valid = False
    gate._carrier_probe_start = None
    gate._carrier_last_level_db = None


def is_squelch_open_edge(squelch_open_now: bool, was_open: bool) -> bool:
    return squelch_open_now and not was_open


def is_squelch_close_edge(squelch_open_now: bool, was_open: bool) -> bool:
    return not squelch_open_now and was_open
