from datetime import datetime
from pathlib import Path


def _clean_prefix(prefix: str) -> str:
    prefix = (prefix or "krakenrelay").strip() or "krakenrelay"
    cleaned = []
    for ch in prefix:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        elif ch.isspace():
            cleaned.append("_")
    cleaned = "".join(cleaned).strip("_-")
    return cleaned or "krakenrelay"


def generate_recording_path(save_path: str, filename_prefix: str) -> Path:
    """
    Generate readable auto-incrementing MP3 filenames.

    Example:
        krakenrelay_2026-04-28_1412.mp3
        krakenrelay_2026-04-28_1412_1.mp3
    """
    directory = Path(save_path or "~/Documents/KrakenRelayRecordings").expanduser()
    directory.mkdir(parents=True, exist_ok=True)

    prefix = _clean_prefix(filename_prefix)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    base_name = f"{prefix}_{timestamp}"

    candidate = directory / f"{base_name}.mp3"
    counter = 1
    while candidate.exists():
        candidate = directory / f"{base_name}_{counter}.mp3"
        counter += 1

    return candidate
