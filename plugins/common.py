"""Common filesystem operations for plugin maintenance scripts."""

import shutil
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def backup_file(path: Path, suffix: str = ".bak") -> None:
    backup = path.with_suffix(path.suffix + suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"Backup created: {backup}")
