"""
Uninstall hook for KrakenRelay's generic plugin uninstaller.

Runtime patches are process-local. Disable the plugin in plugins/config.yaml and
restart KrakenRelay to remove UVK5D support from the running process.
"""

from pathlib import Path
from typing import Any, Dict, Optional

PLUGIN_NAME = "uvk5d_ptt"
BACKUP_SUFFIX = ".uvk5d_ptt.bak"
PATCHED_FILES = [
    "web_ui/templates/index.html",
    "web_ui/static/js/index.js",
    "web_ui/routes/config.py",
    "config/normalize.py",
    "config/template.py",
]


def _import_yaml():
    try:
        import yaml  # type: ignore

        return yaml
    except Exception:
        return None


def _as_path(value: Any) -> Optional[Path]:
    if value is None:
        return None
    try:
        return Path(value).expanduser().resolve()
    except Exception:
        return None


def _find_repo_root(*args, **kwargs) -> Path:
    candidates = []
    for key in ("repo_root", "root", "project_root", "base_dir"):
        path = _as_path(kwargs.get(key))
        if path:
            candidates.append(path)
    for arg in args:
        path = _as_path(arg)
        if path:
            candidates.append(path)
    here = Path(__file__).resolve()
    candidates.extend([Path.cwd().resolve(), here.parents[2], here.parents[1]])
    candidates.extend(Path.cwd().resolve().parents)
    for path in candidates:
        if (path / "plugins").is_dir():
            return path
        if path.name == "plugins" and path.is_dir():
            return path.parent
    return here.parents[2]


def _load_plugin_config(config_path: Path) -> Dict[str, Any]:
    yaml = _import_yaml()
    if not config_path.exists():
        return {"enabled": []}
    if yaml is None:
        enabled = []
        in_enabled = False
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not raw_line.startswith((" ", "\t")):
                in_enabled = stripped == "enabled:"
                continue
            if in_enabled and stripped.startswith("- "):
                name = stripped[2:].strip().strip("'\"")
                if name != PLUGIN_NAME:
                    enabled.append(name)
        return {"enabled": enabled}
    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        data = {}
    if not isinstance(data.get("enabled"), list):
        data["enabled"] = []
    return data


def _save_plugin_config(config_path: Path, data: Dict[str, Any]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    yaml = _import_yaml()
    if yaml is None:
        enabled = data.get("enabled") or []
        text = "enabled:\n" + "".join(f"  - {name}\n" for name in enabled)
        config_path.write_text(text, encoding="utf-8")
        return
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def _disable_plugin(config_path: Path) -> bool:
    data = _load_plugin_config(config_path)
    enabled = data.setdefault("enabled", [])
    if PLUGIN_NAME not in enabled:
        return False
    data["enabled"] = [name for name in enabled if name != PLUGIN_NAME]
    _save_plugin_config(config_path, data)
    return True


def _restore_ui_backups(repo_root: Path) -> None:
    restored = False
    for rel_path in PATCHED_FILES:
        path = repo_root / rel_path
        backup = path.with_name(path.name + BACKUP_SUFFIX)
        if backup.exists():
            path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
            backup.unlink()
            restored = True
            print(f"UVK5D PTT plugin: restored {rel_path}")
    if not restored:
        print("UVK5D PTT plugin: no UI/config backups found to restore")


def uninstall(*args, **kwargs):
    repo_root = _find_repo_root(*args, **kwargs)
    config_path = repo_root / "plugins" / "config.yaml"
    changed = _disable_plugin(config_path)

    remove_config = bool(kwargs.get("remove_config", False))
    if remove_config:
        _restore_ui_backups(repo_root)

    if changed:
        print(f"UVK5D PTT plugin: disabled {PLUGIN_NAME!r} in {config_path}")
    else:
        print(f"UVK5D PTT plugin: {PLUGIN_NAME!r} was not enabled in {config_path}")

    if not remove_config:
        print(
            "UVK5D PTT plugin: UI/config patches left in place. Re-run uninstall with remove_config=True to restore backups."
        )
    print("Restart KrakenRelay to remove process-local runtime patches.")
    return True


def main():
    uninstall()


if __name__ == "__main__":
    main()
