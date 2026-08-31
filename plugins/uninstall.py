#!/usr/bin/env python3

import argparse
from pathlib import Path

import yaml

try:
    from .common import backup_file, read_text, write_text
except ImportError:  # Supports `python plugins/uninstall.py`.
    from common import backup_file, read_text, write_text

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGINS_DIR = REPO_ROOT / "plugins"
PLUGIN_CONFIG = PLUGINS_DIR / "config.yaml"

MARKER_PAIRS = [
    ("# KR_PLUGIN_IMPORT_START", "# KR_PLUGIN_IMPORT_END"),
    ("        # KR_PLUGIN_INIT_START", "        # KR_PLUGIN_INIT_END"),
    ("        # KR_PLUGIN_AUDIO_FRAME_START", "        # KR_PLUGIN_AUDIO_FRAME_END"),
    ("                # KR_PLUGIN_TICK_START", "                # KR_PLUGIN_TICK_END"),
    ("            # KR_PLUGIN_CW_TICK_START", "            # KR_PLUGIN_CW_TICK_END"),
    ("        # KR_PLUGIN_SHUTDOWN_START", "        # KR_PLUGIN_SHUTDOWN_END"),
    ("# KR_PLUGIN_ROUTES_IMPORT_START", "# KR_PLUGIN_ROUTES_IMPORT_END"),
    ("    # KR_PLUGIN_ROUTES_LIST_START", "    # KR_PLUGIN_ROUTES_LIST_END"),
    (
        "<!-- KR_PLUGIN_RECORDING_CARD_START -->",
        "<!-- KR_PLUGIN_RECORDING_CARD_END -->",
    ),
    ("/* KR_PLUGIN_RECORDING_CSS_START */", "/* KR_PLUGIN_RECORDING_CSS_END */"),
    ("// KR_PLUGIN_RECORDING_JS_START", "// KR_PLUGIN_RECORDING_JS_END"),
]

PATCHED_FILES = [
    "core/initialize.py",
    "audio/io.py",
    "core/engine/audio_loop.py",
    "core/lifecycle.py",
    "web_ui/routes/__init__.py",
    "web_ui/templates/index.html",
    "web_ui/static/css/index.css",
    "web_ui/static/js/index.js",
]


def remove_between_markers(text: str, start: str, end: str):
    changed = False
    while start in text and end in text:
        s = text.find(start)
        e = text.find(end, s)
        if e == -1:
            break
        e += len(end)
        if e < len(text) and text[e : e + 1] == "\n":
            e += 1
        text = text[:s] + text[e:]
        changed = True
    return text, changed


def disable_plugin(plugin_name: str, remove_config=False):
    if not PLUGIN_CONFIG.exists():
        print(f"No plugin config found: {PLUGIN_CONFIG}")
        return

    with PLUGIN_CONFIG.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    enabled = data.get("enabled", []) or []
    data["enabled"] = [name for name in enabled if name != plugin_name]

    if remove_config:
        plugins = data.get("plugins", {}) or {}
        plugins.pop(plugin_name, None)
        data["plugins"] = plugins

    with PLUGIN_CONFIG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Disabled plugin: {plugin_name}")


def uninstall_core_hooks():
    for rel in PATCHED_FILES:
        path = REPO_ROOT / rel
        if not path.exists():
            continue
        text = read_text(path)
        new_text = text
        changed_any = False
        for start, end in MARKER_PAIRS:
            new_text, changed = remove_between_markers(new_text, start, end)
            changed_any |= changed
        if changed_any:
            backup_file(path, suffix=".uninstall.bak")
            write_text(path, new_text)
            print(f"Unpatched: {rel}")
        else:
            print(f"No plugin markers found: {rel}")

    route_path = REPO_ROOT / "web_ui" / "routes" / "plugins.py"
    if route_path.exists():
        backup_file(route_path, suffix=".uninstall.bak")
        route_path.unlink()
        print("Removed: web_ui/routes/plugins.py")


def interactive_select():
    print("\nKrakenRelay Plugin Uninstaller")
    print("==============================")
    plugins = []
    if PLUGIN_CONFIG.exists():
        with PLUGIN_CONFIG.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        plugins = data.get("enabled", []) or []

    if plugins:
        print("Enabled plugins:")
        for i, name in enumerate(plugins, start=1):
            print(f"  [{i}] {name}")
        choice = (
            input("Plugins to disable (comma list, all, or Enter for none): ")
            .strip()
            .lower()
        )
    else:
        print("No enabled plugins found in plugins/config.yaml")
        choice = ""

    selected = []
    if choice == "all":
        selected = plugins
    elif choice:
        for part in choice.split(","):
            part = part.strip()
            if not part:
                continue
            try:
                idx = int(part)
            except ValueError:
                if part in plugins:
                    selected.append(part)
                else:
                    print(f"Ignoring unknown plugin: {part}")
                continue
            if 1 <= idx <= len(plugins):
                selected.append(plugins[idx - 1])
            else:
                print(f"Ignoring invalid selection: {part}")

    core_answer = (
        input("Remove core plugin hooks and UI patches too? [y/N]: ").strip().lower()
    )
    remove_core = core_answer in ("y", "yes")
    return list(dict.fromkeys(selected)), remove_core


def main():
    parser = argparse.ArgumentParser(
        description="Uninstall KrakenRelay plugins or plugin hooks"
    )
    parser.add_argument("plugins", nargs="*", help="Plugin names to disable")
    parser.add_argument(
        "--core", action="store_true", help="Remove core plugin hooks and UI patches"
    )
    parser.add_argument(
        "--remove-config",
        action="store_true",
        help="Remove plugin config entries instead of only disabling",
    )
    parser.add_argument(
        "--yes", action="store_true", help="Non-interactive mode; use CLI args only"
    )
    args = parser.parse_args()

    if not args.yes and not args.plugins and not args.core:
        plugins, remove_core = interactive_select()
    else:
        plugins = args.plugins
        remove_core = args.core

    for plugin_name in plugins:
        disable_plugin(plugin_name, remove_config=args.remove_config)

    if remove_core:
        uninstall_core_hooks()


if __name__ == "__main__":
    main()
