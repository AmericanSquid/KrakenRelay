"""
Install hook for KrakenRelay's generic plugin installer.

This enables the UVK5D PTT plugin and patches the settings UI/config save path
so the web UI knows UVK5D is a valid PTT mode.

The installer is intentionally idempotent and repair-oriented: if an older
installer run inserted literal regex replacement text like \n\1 or escaped
Jinja quotes into index.html, this installer cleans that up before adding the
correct option lines.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

PLUGIN_NAME = "uvk5d_ptt"
BACKUP_SUFFIX = ".uvk5d_ptt.bak"


PRIMARY_UVK5D_OPTION = """<option value="UVK5D" {% if primary_mode in ['UVK5D', 'UV-K5D', 'UV_K5D'] %}selected{% endif %}>UVK5D</option>"""
SECONDARY_UVK5D_OPTION = """<option value="UVK5D" {% if secondary_mode in ['UVK5D', 'UV-K5D', 'UV_K5D'] %}selected{% endif %}>UVK5D</option>"""


PRIMARY_UVK5D_BLOCK = """

                  <div class="row" id="primary-uvk5d-wrap">
                    <div class="field">
                      <label for="ptt-uvk5d-host-primary">Primary UVK5D Host</label>
                      <input type="text" id="ptt-uvk5d-host-primary" data-key="ptt.primary.uvk5d_host"
                             value="{{ ptt_primary.get('uvk5d_host', ptt.get('uvk5d_host', '127.0.0.1')) }}" />
                    </div>

                    <div class="field">
                      <label for="ptt-uvk5d-port-primary">Primary UVK5D Port</label>
                      <input type="number" id="ptt-uvk5d-port-primary" data-key="ptt.primary.uvk5d_port" min="1" max="65535"
                             value="{{ ptt_primary.get('uvk5d_port', ptt.get('uvk5d_port', 7355)) }}" />
                    </div>

                    <div class="field">
                      <label for="ptt-uvk5d-timeout-primary">Primary UVK5D Timeout</label>
                      <input type="number" id="ptt-uvk5d-timeout-primary" data-key="ptt.primary.uvk5d_timeout" min="0.1" max="10" step="0.1"
                             value="{{ ptt_primary.get('uvk5d_timeout', ptt.get('uvk5d_timeout', 0.5)) }}" />
                    </div>
                  </div>"""


SECONDARY_UVK5D_BLOCK = """

                  <div class="row" id="secondary-uvk5d-wrap" {% if not dual_ptt %}hidden{% endif %}>
                    <div class="field">
                      <label for="ptt-uvk5d-host-secondary">Secondary UVK5D Host</label>
                      <input type="text" id="ptt-uvk5d-host-secondary" data-key="ptt.secondary.uvk5d_host"
                             value="{{ ptt_secondary.get('uvk5d_host', '127.0.0.1') }}" />
                    </div>

                    <div class="field">
                      <label for="ptt-uvk5d-port-secondary">Secondary UVK5D Port</label>
                      <input type="number" id="ptt-uvk5d-port-secondary" data-key="ptt.secondary.uvk5d_port" min="1" max="65535"
                             value="{{ ptt_secondary.get('uvk5d_port', 7355) }}" />
                    </div>

                    <div class="field">
                      <label for="ptt-uvk5d-timeout-secondary">Secondary UVK5D Timeout</label>
                      <input type="number" id="ptt-uvk5d-timeout-secondary" data-key="ptt.secondary.uvk5d_timeout" min="0.1" max="10" step="0.1"
                             value="{{ ptt_secondary.get('uvk5d_timeout', 0.5) }}" />
                    </div>
                  </div>"""


JS_ELEMENT_HANDLES = """
const pttUvk5dHostPrimaryEl = document.getElementById("ptt-uvk5d-host-primary");
const pttUvk5dPortPrimaryEl = document.getElementById("ptt-uvk5d-port-primary");
const pttUvk5dTimeoutPrimaryEl = document.getElementById("ptt-uvk5d-timeout-primary");
const pttUvk5dHostSecondaryEl = document.getElementById("ptt-uvk5d-host-secondary");
const pttUvk5dPortSecondaryEl = document.getElementById("ptt-uvk5d-port-secondary");
const pttUvk5dTimeoutSecondaryEl = document.getElementById("ptt-uvk5d-timeout-secondary");
const primaryUvk5dWrapEl = document.getElementById("primary-uvk5d-wrap");
const secondaryUvk5dWrapEl = document.getElementById("secondary-uvk5d-wrap");"""


JS_OLD_CONDITIONALS = """  const primaryCm108 = String(pttModePrimaryEl?.value || "").toUpperCase() === "CM108";
  if (pttDevicePrimaryEl) pttDevicePrimaryEl.disabled = locked || !advancedVisible || !primaryCm108;
  if (pttPinPrimaryEl) pttPinPrimaryEl.disabled = locked || !advancedVisible || !primaryCm108;

  const secondaryCm108 = String(pttModeSecondaryEl?.value || "").toUpperCase() === "CM108";
  if (pttDeviceSecondaryEl) pttDeviceSecondaryEl.disabled = locked || !advancedVisible || !dualPtt || !secondaryCm108;
  if (pttPinSecondaryEl) pttPinSecondaryEl.disabled = locked || !advancedVisible || !dualPtt || !secondaryCm108;"""


JS_NEW_CONDITIONALS = """  const primaryMode = String(pttModePrimaryEl?.value || "").toUpperCase();
  const primaryCm108 = primaryMode === "CM108";
  const primaryUvk5d = primaryMode === "UVK5D";

  if (primaryUvk5dWrapEl) primaryUvk5dWrapEl.hidden = !advancedVisible || !primaryUvk5d;
  if (pttDevicePrimaryEl) pttDevicePrimaryEl.disabled = locked || !advancedVisible || !primaryCm108;
  if (pttPinPrimaryEl) pttPinPrimaryEl.disabled = locked || !advancedVisible || !primaryCm108;
  [pttUvk5dHostPrimaryEl, pttUvk5dPortPrimaryEl, pttUvk5dTimeoutPrimaryEl].forEach(el => {
    if (el) el.disabled = locked || !advancedVisible || !primaryUvk5d;
  });

  const secondaryMode = String(pttModeSecondaryEl?.value || "").toUpperCase();
  const secondaryCm108 = secondaryMode === "CM108";
  const secondaryUvk5d = secondaryMode === "UVK5D";

  if (secondaryUvk5dWrapEl) secondaryUvk5dWrapEl.hidden = !advancedVisible || !dualPtt || !secondaryUvk5d;
  if (pttDeviceSecondaryEl) pttDeviceSecondaryEl.disabled = locked || !advancedVisible || !dualPtt || !secondaryCm108;
  if (pttPinSecondaryEl) pttPinSecondaryEl.disabled = locked || !advancedVisible || !dualPtt || !secondaryCm108;
  [pttUvk5dHostSecondaryEl, pttUvk5dPortSecondaryEl, pttUvk5dTimeoutSecondaryEl].forEach(el => {
    if (el) el.disabled = locked || !advancedVisible || !dualPtt || !secondaryUvk5d;
  });"""


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
    """Best-effort repo root detection for direct or generic-installer use."""

    candidates = []

    for key in ("repo_root", "root", "project_root", "base_dir", "plugins_dir"):
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
        if path.name == "plugins" and path.is_dir():
            return path.parent
        if (path / "plugins").is_dir():
            return path

    return here.parents[2]


def _load_plugin_config(config_path: Path) -> Dict[str, Any]:
    yaml = _import_yaml()
    if not config_path.exists():
        return {"enabled": []}

    if yaml is None:
        # Tiny fallback for the common KrakenRelay shape:
        # enabled:\n  - recording\n  - uvk5d_ptt
        enabled = []
        in_enabled = False
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not raw_line.startswith((" ", "\t")):
                in_enabled = stripped == "enabled:"
                continue
            if in_enabled and stripped.startswith("- "):
                enabled.append(stripped[2:].strip().strip("'\""))
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


def _enable_plugin(config_path: Path) -> bool:
    data = _load_plugin_config(config_path)
    enabled = data.setdefault("enabled", [])

    if PLUGIN_NAME in enabled:
        return False

    enabled.append(PLUGIN_NAME)
    _save_plugin_config(config_path, data)
    return True


def _backup_once(path: Path) -> None:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if path.exists() and not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def _write_if_changed(path: Path, text: str) -> bool:
    if not path.exists():
        print(f"UVK5D PTT plugin: missing file, skipped: {path}")
        return False

    current = path.read_text(encoding="utf-8")
    if current == text:
        return False

    _backup_once(path)
    path.write_text(text, encoding="utf-8")
    print(f"UVK5D PTT plugin: patched {path}")
    return True


def _insert_after_once(text: str, marker: str, insertion: str, label: str) -> str:
    if insertion.strip() in text:
        return text
    if marker not in text:
        print(f"UVK5D PTT plugin: marker not found for {label}; skipped")
        return text
    return text.replace(marker, marker + insertion, 1)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        print(f"UVK5D PTT plugin: marker not found for {label}; skipped")
        return text
    return text.replace(old, new, 1)


def _repair_bad_uvk5d_option_insertions(text: str) -> str:
    """Repair literal regex replacement junk from an older installer version."""

    bad_pattern = re.compile(
        r"""\\n\\1<option value="UVK5D" \{% if (primary_mode|secondary_mode) in \[\\'UVK5D\\', \\'UV-K5D\\', \\'UV_K5D\\'\] %\}selected\{% endif %\}>UVK5D</option>"""
    )

    def repl(match: re.Match[str]) -> str:
        mode_var = match.group(1)
        option = (
            PRIMARY_UVK5D_OPTION
            if mode_var == "primary_mode"
            else SECONDARY_UVK5D_OPTION
        )
        return "\n                      " + option

    return bad_pattern.sub(repl, text)


def _ensure_uvk5d_option_after_cm108(
    text: str, mode_var: str, option_line: str, label: str
) -> str:
    if option_line in text:
        return text

    pattern = re.compile(
        rf"(?m)^([ \t]*)<option value=\"CM108\" \{{% if {re.escape(mode_var)} == 'CM108' %\}}selected\{{% endif %\}}>CM108</option>"
    )
    match = pattern.search(text)
    if not match:
        print(f"UVK5D PTT plugin: marker not found for {label}; skipped")
        return text

    insert_at = match.end()
    insertion = "\n" + match.group(1) + option_line
    return text[:insert_at] + insertion + text[insert_at:]


def _insert_after_regex_match_once(
    text: str, pattern: str, insertion: str, token: str, label: str
) -> str:
    if token in text:
        return text

    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        print(f"UVK5D PTT plugin: marker not found for {label}; skipped")
        return text

    return text[: match.end()] + insertion + text[match.end() :]


def _patch_index_html(repo_root: Path) -> None:
    path = repo_root / "web_ui" / "templates" / "index.html"
    if not path.exists():
        print(f"UVK5D PTT plugin: UI template not found: {path}")
        return

    text = path.read_text(encoding="utf-8")

    # First repair damage from the older installer, then make the normal
    # idempotent insertions. No regex replacement syntax is inserted literally.
    text = _repair_bad_uvk5d_option_insertions(text)

    text = _ensure_uvk5d_option_after_cm108(
        text,
        "primary_mode",
        PRIMARY_UVK5D_OPTION,
        "primary UVK5D dropdown option",
    )
    text = _ensure_uvk5d_option_after_cm108(
        text,
        "secondary_mode",
        SECONDARY_UVK5D_OPTION,
        "secondary UVK5D dropdown option",
    )

    text = _insert_after_regex_match_once(
        text,
        r'(?s)([ \t]*<div class="row">\s*<div class="field">\s*<label for="ptt-device-primary".*?</div>\s*<div class="field">\s*<label for="ptt-pin-primary".*?</div>\s*</div>)',
        PRIMARY_UVK5D_BLOCK,
        'id="primary-uvk5d-wrap"',
        "primary UVK5D settings fields",
    )
    text = _insert_after_regex_match_once(
        text,
        r'(?s)([ \t]*<div class="row" id="secondary-ptt-advanced-wrap"[^>]*>\s*<div class="field">\s*<label for="ptt-device-secondary".*?</div>\s*<div class="field">\s*<label for="ptt-pin-secondary".*?</div>\s*</div>)',
        SECONDARY_UVK5D_BLOCK,
        'id="secondary-uvk5d-wrap"',
        "secondary UVK5D settings fields",
    )

    _write_if_changed(path, text)


def _patch_index_js(repo_root: Path) -> None:
    path = repo_root / "web_ui" / "static" / "js" / "index.js"
    if not path.exists():
        print(f"UVK5D PTT plugin: UI javascript not found: {path}")
        return

    text = path.read_text(encoding="utf-8")

    marker = """const pttDeviceSecondaryEl = document.getElementById("ptt-device-secondary");
const pttPinSecondaryEl = document.getElementById("ptt-pin-secondary");"""
    text = _insert_after_once(
        text, marker, JS_ELEMENT_HANDLES, "UVK5D JS element handles"
    )
    text = _replace_once(
        text, JS_OLD_CONDITIONALS, JS_NEW_CONDITIONALS, "UVK5D JS conditional display"
    )

    _write_if_changed(path, text)


def _patch_config_route(repo_root: Path) -> None:
    path = repo_root / "web_ui" / "routes" / "config.py"
    if not path.exists():
        print(f"UVK5D PTT plugin: config route not found: {path}")
        return

    text = path.read_text(encoding="utf-8")

    text = _insert_after_once(
        text,
        '    ptt_cfg["gpio_pin"] = primary_cfg.get("gpio_pin", ptt_cfg.get("gpio_pin", 3))',
        '\n    ptt_cfg["uvk5d_host"] = primary_cfg.get("uvk5d_host", ptt_cfg.get("uvk5d_host", "127.0.0.1"))\n    ptt_cfg["uvk5d_port"] = primary_cfg.get("uvk5d_port", ptt_cfg.get("uvk5d_port", 7355))\n    ptt_cfg["uvk5d_timeout"] = primary_cfg.get("uvk5d_timeout", ptt_cfg.get("uvk5d_timeout", 0.5))',
        "UVK5D legacy PTT config sync",
    )

    text = _insert_after_once(
        text,
        '        "ptt.gpio_pin",',
        '\n        "ptt.uvk5d_host",\n        "ptt.uvk5d_port",\n        "ptt.uvk5d_timeout",',
        "UVK5D root PTT key routing",
    )

    _write_if_changed(path, text)


def _patch_config_normalize(repo_root: Path) -> None:
    path = repo_root / "config" / "normalize.py"
    if not path.exists():
        print(f"UVK5D PTT plugin: config normalizer not found: {path}")
        return

    text = path.read_text(encoding="utf-8")

    text = _insert_after_once(
        text,
        "    flat_pin = ptt.get('gpio_pin', primary_defaults['gpio_pin'])",
        "\n    flat_uvk5d_host = ptt.get('uvk5d_host', primary_defaults.get('uvk5d_host', '127.0.0.1'))\n    flat_uvk5d_port = ptt.get('uvk5d_port', primary_defaults.get('uvk5d_port', 7355))\n    flat_uvk5d_timeout = ptt.get('uvk5d_timeout', primary_defaults.get('uvk5d_timeout', 0.5))",
        "UVK5D flat PTT defaults",
    )
    text = _insert_after_once(
        text,
        "    primary.setdefault('gpio_pin', flat_pin)",
        "\n    primary.setdefault('uvk5d_host', flat_uvk5d_host)\n    primary.setdefault('uvk5d_port', flat_uvk5d_port)\n    primary.setdefault('uvk5d_timeout', flat_uvk5d_timeout)",
        "UVK5D primary PTT defaults",
    )
    text = _insert_after_once(
        text,
        "    secondary.setdefault('gpio_pin', secondary_defaults['gpio_pin'])",
        "\n    secondary.setdefault('uvk5d_host', secondary_defaults.get('uvk5d_host', '127.0.0.1'))\n    secondary.setdefault('uvk5d_port', secondary_defaults.get('uvk5d_port', 7355))\n    secondary.setdefault('uvk5d_timeout', secondary_defaults.get('uvk5d_timeout', 0.5))",
        "UVK5D secondary PTT defaults",
    )
    text = _insert_after_once(
        text,
        "    ptt['gpio_pin'] = primary.get('gpio_pin', flat_pin)",
        "\n    ptt['uvk5d_host'] = primary.get('uvk5d_host', flat_uvk5d_host)\n    ptt['uvk5d_port'] = primary.get('uvk5d_port', flat_uvk5d_port)\n    ptt['uvk5d_timeout'] = primary.get('uvk5d_timeout', flat_uvk5d_timeout)",
        "UVK5D root PTT normalization",
    )

    _write_if_changed(path, text)


def _patch_template_defaults(repo_root: Path) -> None:
    path = repo_root / "config" / "template.py"
    if not path.exists():
        print(f"UVK5D PTT plugin: config template not found: {path}")
        return

    text = path.read_text(encoding="utf-8")

    if "'uvk5d_host': '127.0.0.1'" not in text:
        text = text.replace(
            "'gpio_pin': '3',",
            "'gpio_pin': '3',\n        'uvk5d_host': '127.0.0.1',\n        'uvk5d_port': 7355,\n        'uvk5d_timeout': 0.5,",
            3,
        )

    _write_if_changed(path, text)


def _patch_uvk5d_ui_support(repo_root: Path) -> None:
    print("UVK5D PTT plugin: installing UI/config support")
    _patch_index_html(repo_root)
    _patch_index_js(repo_root)
    _patch_config_route(repo_root)
    _patch_config_normalize(repo_root)
    _patch_template_defaults(repo_root)


def install(*args, **kwargs):
    repo_root = _find_repo_root(*args, **kwargs)
    config_path = (
        _as_path(kwargs.get("config_path")) or repo_root / "plugins" / "config.yaml"
    )
    changed = _enable_plugin(config_path)
    _patch_uvk5d_ui_support(repo_root)

    if changed:
        print(f"UVK5D PTT plugin: enabled {PLUGIN_NAME!r} in {config_path}")
    else:
        print(f"UVK5D PTT plugin: {PLUGIN_NAME!r} is already enabled in {config_path}")

    print(
        "Set main config.yaml ptt.mode to UVK5D and configure uvk5d_host/port/timeout."
    )
    return True


def main():
    install()


if __name__ == "__main__":
    main()
