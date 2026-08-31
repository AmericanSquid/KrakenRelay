#!/usr/bin/env python3

import argparse
import importlib.util
import re
from pathlib import Path

import yaml

try:
    from .common import backup_file, read_text, write_text
except ImportError:  # Supports `python plugins/install.py`.
    from common import backup_file, read_text, write_text

REPO_ROOT = Path(__file__).resolve().parents[1]
PLUGINS_DIR = REPO_ROOT / "plugins"
PLUGIN_CONFIG = PLUGINS_DIR / "config.yaml"


RECORDING_DEFAULTS = {
    "save_path": "~/Documents/KrakenRelayRecordings",
    "bitrate": "64k",
    "filename_prefix": "krakenrelay",
}


PLUGIN_IMPORT_START = "# KR_PLUGIN_IMPORT_START"
PLUGIN_IMPORT_END = "# KR_PLUGIN_IMPORT_END"

PLUGIN_INIT_START = "        # KR_PLUGIN_INIT_START"
PLUGIN_INIT_END = "        # KR_PLUGIN_INIT_END"

PLUGIN_AUDIO_START = "        # KR_PLUGIN_AUDIO_FRAME_START"
PLUGIN_AUDIO_END = "        # KR_PLUGIN_AUDIO_FRAME_END"

PLUGIN_TICK_START = "                # KR_PLUGIN_TICK_START"
PLUGIN_TICK_END = "                # KR_PLUGIN_TICK_END"

PLUGIN_CW_TICK_START = "            # KR_PLUGIN_CW_TICK_START"
PLUGIN_CW_TICK_END = "            # KR_PLUGIN_CW_TICK_END"

PLUGIN_SHUTDOWN_START = "        # KR_PLUGIN_SHUTDOWN_START"
PLUGIN_SHUTDOWN_END = "        # KR_PLUGIN_SHUTDOWN_END"

ROUTES_IMPORT_START = "# KR_PLUGIN_ROUTES_IMPORT_START"
ROUTES_IMPORT_END = "# KR_PLUGIN_ROUTES_IMPORT_END"

ROUTES_LIST_START = "    # KR_PLUGIN_ROUTES_LIST_START"
ROUTES_LIST_END = "    # KR_PLUGIN_ROUTES_LIST_END"

HTML_START = "<!-- KR_PLUGIN_RECORDING_CARD_START -->"
HTML_END = "<!-- KR_PLUGIN_RECORDING_CARD_END -->"

CSS_START = "/* KR_PLUGIN_RECORDING_CSS_START */"
CSS_END = "/* KR_PLUGIN_RECORDING_CSS_END */"

JS_START = "// KR_PLUGIN_RECORDING_JS_START"
JS_END = "// KR_PLUGIN_RECORDING_JS_END"


PLUGIN_ROUTE_FILE = """from flask import Blueprint, jsonify, request

import web_ui.app as state

plugin_bp = Blueprint("kraken_plugins", __name__)


@plugin_bp.route("/plugins/<plugin_name>/<action>", methods=["GET", "POST"])
def plugin_http_dispatch(plugin_name, action):
    if state.lifecycle is None:
        return jsonify({
            "ok": False,
            "error": "Repeater is not running.",
        }), 400

    if state.plugins is None:
        return jsonify({
            "ok": False,
            "error": "Plugin manager is not available.",
        }), 500

    result, status_code = state.plugins.dispatch_http(plugin_name, action, request)
    return jsonify(result), status_code
"""


RECORDING_HTML = """
        <!-- KR_PLUGIN_RECORDING_CARD_START -->
        <div class="card recording-card plugin-recording-card" style="grid-area:recording;" id="recording-card" aria-label="Recording controls">
          <div class="card-hd recording-card-hd">
            <div>
              <h2>Recording</h2>
              <div class="recording-subtitle">On-demand MP3 session recorder</div>
            </div>

            <button type="button" id="record-toggle-btn" class="record-toggle" disabled>
              <span id="record-icon" class="record-icon record-icon-idle" aria-hidden="true">●</span>
              <span id="record-label">Record</span>
            </button>
          </div>

          <div class="card-bd">
            <div class="recording-compact-meta">
              <div class="recording-meta-row">
                <span class="recording-meta-label">Status</span>
                <span id="record-status-text">Controller stopped</span>
              </div>

              <div class="recording-meta-row">
                <span class="recording-meta-label">Length</span>
                <span id="record-duration">00:00</span>
              </div>

              <div class="recording-meta-row">
                <span class="recording-meta-label">File</span>
                <span id="record-file">—</span>
              </div>
            </div>
          </div>
        </div>
        <!-- KR_PLUGIN_RECORDING_CARD_END -->
"""


RECORDING_CSS = r"""

/* KR_PLUGIN_RECORDING_CSS_START */
/* ====== Plugin grid placement ====== */
.grid {
  grid-template-areas:
    "left right"
    "left recording";
}

/* ====== Recording Plugin Card ====== */
.plugin-recording-card {
  margin: 0;
}

.recording-card-hd {
  align-items: center;
  gap: 12px;
}

.recording-subtitle {
  margin-top: 3px;
  color: var(--muted);
  font-size: 12px;
}

.record-toggle {
  min-width: 104px;
  padding: 8px 12px;
  border-radius: 999px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  white-space: nowrap;
}

.record-icon {
  font-size: 18px;
  line-height: 1;
}

.record-icon-idle {
  color: #ff3b3b;
  text-shadow: 0 0 12px rgba(255,59,59,.35);
}

.record-icon-active {
  color: #050505;
  text-shadow: none;
}

.recording-compact-meta {
  display: grid;
  gap: 10px;
}

.recording-meta-row {
  display: grid;
  grid-template-columns: 72px minmax(0, 1fr);
  gap: 12px;
  align-items: baseline;
  border: 1px solid rgba(255,255,255,.07);
  border-radius: 12px;
  padding: 10px;
  background: rgba(255,255,255,.025);
  min-width: 0;
}

.recording-meta-label {
  color: var(--muted);
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .35px;
}

#record-duration {
  font-variant-numeric: tabular-nums;
  font-weight: 900;
}

#record-file {
  display: block;
  word-break: break-all;
  color: var(--text);
}

@media (max-width: 920px) {
  .grid {
    grid-template-columns: 1fr;
    grid-template-areas:
      "left"
      "right"
      "recording";
  }

  .recording-card-hd {
    flex-direction: column;
    align-items: stretch;
  }

  .record-toggle {
    width: 100%;
  }
}
/* KR_PLUGIN_RECORDING_CSS_END */
"""


RECORDING_JS = r"""

// KR_PLUGIN_RECORDING_JS_START
// ====== Recording Plugin UI ======
const recordToggleBtn = document.getElementById('record-toggle-btn');
const recordIcon = document.getElementById('record-icon');
const recordLabel = document.getElementById('record-label');
const recordStatusText = document.getElementById('record-status-text');
const recordDuration = document.getElementById('record-duration');
const recordFile = document.getElementById('record-file');

let recordingState = {
  recording: false,
  file: null,
  started_at: null,
  elapsed_seconds: 0,
};

function formatRecordingDuration(seconds) {
  seconds = Math.max(0, Math.floor(seconds || 0));

  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;

  if (hrs > 0) {
    return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

function setRecordingUI(status) {
  if (!recordToggleBtn || !recordIcon || !recordLabel) return;

  recordingState = {
    recording: !!status.recording,
    file: status.file || null,
    started_at: status.started_at || null,
    elapsed_seconds: Number(status.elapsed_seconds || 0),
  };

  const isRecording = recordingState.recording;
  recordToggleBtn.disabled = !uiRunning;

  recordIcon.classList.remove('record-icon-idle', 'record-icon-active');

  if (isRecording) {
    recordIcon.textContent = '■';
    recordIcon.classList.add('record-icon-active');
    recordLabel.textContent = 'Stop';

    if (recordStatusText) recordStatusText.textContent = 'Recording';
    if (recordFile) recordFile.textContent = recordingState.file || '—';
  } else {
    recordIcon.textContent = '●';
    recordIcon.classList.add('record-icon-idle');
    recordLabel.textContent = 'Record';

    if (recordStatusText) recordStatusText.textContent = uiRunning ? 'Idle' : 'Controller stopped';
    if (recordFile) recordFile.textContent = '—';
  }

  if (recordDuration) {
    recordDuration.textContent = formatRecordingDuration(recordingState.elapsed_seconds);
  }
}

async function updateRecordingStatus() {
  if (!shouldPoll()) return;

  if (!uiRunning) {
    setRecordingUI({ recording: false, file: null, elapsed_seconds: 0 });
    return;
  }

  try {
    const res = await fetch(apiUrl('plugins/recording/status'), { cache: 'no-store' });
    const status = await res.json();

    if (!res.ok || status.ok === false) {
      setRecordingUI({ recording: false, file: null, elapsed_seconds: 0 });
      return;
    }

    setRecordingUI(status);
  } catch (_) {
    setRecordingUI({ recording: false, file: null, elapsed_seconds: 0 });
  }
}

if (recordToggleBtn) {
  recordToggleBtn.addEventListener('click', async () => {
    const action = recordingState.recording ? 'stop' : 'start';

    try {
      recordToggleBtn.disabled = true;

      const res = await fetch(apiUrl(`plugins/recording/${action}`), {
        method: 'POST',
        cache: 'no-store',
      });

      const result = await res.json();

      if (!res.ok || result.ok === false) {
        alert('Recording action failed: ' + (result.error || result.message || 'Unknown error'));
        return;
      }

      await updateRecordingStatus();
    } catch (err) {
      alert('Recording action failed: ' + err);
    } finally {
      recordToggleBtn.disabled = !uiRunning;
    }
  });
}

setInterval(updateRecordingStatus, 2000);
updateRecordingStatus();

setInterval(() => {
  if (!recordingState.recording || !recordDuration) return;

  recordingState.elapsed_seconds += 1;
  recordDuration.textContent = formatRecordingDuration(recordingState.elapsed_seconds);
}, 1000);
// KR_PLUGIN_RECORDING_JS_END
"""


def remove_block(text: str, start_marker: str, end_marker: str) -> tuple[str, bool]:
    start = text.find(start_marker)
    if start == -1:
        return text, False

    end = text.find(end_marker, start)
    if end == -1:
        return text, False

    end += len(end_marker)
    return text[:start] + text[end:], True


def remove_all_blocks(
    text: str, start_marker: str, end_marker: str
) -> tuple[str, bool]:
    changed = False
    while True:
        text2, removed = remove_block(text, start_marker, end_marker)
        if not removed:
            return text, changed
        text = text2
        changed = True


def add_after(text: str, needle: str, snippet: str, label: str) -> tuple[str, bool]:
    if snippet.strip() in text:
        return text, False

    if needle not in text:
        print(f"Could not patch {label}: marker not found")
        return text, False

    return text.replace(needle, needle + snippet, 1), True


def find_grid_close_pos(html: str) -> int:
    """
    Find the closing </div> for the first <div class="grid"> block.
    Returns the insertion position immediately before that closing </div>.
    """
    grid_match = re.search(r'<div\s+class=["\']grid["\'][^>]*>', html)
    if not grid_match:
        return -1

    pos = grid_match.end()
    depth = 1

    tag_re = re.compile(r"<(/?)div\b[^>]*>", re.IGNORECASE)

    for match in tag_re.finditer(html, pos):
        is_close = match.group(1) == "/"

        if is_close:
            depth -= 1
            if depth == 0:
                return match.start()
        else:
            depth += 1

    return -1


def cleanup_old_layout_artifacts(text: str) -> tuple[str, bool]:
    """
    Clean up earlier bad installer attempts that wrapped left/right columns.
    This only targets the exact wrappers produced by the previous plugin installer attempts.
    """
    changed = False

    old_left = (
        "        <!-- LEFT: Live control + plugins -->\n"
        '        <div class="left-column" style="grid-area:left;">\n'
        '          <div class="card live-card">'
    )
    new_left = (
        "        <!-- LEFT: Live control -->\n"
        '        <div class="card live-card" style="grid-area:left;">'
    )

    if old_left in text:
        text = text.replace(old_left, new_left, 1)
        changed = True

    # Remove a stray closing div that previous left-column patch may have inserted
    # immediately before RIGHT Settings.
    stray_left_close = "\n        </div>\n\n        <!-- RIGHT: Settings -->"
    normal_right = "\n\n        <!-- RIGHT: Settings -->"
    if stray_left_close in text:
        text = text.replace(stray_left_close, normal_right, 1)
        changed = True

    return text, changed


def load_plugin_config():
    if not PLUGIN_CONFIG.exists():
        return {"enabled": [], "plugins": {}}

    with PLUGIN_CONFIG.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    data.setdefault("enabled", [])
    data.setdefault("plugins", {})
    return data


def save_plugin_config(data):
    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)

    with PLUGIN_CONFIG.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"Updated plugin config: {PLUGIN_CONFIG}")


def discover_plugins():
    names = []

    if not PLUGINS_DIR.exists():
        return names

    for child in sorted(PLUGINS_DIR.iterdir()):
        if child.is_dir() and (child / "plugin.py").exists():
            names.append(child.name)

    return names


def run_plugin_install_hook(plugin_name: str) -> None:
    """Run an optional plugin-specific installer after enabling a plugin."""

    install_path = PLUGINS_DIR / plugin_name / "install.py"
    if not install_path.is_file():
        return

    module_name = f"krakenrelay_plugin_install_{plugin_name}"
    spec = importlib.util.spec_from_file_location(module_name, install_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load plugin install hook: {install_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    install_hook = getattr(module, "install", None)
    if not callable(install_hook):
        raise RuntimeError(
            f"Plugin install hook has no install() function: {install_path}"
        )

    install_hook(repo_root=REPO_ROOT, config_path=PLUGIN_CONFIG)


def enable_plugin(plugin_name: str):
    plugin_dir = PLUGINS_DIR / plugin_name

    if not plugin_dir.exists():
        raise SystemExit(f"Plugin folder not found: {plugin_dir}")

    data = load_plugin_config()
    data.setdefault("enabled", [])
    data.setdefault("plugins", {})

    if plugin_name not in data["enabled"]:
        data["enabled"].append(plugin_name)

    data["plugins"].setdefault(plugin_name, {})

    if plugin_name == "recording":
        for key, value in RECORDING_DEFAULTS.items():
            data["plugins"][plugin_name].setdefault(key, value)

    save_plugin_config(data)
    run_plugin_install_hook(plugin_name)
    print(f"Enabled plugin: {plugin_name}")


def patch_core_initialize():
    path = REPO_ROOT / "core" / "initialize.py"

    if not path.exists():
        print(f"Missing file: {path}")
        return

    text = read_text(path)
    changed = False

    if "from plugins.manager import PluginManager" not in text:
        snippet = (
            f"{PLUGIN_IMPORT_START}\n"
            "from plugins.manager import PluginManager\n"
            f"{PLUGIN_IMPORT_END}\n"
        )

        text, did = add_after(
            text,
            "from audio.health import AudioHealthMonitor\n",
            snippet,
            "core/initialize.py import",
        )
        changed |= did

    if "plugins = PluginManager(config)" not in text:
        snippet = (
            f"\n{PLUGIN_INIT_START}\n"
            "        plugins = PluginManager(config)\n"
            "        plugins.load_enabled()\n"
            f"{PLUGIN_INIT_END}\n"
        )

        text, did = add_after(
            text,
            "        state = RuntimeState()\n",
            snippet,
            "core/initialize.py plugin init",
        )
        changed |= did

    if changed:
        backup_file(path)
        write_text(path, text)
        print("Patched: core/initialize.py")
    else:
        print("Already patched: core/initialize.py")


def patch_audio_io():
    path = REPO_ROOT / "audio" / "io.py"

    if not path.exists():
        print(f"Missing file: {path}")
        return

    text = read_text(path)

    if "plugins.emit_audio_frame(pcm)" in text:
        print("Already patched: audio/io.py")
        return

    snippet = (
        f"\n{PLUGIN_AUDIO_START}\n"
        "        if self.plugins is not None:\n"
        "            self.plugins.emit_audio_frame(pcm)\n"
        f"{PLUGIN_AUDIO_END}\n"
    )

    text, changed = add_after(
        text,
        "        data = pcm_to_int16_bytes(pcm)\n",
        snippet,
        "audio/io.py",
    )

    if changed:
        backup_file(path)
        write_text(path, text)
        print("Patched: audio/io.py")


def patch_audio_loop():
    path = REPO_ROOT / "core" / "engine" / "audio_loop.py"

    if not path.exists():
        print(f"Missing file: {path}")
        return

    text = read_text(path)
    changed = False

    if "KR_PLUGIN_CW_TICK_START" not in text:
        snippet = (
            f"{PLUGIN_CW_TICK_START}\n"
            "            if self.plugins is not None:\n"
            "                self.plugins.emit_tick()\n"
            f"{PLUGIN_CW_TICK_END}\n"
        )

        text, did = add_after(
            text,
            "            self.send_pcm(chunk)\n",
            snippet,
            "core/engine/audio_loop.py CW tick",
        )
        changed |= did

    if "KR_PLUGIN_TICK_START" not in text:
        snippet = (
            f"\n{PLUGIN_TICK_START}\n"
            "                if self.plugins is not None:\n"
            "                    self.plugins.emit_tick()\n"
            f"{PLUGIN_TICK_END}\n"
        )

        text, did = add_after(
            text,
            "                self._handle_normal_audio(manual_id_event)\n",
            snippet,
            "core/engine/audio_loop.py normal tick",
        )
        changed |= did

    if changed:
        backup_file(path)
        write_text(path, text)
        print("Patched: core/engine/audio_loop.py")
    else:
        print("Already patched: core/engine/audio_loop.py")


def patch_lifecycle():
    path = REPO_ROOT / "core" / "lifecycle.py"

    if not path.exists():
        print(f"Missing file: {path}")
        return

    text = read_text(path)

    if "KR_PLUGIN_SHUTDOWN_START" in text:
        print("Already patched: core/lifecycle.py")
        return

    snippet = (
        f"\n{PLUGIN_SHUTDOWN_START}\n"
        "        if self.plugins is not None:\n"
        "            self.plugins.emit_shutdown()\n"
        f"{PLUGIN_SHUTDOWN_END}\n"
    )

    text, changed = add_after(
        text,
        '        ok &= _join(getattr(self, "audio_thread", None), "audio_thread", 3.0)\n',
        snippet,
        "core/lifecycle.py",
    )

    if changed:
        backup_file(path)
        write_text(path, text)
        print("Patched: core/lifecycle.py")


def install_plugin_route_file():
    routes_dir = REPO_ROOT / "web_ui" / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)

    route_path = routes_dir / "plugins.py"

    if (
        route_path.exists()
        and read_text(route_path).strip() == PLUGIN_ROUTE_FILE.strip()
    ):
        print("Already installed: web_ui/routes/plugins.py")
        return

    if route_path.exists():
        backup_file(route_path)

    write_text(route_path, PLUGIN_ROUTE_FILE)
    print("Installed: web_ui/routes/plugins.py")


def patch_routes_init():
    path = REPO_ROOT / "web_ui" / "routes" / "__init__.py"

    if not path.exists():
        print(f"Missing file: {path}")
        return

    text = read_text(path)
    changed = False

    # Remove old marker blocks from previous installer versions.
    text, removed = remove_all_blocks(text, ROUTES_IMPORT_START, ROUTES_IMPORT_END)
    changed |= removed

    text, removed = remove_all_blocks(text, ROUTES_LIST_START, ROUTES_LIST_END)
    changed |= removed

    # Remove stale duplicate imports/list entries from old failed installs.
    lines = text.splitlines()
    cleaned_lines = []
    seen_plugin_import = False

    for line in lines:
        if line.strip() == "from .plugins import plugin_bp":
            if seen_plugin_import:
                changed = True
                continue

            seen_plugin_import = True

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines) + "\n"

    # Ensure import exists exactly once.
    if "from .plugins import plugin_bp" not in text:
        snippet = (
            f"\n{ROUTES_IMPORT_START}\n"
            "from .plugins import plugin_bp\n"
            f"{ROUTES_IMPORT_END}"
        )

        if "from .admin import admin_bp" in text:
            text = text.replace(
                "from .admin import admin_bp",
                "from .admin import admin_bp" + snippet,
                1,
            )
            changed = True
        else:
            print(
                "Could not patch web_ui/routes/__init__.py import: admin import marker not found"
            )

    # Ensure list entry exists exactly once.
    if "plugin_bp," not in text:
        snippet = f"\n{ROUTES_LIST_START}\n    plugin_bp,\n{ROUTES_LIST_END}"

        if "    admin_bp," in text:
            text = text.replace("    admin_bp,", "    admin_bp," + snippet, 1)
            changed = True
        else:
            print(
                "Could not patch web_ui/routes/__init__.py blueprint list: admin_bp marker not found"
            )

    elif "KR_PLUGIN_ROUTES_LIST_START" not in text:
        # If plugin_bp exists unmarked from a previous attempt, leave it alone.
        pass

    if changed:
        backup_file(path)
        write_text(path, text)
        print("Patched: web_ui/routes/__init__.py")
    else:
        print("Already patched: web_ui/routes/__init__.py")


def patch_index_html():
    path = REPO_ROOT / "web_ui" / "templates" / "index.html"

    if not path.exists():
        print(f"Missing file: {path}")
        return

    text = read_text(path)
    changed = False

    # Remove any old recording card block so placement is always corrected.
    text, removed = remove_all_blocks(text, HTML_START, HTML_END)
    changed |= removed

    # Clean up older bad left-column/right-column attempts from prior installer versions.
    text, cleaned = cleanup_old_layout_artifacts(text)
    changed |= cleaned

    if HTML_START in text:
        print("Already patched: web_ui/templates/index.html")
        return

    insert_pos = find_grid_close_pos(text)
    if insert_pos == -1:
        print("Could not patch index.html: could not find closing </div> for .grid")
        return

    text = (
        text[:insert_pos].rstrip()
        + "\n\n"
        + RECORDING_HTML.rstrip()
        + "\n"
        + text[insert_pos:]
    )
    changed = True

    if changed:
        backup_file(path)
        write_text(path, text)
        print("Patched: web_ui/templates/index.html")


def patch_index_css():
    path = REPO_ROOT / "web_ui" / "static" / "css" / "index.css"

    if not path.exists():
        print(f"Missing file: {path}")
        return

    text = read_text(path)

    text, removed = remove_all_blocks(text, CSS_START, CSS_END)
    text = text.rstrip() + "\n" + RECORDING_CSS + "\n"

    backup_file(path)
    write_text(path, text)

    if removed:
        print("Repatched: web_ui/static/css/index.css")
    else:
        print("Patched: web_ui/static/css/index.css")


def patch_index_js():
    path = REPO_ROOT / "web_ui" / "static" / "js" / "index.js"

    if not path.exists():
        print(f"Missing file: {path}")
        return

    text = read_text(path)

    text, removed = remove_all_blocks(text, JS_START, JS_END)
    text = text.rstrip() + "\n" + RECORDING_JS + "\n"

    backup_file(path)
    write_text(path, text)

    if removed:
        print("Repatched: web_ui/static/js/index.js")
    else:
        print("Patched: web_ui/static/js/index.js")


def install_core_hooks(include_ui=True):
    patch_core_initialize()
    patch_audio_io()
    patch_audio_loop()
    patch_lifecycle()
    install_plugin_route_file()
    patch_routes_init()

    if include_ui:
        patch_index_html()
        patch_index_css()
        patch_index_js()

    print("Core plugin hooks installed.")


def interactive_select():
    available = discover_plugins()

    print("\nKrakenRelay Plugin Installer")
    print("============================")
    print("Core plugin hooks are required for plugins to run.")

    core_answer = input("Install/update core plugin hooks? [Y/n]: ").strip().lower()
    install_core = core_answer not in ("n", "no")

    if not available:
        print("No plugins found under plugins/*/plugin.py")
        return install_core, []

    print("\nAvailable plugins:")
    for i, name in enumerate(available, start=1):
        print(f"  [{i}] {name}")

    print(
        "\nEnter plugin numbers separated by commas, 'all', or press Enter for core only."
    )
    choice = input("Plugins to enable: ").strip().lower()

    selected = []

    if choice == "all":
        selected = available
    elif choice:
        for part in choice.split(","):
            part = part.strip()

            if not part:
                continue

            try:
                idx = int(part)
            except ValueError:
                if part in available:
                    selected.append(part)
                else:
                    print(f"Ignoring unknown plugin: {part}")
                continue

            if 1 <= idx <= len(available):
                selected.append(available[idx - 1])
            else:
                print(f"Ignoring invalid selection: {part}")

    return install_core, list(dict.fromkeys(selected))


def main():
    parser = argparse.ArgumentParser(
        description="Install KrakenRelay plugin hooks and enable plugins"
    )
    parser.add_argument(
        "plugins", nargs="*", help="Plugin names to enable, e.g. recording"
    )
    parser.add_argument(
        "--core-only", action="store_true", help="Only install generic plugin hooks"
    )
    parser.add_argument(
        "--no-ui", action="store_true", help="Do not patch the web UI recording card"
    )
    parser.add_argument(
        "--yes", action="store_true", help="Non-interactive mode; use CLI args only"
    )
    args = parser.parse_args()

    if not args.yes and not args.core_only and not args.plugins:
        install_core, selected = interactive_select()
    else:
        install_core = True
        selected = [] if args.core_only else args.plugins

    if install_core:
        install_core_hooks(include_ui=not args.no_ui)

    for plugin_name in selected:
        enable_plugin(plugin_name)


if __name__ == "__main__":
    main()
