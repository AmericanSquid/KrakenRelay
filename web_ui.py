from flask import Flask, render_template, request, jsonify
import threading, time
import yaml
import math
import logging
from collections import deque

app = Flask(__name__)

# Global objects (to be initialized in main or at startup)
config = None
audio_manager = None
auto_start_error = None
controller = None  # This will hold the RepeaterController when started
controller_lock = threading.RLock()
config_locked = True
_config_lock = threading.Lock()

def _set_path(d: dict, path: str, value):
    keys = path.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def _coerce(raw):
    # raw may be bool from JS, or string from forms
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    s = str(raw).strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        if "." in s:
            return float(s)
        return int(s)
    except ValueError:
        return s

def _config_path():
    # tries common names used in ConfigManager
    return getattr(config, "config_path", None) or getattr(config, "path", None) or "config.yaml"

# Lock for thread-safe access to controller state (optional, for safety)
state_lock = threading.Lock()

# ----- In-memory log capture for the Web UI ("Recent Events") -----
# Keeps a small ring buffer of recent log lines so the UI can poll /logs.
_UI_LOG_MAX = 300
_ui_log_lock = threading.Lock()
_ui_log_buf = deque(maxlen=_UI_LOG_MAX)
_ui_log_seq = 0
_ui_log_ready = False

class _UILogRingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        global _ui_log_seq
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()

        with _ui_log_lock:
            _ui_log_seq += 1
            _ui_log_buf.append({
                "id": _ui_log_seq,
                "ts": time.time(),
                "level": record.levelname,
                "message": msg,
            })

def init_ui_log_capture() -> None:
    """Attach a ring-buffer log handler to the root logger (once)."""
    global _ui_log_ready
    if _ui_log_ready:
        return

    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, _UILogRingHandler):
            _ui_log_ready = True
            return

    h = _UILogRingHandler()
    h.setLevel(logging.DEBUG)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(h)
    _ui_log_ready = True

# Enable UI log capture on import (safe if called multiple times)
init_ui_log_capture()

@app.route('/')
def index():
    # Render the main page with configuration controls
    # (We'll set up the HTML template separately)
    # For initial load, pass necessary data like device lists and config values.
    devices = audio_manager.list_devices()
    input_devices = [{"index": d["index"], "name": d["name"]} for d in devices if d["maxInputChannels"] > 0]
    output_devices = [{"index": d["index"], "name": d["name"]} for d in devices if d["maxOutputChannels"] > 0]
    return render_template('index.html',
                           input_devices=input_devices,
                           output_devices=output_devices,
                           config=config.config)

@app.route("/lock", methods=["GET", "POST"])
def lock_config():
    """
    GET -> {"locked": bool}
    POST -> {"status":"ok","locked": bool}
    """
    global config_locked

    if request.method == "GET":
        with _config_lock:
            return jsonify({"locked": bool(config_locked)})

    locked = bool((request.json or {}).get("locked", True))
    with _config_lock:
        config_locked = locked
        return jsonify({"status": "ok", "locked": bool(config_locked)})

@app.route("/config/live", methods=["POST"])
def config_live():
    global config_locked
    data = request.json or {}
    key = data.get("key")
    value = data.get("value")

    if not key or not isinstance(key, str):
        return jsonify({"status": "error", "message": "Missing key"}), 400

    with _config_lock:
        if config_locked:
            return jsonify({"status": "locked"}), 423
        _set_path(config.config, key, _coerce(value))
    return jsonify({"status": "ok"})


@app.route("/config/apply", methods=["POST"])
def config_apply():
    global controller
    try:
        with _config_lock:
            path = _config_path()
            with open(path, "w") as f:
                yaml.safe_dump(config.config, f, sort_keys=False)

        if controller:
            with state_lock:
                controller.cleanup()
            audio_manager.cleanup()

            input_idx = controller.input_device
            output_idx = controller.output_device
            controller = None

            try:
                controller = RepeaterController(input_idx, output_idx, config, audio_manager)
                controller.start()

            except Exception as e:
                return jsonify({"status": "error", "message": f"Failed to restart: {e}"}), 500

        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/start', methods=['POST'])
def start_repeater():
    global controller, auto_start_error

    with state_lock:
        if controller and getattr(controller, "running", False):
            return jsonify({"status": "already_running"}), 400

         # Get selected device indices from form data (fallback to config)
        raw_in = request.form.get("input_index", None)
        raw_out = request.form.get("output_index", None)
        audio_cfg = (config.config.get("audio", {}) or {})
 
        if raw_in is None:
            raw_in = audio_cfg.get("input_index", None)
        if raw_out is None:
            raw_out = audio_cfg.get("output_index", None)

        try:
            input_idx = int(raw_in)
            output_idx = int(raw_out)
        except Exception:
            return jsonify({"status": "error", "message": "Invalid device indices"}), 400

        if input_idx < 0 or output_idx < 0:
            return jsonify({"status": "error", "message": "Invalid device indices"}), 400

        # Persist chosen devices so auto-start can work after "Start once"
        try:
           with _config_lock:
                cfg_audio = (config.config.setdefault("audio", {}) or {})
                cfg_audio["input_index"] = int(input_idx)
                cfg_audio["output_index"] = int(output_idx)

                # write immediately so reboot auto-start has the saved indices
                path = _config_path()
                with open(path, "w") as f:
                    yaml.safe_dump(config.config, f, sort_keys=False)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to save device selection: {e}"}), 500

        config.config['audio']['input_index'] = input_idx
        config.config['audio']['output_index'] = output_idx
        config.save_config()

        try:
            # Initialize and start the RepeaterController in a background thread
            controller = RepeaterController(input_idx, output_idx, config, audio_manager)
            controller.start()  # starts the audio thread
            auto_start_error = None
        except AudioDeviceError as e:
            return jsonify({"status": "error", "message": str(e)}), 500
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to start: {e}"}), 500

        return jsonify({"status": "running"})

@app.route('/stop', methods=['POST'])
def stop_repeater():
    global controller
    if not controller:
        return jsonify({"status": "not_running"})
    # Signal the controller to stop
    with state_lock:
        controller.cleanup()
    # Also close audio streams and reset controller
    audio_manager.cleanup()
    controller = None
    return jsonify({"status": "stopped"})

@app.route("/status")
def get_status():
    c = controller  # snapshot
    if c is None:
        return jsonify({"running": False, "auto_start_error": auto_start_error})

    try:
        with state_lock:
            tx = bool(getattr(c, "transmitting", False))

            # Prefer rx_db if you’ve added it; fallback to current_rms
            rx_db = getattr(c, "rx_db", None)
            if rx_db is None:
                current_level = float(getattr(c, "current_rms", 0.0) or 0.0)
                if current_level > 0:
                    current_db = 20.0 * math.log10(current_level / 32767.0)
                else:
                    current_db = -60.0
            else:
                current_db = float(rx_db)

            squelch_db = float(config.config.get("audio", {}).get("squelch_threshold", -40))
            rx = bool(current_db > squelch_db)

            # PTT status
            status_text, status_color = ("PTT: Unknown", "#ccc")
            if getattr(c, "ptt_manager", None):
                status_text, status_color = c.ptt_manager.get_ptt_status()

            # TOT
            tot_conf = config.config.get("tot", {})
            tot_enabled = bool(tot_conf.get("tot_enabled", False))
            tot_limit = float(tot_conf.get("tot_time", 0.0) or 0.0)
            tot_lockout = bool(c.tot_manager.is_locked()) if getattr(c, "tot_manager", None) else False

            tot_elapsed = 0.0
            lockout_remaining = 0.0
            if tot_enabled and getattr(c, "tot_manager", None):
                tm = c.tot_manager
                if tx and getattr(tm, "tx_start_time", None):
                    tot_elapsed = float(time.time() - tm.tx_start_time)
                if tot_lockout and getattr(tm, "lockout_start", None):
                    lockout_time = float(tot_conf.get("tot_lockout_time", 0.0) or 0.0)
                    lockout_remaining = max(0.0, lockout_time - float(time.time() - tm.lockout_start))

            # ID
            sending_id = bool(getattr(getattr(c, "schedule_id", None), "sending_id", False))
            next_id_in = None
            ident = config.config.get("identification", {})
            if bool(ident.get("cw_enabled", False)) and getattr(c, "schedule_id", None):
                interval = float(ident.get("interval_minutes", 10)) * 60.0
                last_id = float(getattr(c.schedule_id, "last_id_time", time.time()) or time.time())
                next_id_in = max(0.0, interval - float(time.time() - last_id))

            # clip/limit
            now = time.time()
            last_clip = float(getattr(c, "last_clip_time", 0.0) or 0.0)
            last_limit = float(getattr(c, "last_limit_time", 0.0) or 0.0)
            clipping = bool((now - last_clip) < 1.0) if last_clip > 0 else False
            limiting = bool((now - last_limit) < 1.0) if last_limit > 0 else False
            
            auto_start_err = getattr(c, "auto_start_error", None)

        return jsonify({
            "auto_start_error": auto_start_error,
            "running": True,
            "tx": tx,
            "rx": rx,
            "audio_db": float(round(current_db, 1)),
            "ptt_status_text": str(status_text),
            "ptt_status_color": str(status_color),
            "tot_enabled": tot_enabled,
            "tot_elapsed": float(round(tot_elapsed, 1)),
            "tot_limit": float(tot_limit),
            "tot_lockout": tot_lockout,
            "lockout_remaining": float(round(lockout_remaining, 1)),
            "sending_id": sending_id,
            "next_id_in": float(round(next_id_in, 1)) if next_id_in is not None else None,
            "clipping": clipping,
            "limiting": limiting,
        })
    except Exception as e:
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("[WebUI] /status failed: %s", e)
        return jsonify({"running": False, "auto_start_error": auto_start_error})

@app.route("/meter")
def meter():
    c = controller
    if c is None:
        return jsonify({"running": False, "rx_db": -60, "tx_db": -60, "rx_peak_db": -60, "tx_peak_db": -60,
                        "clipping": False, "limiting": False})

    lock = getattr(c, "_meter_lock", state_lock)

    try:
        with lock:
           rx_db = c.rx_db
           tx_db = c.tx_db
           rx_peak_db = c.rx_peak_db
           tx_peak_db = c.tx_peak_db

           now = time.time()
           last_clip = float(getattr(c, "last_clip_time", 0.0) or 0.0)
           last_limit = float(getattr(c, "last_limit_time", 0.0) or 0.0)

           clipping = bool((now - last_clip) < 1.0) if last_clip > 0 else False
           limiting = bool((now - last_limit) < 1.0) if last_limit > 0 else False

        return jsonify({
            "running": True,
            "rx_db": round(rx_db, 1),
            "tx_db": round(tx_db, 1),
            "rx_peak_db": round(rx_peak_db, 1),
            "tx_peak_db": round(tx_peak_db, 1),
            "clipping": clipping,
            "limiting": limiting,
        })
    
    except Exception as e:
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logging.debug("[WebUI] /meter failed: %s", e)
        return jsonify({"running": False, "rx_db": -60.0, "tx_db": -60.0, "clipping": False, "limiting": False}) 


@app.route("/logs")
def logs_tail():
    """Return recent log entries for the UI."""
    try:
        after = int(request.args.get("after", 0) or 0)
    except Exception:
        after = 0

    try:
        limit = int(request.args.get("limit", 200) or 200)
    except Exception:
        limit = 200

    # clamp
    limit = max(1, min(limit, 500))

    with _ui_log_lock:
        entries = [e for e in _ui_log_buf if int(e.get("id", 0)) > after]

    if len(entries) > limit:
        entries = entries[-limit:]

    return jsonify({"entries": entries})

@app.route("/logs/clear", methods=["POST"])
def logs_clear():
    """Clear the in-memory UI log buffer."""
    with _ui_log_lock:
        _ui_log_buf.clear()
    return jsonify({"ok": True})
@app.route("/manual_id", methods=["POST"])
def manual_id():
    global controller
    if controller is None or not getattr(controller, "running", False):
        return jsonify({"ok": False, "error": "not running"}), 400

    controller.request_manual_id()
    return jsonify({"ok": True})
