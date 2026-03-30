from flask import Flask, render_template, request, jsonify
import threading, time
import yaml
import math
import logging
from .stats import SystemStats
from collections import deque
from .utils.config import _set_path, _coerce, _config_path
from .utils.ui_log import init_ui_log_capture
from runtime.launch.common import cfg_bootstrap

app = Flask(__name__)

from web_ui.routes import ALL_BLUEPRINTS

for bp in ALL_BLUEPRINTS:
    app.register_blueprint(bp)

# Global objects (to be initialized in main or at startup)
#config = None
try:
    config, _ = cfg_bootstrap()
except Exception as e:
    logging.error(f"[Flask] Failed to initialize config: {e}")
    config = None

audio_manager = None
auto_start_error = None
controller = None  # This will hold the RepeaterController when started
controller_lock = threading.RLock()
config_locked = True
_config_lock = threading.Lock()
system_stats = SystemStats()
maintenance_mode = False
restarting = False

# Lock for thread-safe access to controller state (optional, for safety)
state_lock = threading.Lock()

init_ui_log_capture()
