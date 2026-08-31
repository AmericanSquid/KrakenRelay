import logging
import threading

from flask import Flask

from runtime.launch.common import cfg_bootstrap
from web_ui.routes import ALL_BLUEPRINTS

from .stats import SystemStats
from .utils.ui_log import init_ui_log_capture

app = Flask(__name__)

for bp in ALL_BLUEPRINTS:
    app.register_blueprint(bp)

# Global objects (to be initialized in main or at startup)
# config = None
try:
    config, _ = cfg_bootstrap()
except Exception as e:
    logging.error(f"[Flask] Failed to initialize config: {e}")
    config = None

audit = None
audio_manager = None
auto_start_error = None
lifecycle = None
repeater_state = None
ptt_manager = None
tot_manager = None
signal_gate = None
tx_state = None
streams = None
meter = None
request_cw = None
schedule_id = None
plugins = None
config_locked = True
_config_lock = threading.Lock()
system_stats = SystemStats()
maintenance_mode = False
restarting = False

# Lock for thread-safe access to repeater services.
state_lock = threading.Lock()

init_ui_log_capture()
