from .admin import admin_bp
from .config import config_bp
from .help import help_bp
from .id import id_bp

# KR_PLUGIN_ROUTES_IMPORT_END
# lifecycle
from .lifecycle.start import start_bp
from .lifecycle.stop import stop_bp
from .lock import lock_bp
from .logs import logs_bp
from .maintenance import maintenance_bp
from .meter import meter_bp

# KR_PLUGIN_ROUTES_IMPORT_START
from .plugins import plugin_bp
from .root import root_bp
from .stats import stats_bp
from .status import status_bp

ALL_BLUEPRINTS = [
    status_bp,
    meter_bp,
    stats_bp,
    config_bp,
    logs_bp,
    maintenance_bp,
    root_bp,
    help_bp,
    lock_bp,
    start_bp,
    stop_bp,
    id_bp,
    admin_bp,
    # KR_PLUGIN_ROUTES_LIST_START
    plugin_bp,
    # KR_PLUGIN_ROUTES_LIST_END
]
