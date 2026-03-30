from .status import status_bp
from .meter import meter_bp
from .stats import stats_bp
from .config import config_bp
from .logs import logs_bp
from .maintenance import maintenance_bp
from .root import root_bp
from .help import help_bp
from .lock import lock_bp
from .id import id_bp

# lifecycle
from .lifecycle.start import start_bp
from .lifecycle.stop import stop_bp

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
]
