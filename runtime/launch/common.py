import logging
from config.manager import ConfigManager
from config.normalize import _config_accepts_path, normalize_config_for_template

def cfg_bootstrap():
    config = ConfigManager("config.yaml") if _config_accepts_path() else ConfigManager()

    if not getattr(config, "config", None):
        logging.warning("Config failed to load. Falling back to default config.")

    cfg = config.config
    normalize_config_for_template(cfg)

    return config, cfg

def cleanup(controller, am):
    try:
        if controller:
            controller.lifecycle.cleanup()
    except Exception:
        logging.exception("[Shutdown] Controller cleanup failed.")

    try:
        am.cleanup()
    except Exception:
        logging.exception("[Shutdown] Audio cleanup failed.")
