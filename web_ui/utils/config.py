import web_ui.app as state 

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
    cfg = state.config
    # tries common names used in ConfigManager
    return getattr(cfg, "config_path", None) or getattr(cfg, "path", None) or "config.yaml"
