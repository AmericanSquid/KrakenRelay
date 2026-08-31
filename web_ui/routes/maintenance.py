from flask import Blueprint, jsonify

import web_ui.app as state
from runtime.audit import AuditEvent

from .common import _ok

maintenance_bp = Blueprint("maintenance", __name__)


def _set_flag(attr, response_key, value):
    old_value = getattr(state, attr, None)

    setattr(state, attr, value)

    if attr == "maintenance_mode" and old_value != value:
        if hasattr(state, "audit") and state.audit:
            event_type = (
                AuditEvent.MAINTENANCE_MODE_ENABLED
                if value
                else AuditEvent.MAINTENANCE_MODE_DISABLED
            )

            state.audit.info(
                event_type=event_type,
                source="web_ui",
                username="local_admin",
                message=(
                    "Maintenance mode enabled" if value else "Maintenance mode disabled"
                ),
                metadata={
                    "old": old_value,
                    "new": value,
                },
            )

    return _ok(**{response_key: value})


def maintenance_on():
    return _set_flag("maintenance_mode", "maintenance", True)


def maintenance_off():
    return _set_flag("maintenance_mode", "maintenance", False)


def maintenance_restart():
    return _set_flag("restarting", "restarting", True)


def restart_clear():
    return _set_flag("restarting", "restarting", False)


@maintenance_bp.route("/maintenance")
def maintenance_status():
    return jsonify(
        {"maintenance": state.maintenance_mode, "restarting": state.restarting}
    )
