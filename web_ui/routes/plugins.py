from flask import Blueprint, jsonify, request

import web_ui.app as state

plugin_bp = Blueprint("kraken_plugins", __name__)


@plugin_bp.route("/plugins/<plugin_name>/<action>", methods=["GET", "POST"])
def plugin_http_dispatch(plugin_name, action):
    if state.lifecycle is None:
        return jsonify(
            {
                "ok": False,
                "error": "Repeater is not running.",
            }
        ), 400

    if state.plugins is None:
        return jsonify(
            {
                "ok": False,
                "error": "Plugin manager is not available.",
            }
        ), 500

    result, status_code = state.plugins.dispatch_http(plugin_name, action, request)
    return jsonify(result), status_code
