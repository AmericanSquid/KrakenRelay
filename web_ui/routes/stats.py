from flask import Blueprint, jsonify
import web_ui.app as state

stats_bp = Blueprint("stats", __name__)

@stats_bp.route("/api/stats")
def api_stats():
    return jsonify(state.system_stats.get())
