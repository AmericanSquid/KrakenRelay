import subprocess

from flask import Blueprint, jsonify, request

admin_bp = Blueprint("admin", __name__)

ADMIN_PIN = "438125"


def pin_page(action_label):
    return f"""
    <!doctype html>
    <html>
      <head>
        <title>{action_label}</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
      </head>

      <body style="font-family: sans-serif; padding: 2rem;">
        <h2>{action_label}</h2>

        <form method="post">
          <input
            type="password"
            name="pin"
            inputmode="numeric"
            placeholder="PIN"
            autofocus
            style="font-size: 1.2rem; padding: 0.4rem;"
          >

          <button
            type="submit"
            style="font-size: 1.2rem; padding: 0.4rem 0.7rem;"
          >
            {action_label}
          </button>
        </form>
      </body>
    </html>
    """


def require_admin_pin():
    supplied_pin = request.form.get("pin", "").strip()

    if supplied_pin != ADMIN_PIN:
        return jsonify({"ok": False, "error": "bad or missing pin"}), 403

    return None


@admin_bp.route("/hidden/restart-service", methods=["GET", "POST"])
def hidden_restart_service():
    if request.method == "GET":
        return pin_page("Restart KrakenRelay Service")

    pin_error = require_admin_pin()
    if pin_error:
        return pin_error

    try:
        subprocess.Popen(
            ["sudo", "-n", "/bin/systemctl", "restart", "krakenrelay.service"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        return jsonify({"ok": True, "message": "KrakenRelay service restart requested"})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@admin_bp.route("/hidden/reboot", methods=["GET", "POST"])
def hidden_reboot():
    if request.method == "GET":
        return pin_page("Reboot Pi")

    pin_error = require_admin_pin()
    if pin_error:
        return pin_error

    try:
        subprocess.Popen(
            ["sudo", "-n", "/sbin/reboot"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        return jsonify({"ok": True, "message": "Pi reboot requested"})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
