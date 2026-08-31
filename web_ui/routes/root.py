from flask import Blueprint, render_template

import web_ui.app as state
from audio.device_identity import selectable_devices

root_bp = Blueprint("root", __name__)


@root_bp.route("/")
def index():
    # Render the main page with configuration controls.
    devices = state.audio_manager.list_devices()

    input_devices = selectable_devices(devices, "input")
    output_devices = selectable_devices(devices, "output")

    return render_template(
        "index.html",
        input_devices=input_devices,
        output_devices=output_devices,
        config=state.config.config,
    )
