from flask import Blueprint, render_template
import web_ui.app as state

root_bp = Blueprint("root", __name__)

@root_bp.route('/')
def index():
    # Render the main page with configuration controls
    # (We'll set up the HTML template separately)
    # For initial load, pass necessary data like device lists and config values.
    devices = state.audio_manager.list_devices()
    input_devices = [{"index": d["index"], "name": d["name"]} for d in devices if d["maxInputChannels"] > 0]
    output_devices = [{"index": d["index"], "name": d["name"]} for d in devices if d["maxOutputChannels"] > 0]
    return render_template('index.html',
                           input_devices=input_devices,
                           output_devices=output_devices,
                           config=state.config.config)
