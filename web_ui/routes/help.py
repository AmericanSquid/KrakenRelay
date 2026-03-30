from flask import Blueprint, render_template

help_bp = Blueprint("help", __name__)

@help_bp.route("/help")
def help_page():
    """
    KrakenRelay operator documentation page.
    Provides explanations of UI controls and system behavior.
    """
    return render_template("help.html")
