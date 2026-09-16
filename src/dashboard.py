"""
City Traffic Intelligence Dashboard (Flask).

Serves the JSON produced by pipeline.py (camera volumes, congestion levels,
travel legs, O-D patterns, bottlenecks, smart alerts) plus the GIS trajectory
map, as a single-page dashboard mirroring the deck's "City Traffic
Intelligence Dashboard - LIVE VIEW" mockup.

Run with:  python -m src.dashboard
Then open: http://localhost:5050
"""
import json
import os

from flask import Flask, jsonify, render_template, send_from_directory

from . import config

app = Flask(__name__, template_folder="templates", static_folder="static")


def _load(name):
    path = os.path.join(config.OUTPUT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(config.OUTPUT_DIR, filename)


@app.route("/api/data")
def api_data():
    report = _load("traffic_analytics.json")
    trajectories = _load("trajectories.json")
    validation = _load("validation_report.json")
    if report is None:
        return jsonify({"error": "No pipeline output found. Run: python -m src.pipeline"}), 404
    return jsonify({
        "cameras": config.CAMERAS,
        "camera_order": config.CAMERA_ORDER,
        "report": report,
        "trajectories": trajectories,
        "validation": validation,
    })


def main():
    print("UTTHAAN City Traffic Intelligence Dashboard running at http://localhost:5050")
    app.run(host="0.0.0.0", port=5050, debug=False)


if __name__ == "__main__":
    main()
