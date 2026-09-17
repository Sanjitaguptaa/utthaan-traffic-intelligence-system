"""
Streamlit entrypoint for deployment (e.g. Streamlit Community Cloud).

`src/dashboard.py` (Flask) is kept for local use, but Streamlit Cloud looks
for a top-level `app.py` running a Streamlit app specifically — the two
frameworks aren't interchangeable, so this is a separate UI built on the same
`src/` pipeline, not a rename of the Flask one.

Deploy: point Streamlit Community Cloud at this repo, main file = app.py.
Local run: streamlit run app.py
"""
import json
import os
import sys

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src import config  # noqa: E402

st.set_page_config(
    page_title="UTTHAAN | City Traffic Intelligence",
    page_icon="🚦",
    layout="wide",
)


def load_json(name):
    path = os.path.join(config.OUTPUT_DIR, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def run_pipeline():
    from src.pipeline import run
    with st.spinner("Generating synthetic camera feeds and running the full "
                     "detection -> OCR -> tracking -> Re-ID -> cross-camera "
                     "association -> GIS -> analytics pipeline... "
                     "(first run takes ~1-2 minutes)"):
        run(verbose=False)
    st.rerun()


st.markdown(
    """
    <div style="background:linear-gradient(90deg,#0b1f3a,#132a4d);
                padding:16px 24px;border-radius:8px;border-bottom:3px solid #e07a1f;
                margin-bottom:18px;">
      <h2 style="color:white;margin:0;">🚦 UTTHAAN — City-Wide AI Traffic Intelligence Dashboard</h2>
      <span style="color:#93a4bb;font-size:13px;">SIH 2026 · PS 26127 · Bharat Electronics Limited (BEL)</span>
    </div>
    """,
    unsafe_allow_html=True,
)

report = load_json("traffic_analytics.json")
trajectories = load_json("trajectories.json")
validation = load_json("validation_report.json")
map_path = os.path.join(config.OUTPUT_DIR, "city_traffic_map.html")

col_a, col_b = st.columns([5, 1])
with col_b:
    if st.button("🔄 Regenerate demo", use_container_width=True):
        run_pipeline()

if report is None:
    st.info("No pipeline output found yet. Click **Regenerate demo** above to "
            "generate the synthetic 4-camera dataset and run the full pipeline.")
    st.stop()

# ---- KPIs ----
total_veh = sum(v["vehicle_count"] for v in report["camera_volume"].values())
legs = report["travel_legs"]
avg_speed = round(sum(l["speed_kmh"] for l in legs if l["speed_kmh"]) /
                   max(1, sum(1 for l in legs if l["speed_kmh"])), 1) if legs else 0
avg_travel = round(sum(l["travel_time_s"] for l in legs) / max(1, len(legs)), 1) if legs else 0

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Vehicles Tracked", total_veh)
k2.metric("Multi-Camera Trajectories", report["multi_camera_trajectories"])
k3.metric("Avg Segment Speed", f"{avg_speed} km/h")
k4.metric("Avg Travel Time", f"{avg_travel} s")
k5.metric("Active Alerts", len(report["alerts"]))

if validation:
    st.caption(
        f"Ground-truth validation: **{validation['correctly_reconstructed']}/"
        f"{validation['total_real_vehicles']}** real vehicles correctly reconstructed "
        f"end-to-end across all 4 cameras (**{validation['accuracy_pct']}%** accuracy)."
    )

left, right = st.columns([1.3, 1])

with left:
    st.subheader("City Traffic Digital Twin (GIS Layer)")
    if os.path.exists(map_path):
        with open(map_path) as fh:
            components.html(fh.read(), height=520)
    else:
        st.warning("Map not generated yet.")

    st.subheader("Reconstructed Vehicle Trajectories")
    if trajectories:
        df = pd.DataFrame([{
            "Vehicle / Plate": t["vehicle"],
            "Camera Route": " → ".join(t["camera_sequence"]),
            "Confidence": f"{t['confidence']*100:.0f}%",
            "Label": t["label"],
        } for t in trajectories])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.write("No trajectories reconstructed.")

with right:
    st.subheader("Camera Status & Congestion")
    cam_df = pd.DataFrame([{
        "Camera": cam,
        "Vehicles": v["vehicle_count"],
        "Veh/min": v["vehicles_per_minute"],
        "Congestion": v["congestion_level"],
    } for cam, v in report["camera_volume"].items()])
    st.dataframe(cam_df, use_container_width=True, hide_index=True)

    st.subheader("Segment Speed")
    if legs:
        speed_df = pd.DataFrame([{
            "Segment": f"{l['vehicle']} ({l['from']}→{l['to']})",
            "Speed (km/h)": l["speed_kmh"],
        } for l in legs if l["speed_kmh"] is not None]).set_index("Segment")
        st.bar_chart(speed_df)
    else:
        st.write("No travel-time data.")

    st.subheader("Smart Alerts")
    if report["alerts"]:
        for a in report["alerts"]:
            msg = (f"**{a['type']}** — Plate `{a['plate']}` · Route {a['route']} · "
                   f"Confidence {a['confidence']*100:.0f}%")
            if a.get("requires_human_review"):
                msg += "  \n:orange[Human review required]"
            st.error(msg) if a["type"].startswith("Blacklisted") else st.warning(msg)
    else:
        st.write("No active alerts.")

st.caption(
    "This is a demonstrated prototype: camera input is a deterministic synthetic "
    "dataset (see README for why), run through the real detection → OCR → tracking "
    "→ Re-ID → cross-camera association → GIS → analytics pipeline."
)
