"""
GIS / Map Intelligence Layer.

Renders camera locations and reconstructed vehicle trajectories on an
interactive Folium (Leaflet/OpenStreetMap) map, matching the deck's
"GIS / Map Layer (map-provider agnostic)" stage. Folium is used instead of a
paid Google Maps key so the map renders with zero API credentials; the deck
explicitly calls this layer "map-provider agnostic".
"""
import os

import folium

from . import config

TRAJECTORY_COLORS = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#9A6324",
]


def build_city_map(trajectories, output_path=None):
    output_path = output_path or os.path.join(config.OUTPUT_DIR, "city_traffic_map.html")

    lats = [c["lat"] for c in config.CAMERAS.values()]
    lons = [c["lon"] for c in config.CAMERAS.values()]
    center = [sum(lats) / len(lats), sum(lons) / len(lons)]

    fmap = folium.Map(location=center, zoom_start=14, tiles="OpenStreetMap")

    # Camera nodes
    for cam_id, cam in config.CAMERAS.items():
        folium.Marker(
            location=[cam["lat"], cam["lon"]],
            tooltip=f"{cam_id}: {cam['name']}",
            icon=folium.Icon(color="darkblue", icon="video", prefix="fa"),
        ).add_to(fmap)

    # Road network line (linear corridor here, connecting cameras in order)
    corridor = [[config.CAMERAS[c]["lat"], config.CAMERAS[c]["lon"]] for c in config.CAMERA_ORDER]
    folium.PolyLine(corridor, color="#888888", weight=3, opacity=0.5, dash_array="5,10").add_to(fmap)

    # Reconstructed vehicle trajectories
    multi_camera_trajs = [t for t in trajectories if len(t.camera_sequence) > 1]
    for i, traj in enumerate(multi_camera_trajs):
        color = TRAJECTORY_COLORS[i % len(TRAJECTORY_COLORS)]
        points = [[config.CAMERAS[c]["lat"], config.CAMERAS[c]["lon"]] for c in traj.camera_sequence]
        popup = (f"<b>Vehicle:</b> {traj.plate or traj.vehicle_key}<br>"
                 f"<b>Route:</b> {' -> '.join(traj.camera_sequence)}<br>"
                 f"<b>Confidence:</b> {traj.overall_confidence:.0%} ({traj.label})")
        folium.PolyLine(
            points, color=color, weight=5, opacity=0.85,
            tooltip=f"{traj.plate or traj.vehicle_key} ({traj.label})",
            popup=folium.Popup(popup, max_width=300),
        ).add_to(fmap)
        for pt, cam_id in zip(points, traj.camera_sequence):
            folium.CircleMarker(pt, radius=5, color=color, fill=True, fill_opacity=1).add_to(fmap)

    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 6px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.3); font-family: sans-serif; font-size: 13px;">
        <b>UTTHAAN City Traffic Digital Twin</b><br>
        Blue markers = ANPR cameras<br>
        Coloured lines = reconstructed cross-camera vehicle trajectories
    </div>
    """
    fmap.get_root().html.add_child(folium.Element(legend_html))
    fmap.save(output_path)
    return output_path
