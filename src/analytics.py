"""
Traffic Analytics + Smart Alerts layer (the deck's "Outputs" column:
traffic analytics, congestion heatmaps, O-D patterns, bottleneck detection,
real-time alerts, unified dashboard data).
"""
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from . import config

BLACKLIST_PLATES = set()  # populated by pipeline.py from the watch-list
PLATE_MATCH_THRESHOLD = 0.85  # tolerate a 1-2 character OCR misread, as real ANPR must


def _best_watchlist_match(plate, watch_list):
    best_plate, best_score = None, 0.0
    for w in watch_list:
        score = SequenceMatcher(None, plate, w).ratio()
        if score > best_score:
            best_plate, best_score = w, score
    return best_plate, best_score


def per_camera_volume(observations_by_camera, video_duration_s):
    """Vehicles/minute at each camera -> congestion level."""
    result = {}
    for cam, obs_list in observations_by_camera.items():
        rate_per_min = len(obs_list) / (video_duration_s / 60.0)
        if rate_per_min <= config.CONGESTION_LOW_MAX:
            level = "Low"
        elif rate_per_min <= config.CONGESTION_NORMAL_MAX:
            level = "Normal"
        else:
            level = "High"
        result[cam] = {"vehicle_count": len(obs_list),
                        "vehicles_per_minute": round(rate_per_min, 2),
                        "congestion_level": level}
    return result


def travel_time_and_speed(trajectories):
    legs_summary = []
    for traj in trajectories:
        for leg in traj.legs:
            distance_m = (config.CAMERAS[leg.camera_b]["position_m"]
                          - config.CAMERAS[leg.camera_a]["position_m"])
            dt = leg.obs_b.t_mid - leg.obs_a.t_mid
            speed_kmh = (distance_m / dt) * 3.6 if dt > 0 else None
            legs_summary.append({
                "vehicle": traj.plate or traj.vehicle_key,
                "from": leg.camera_a, "to": leg.camera_b,
                "travel_time_s": round(dt, 1),
                "speed_kmh": round(speed_kmh, 1) if speed_kmh else None,
                "confidence": round(leg.confidence, 2),
            })
    return legs_summary


def origin_destination_patterns(trajectories):
    od_counts = Counter()
    for traj in trajectories:
        if len(traj.camera_sequence) >= 2:
            od_counts[(traj.camera_sequence[0], traj.camera_sequence[-1])] += 1
    return [{"origin": o, "destination": d, "vehicle_count": c} for (o, d), c in od_counts.items()]


def bottleneck_detection(travel_legs):
    """Flags camera-to-camera segments whose average speed is well below the
    free-flow band -> candidate congestion bottleneck."""
    by_segment = defaultdict(list)
    for leg in travel_legs:
        if leg["speed_kmh"] is not None:
            by_segment[(leg["from"], leg["to"])].append(leg["speed_kmh"])
    bottlenecks = []
    for (a, b), speeds in by_segment.items():
        avg_speed = sum(speeds) / len(speeds)
        if avg_speed < config.EXPECTED_SPEED_MIN_KMH * 1.15:
            bottlenecks.append({"segment": f"{a}->{b}", "avg_speed_kmh": round(avg_speed, 1),
                                 "sample_size": len(speeds)})
    return bottlenecks


def smart_alerts(trajectories, watch_list):
    alerts = []
    for traj in trajectories:
        watch_match, watch_score = (_best_watchlist_match(traj.plate, watch_list)
                                     if traj.plate and watch_list else (None, 0.0))
        if watch_match and watch_score >= PLATE_MATCH_THRESHOLD:
            alerts.append({
                "type": "Blacklisted / Watch-listed vehicle",
                "plate": traj.plate,
                "matched_watchlist_plate": watch_match,
                "plate_match_score": round(watch_score, 2),
                "route": " -> ".join(traj.camera_sequence),
                "confidence": round(traj.overall_confidence, 2),
                "label": traj.label,
                "requires_human_review": traj.label != "High Confidence" or watch_score < 1.0,
            })
        elif traj.label == "Review Required" and len(traj.camera_sequence) > 1:
            alerts.append({
                "type": "Low-confidence cross-camera match",
                "plate": traj.plate or traj.vehicle_key,
                "route": " -> ".join(traj.camera_sequence),
                "confidence": round(traj.overall_confidence, 2),
                "label": traj.label,
                "requires_human_review": True,
            })
    return alerts


def build_report(observations_by_camera, trajectories, video_duration_s, watch_list=None):
    watch_list = watch_list or set()
    travel_legs = travel_time_and_speed(trajectories)
    return {
        "camera_volume": per_camera_volume(observations_by_camera, video_duration_s),
        "travel_legs": travel_legs,
        "origin_destination": origin_destination_patterns(trajectories),
        "bottlenecks": bottleneck_detection(travel_legs),
        "alerts": smart_alerts(trajectories, watch_list),
        "total_trajectories": len(trajectories),
        "multi_camera_trajectories": sum(1 for t in trajectories if len(t.camera_sequence) > 1),
    }
