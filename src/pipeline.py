"""
End-to-end orchestrator for the UTTHAAN City-Wide AI Traffic Intelligence
Engine prototype.

Stage order mirrors the SIH deck exactly:
  Input Layer -> AI Processing Layer -> Cross-Camera Intelligence ->
  GIS / Map Intelligence Layer -> Outputs (analytics + smart alerts)

Run with:  python -m src.pipeline
"""
import argparse
import json
import os
import time
from difflib import SequenceMatcher

from . import config
from .analytics import build_report
from .association import build_trajectories
from .camera_pipeline import process_camera_video
from .gis_layer import build_city_map
from .synthetic_data import generate_dataset


def run(data_dir=None, output_dir=None, verbose=True):
    data_dir = data_dir or config.DATA_DIR
    output_dir = output_dir or config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    gt_path = os.path.join(data_dir, "ground_truth.json")
    if not os.path.exists(gt_path):
        if verbose:
            print("No camera footage found - generating synthetic 4-camera dataset...")
        generate_dataset(data_dir)
    with open(gt_path) as fh:
        ground_truth = json.load(fh)
    video_duration_s = max(e["t_exit"] for e in ground_truth["events"]) + 5

    watch_list = {v["plate"] for v in ground_truth["real_vehicles"] if v["blacklisted"]}

    # ---- Input Layer + AI Processing Layer (per camera) ----
    t0 = time.time()
    observations_by_camera = {}
    for cam in config.CAMERA_ORDER:
        video_path = os.path.join(data_dir, f"{cam}.avi")
        obs = process_camera_video(cam, video_path, verbose=verbose)
        observations_by_camera[cam] = obs
    detection_time = time.time() - t0

    # ---- Cross-Camera Intelligence ----
    trajectories = build_trajectories(observations_by_camera)
    multi_cam = [t for t in trajectories if len(t.camera_sequence) > 1]
    if verbose:
        print(f"\nReconstructed {len(multi_cam)} multi-camera trajectories "
              f"out of {len(trajectories)} total tracked vehicles.")

    # ---- GIS / Map Intelligence Layer ----
    map_path = build_city_map(trajectories, os.path.join(output_dir, "city_traffic_map.html"))

    # ---- Outputs: analytics + smart alerts ----
    report = build_report(observations_by_camera, trajectories, video_duration_s, watch_list)
    report["processing_time_s"] = round(detection_time, 2)
    report_path = os.path.join(output_dir, "traffic_analytics.json")
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)

    trajectories_export = [{
        "vehicle": t.plate or t.vehicle_key,
        "camera_sequence": t.camera_sequence,
        "confidence": round(t.overall_confidence, 3),
        "label": t.label,
        "legs": [{"from": l.camera_a, "to": l.camera_b, "confidence": round(l.confidence, 3),
                  "breakdown": {k: (round(v, 3) if v is not None else None)
                                for k, v in l.score_breakdown.items()}}
                 for l in t.legs],
    } for t in trajectories]
    traj_path = os.path.join(output_dir, "trajectories.json")
    with open(traj_path, "w") as fh:
        json.dump(trajectories_export, fh, indent=2)

    validation = validate_against_ground_truth(trajectories, ground_truth)
    alerted_plates = {a["plate"] for a in report["alerts"] if a["type"].startswith("Blacklisted")}
    validation["blacklist_alerts_correct"] = any(
        SequenceMatcher(None, p, ap).ratio() >= 0.85
        for p in watch_list for ap in alerted_plates
    ) if watch_list else None
    val_path = os.path.join(output_dir, "validation_report.json")
    with open(val_path, "w") as fh:
        json.dump(validation, fh, indent=2)

    if verbose:
        print(f"\n=== VALIDATION AGAINST GROUND TRUTH ===")
        print(f"Real vehicles that traversed all 4 cameras: {validation['total_real_vehicles']}")
        print(f"Correctly reconstructed end-to-end (plate-matched): {validation['correctly_reconstructed']}")
        print(f"Trajectory reconstruction accuracy: {validation['accuracy_pct']}%")
        print(f"Blacklist alerts fired correctly: {validation['blacklist_alerts_correct']}")
        print(f"\nOutputs written to: {output_dir}")
        print(f"  - {os.path.basename(map_path)} (interactive GIS trajectory map)")
        print(f"  - {os.path.basename(report_path)} (traffic analytics + alerts)")
        print(f"  - {os.path.basename(traj_path)} (reconstructed trajectories)")
        print(f"  - {os.path.basename(val_path)} (ground-truth validation)")

    return {
        "observations_by_camera": observations_by_camera,
        "trajectories": trajectories,
        "report": report,
        "validation": validation,
        "map_path": map_path,
    }


def validate_against_ground_truth(trajectories, ground_truth):
    """A trajectory counts as 'correctly reconstructed' when it spans all 4
    cameras in order AND its (OCR-read, possibly imperfect) plate is a close
    match to a real vehicle's true plate. Requiring a *close* match rather
    than an exact string match is deliberate and matches how the deck's own
    confidence-scored association is meant to behave: OCR on real CCTV footage
    is never 100% character-perfect, so the system is designed to still link
    a vehicle across cameras via combined plate+appearance+timing evidence
    even when a character or two is misread — an exact-string requirement
    would penalize the system for realistic OCR noise, not for a real
    association failure."""
    full_route = tuple(config.CAMERA_ORDER)
    full_route_trajs = [t for t in trajectories if tuple(t.camera_sequence) == full_route and t.plate]

    reconstructed, missed = [], []
    for v in ground_truth["real_vehicles"]:
        best = max(
            (SequenceMatcher(None, v["plate"], t.plate).ratio() for t in full_route_trajs),
            default=0.0,
        )
        (reconstructed if best >= 0.85 else missed).append(v["plate"])

    total = len(ground_truth["real_vehicles"])
    correct = len(reconstructed)

    return {
        "total_real_vehicles": total,
        "correctly_reconstructed": correct,
        "accuracy_pct": round(100 * correct / total, 1) if total else 0.0,
        "reconstructed_plates": sorted(reconstructed),
        "missed_plates": sorted(missed),
        "note": "A trajectory counts as correct if it spans all 4 cameras and its "
                "read plate is a >=85% string match to the true plate (tolerates "
                "realistic single-character OCR noise, as real ANPR systems must).",
    }


def main():
    parser = argparse.ArgumentParser(description="UTTHAAN City-Wide Traffic Intelligence Engine")
    parser.add_argument("--regenerate-data", action="store_true",
                         help="Force-regenerate the synthetic camera dataset")
    parser.add_argument("--data-dir", default=config.DATA_DIR)
    parser.add_argument("--output-dir", default=config.OUTPUT_DIR)
    args = parser.parse_args()

    if args.regenerate_data:
        gt_path = os.path.join(args.data_dir, "ground_truth.json")
        if os.path.exists(gt_path):
            os.remove(gt_path)
        generate_dataset(args.data_dir)

    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
