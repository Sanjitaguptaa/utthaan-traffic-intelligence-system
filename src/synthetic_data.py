"""
Synthetic multi-camera dataset generator.

The SIH deck assumes real CCTV/ANPR camera feeds. Since no live camera network
is available in this environment, this module generates a deterministic,
physically-consistent synthetic dataset (4 camera videos + ground truth) that
exercises every stage of the real pipeline exactly the way live footage would:
moving vehicles with readable license plates crossing camera views at times
consistent with a real speed/distance relationship, plus unrelated "distractor"
vehicles at each camera so cross-camera association has to do real work.

Swapping this module out for real RTSP/video-file ingestion (see detection.py's
docstring) is the only change needed to run the same pipeline on live cameras.
"""
import json
import os
import random

import cv2
import numpy as np

from . import config

random.seed(42)
np.random.seed(42)

ROAD_Y = config.FRAME_H // 2 + 40
VEHICLE_W, VEHICLE_H = 90, 46
PLATE_H = 16

VEHICLE_COLORS = {
    "red": (40, 40, 200),
    "blue": (200, 90, 40),
    "white": (235, 235, 235),
    "black": (30, 30, 30),
    "green": (60, 160, 60),
    "silver": (180, 180, 180),
}


def _draw_vehicle(frame, x, y, color_bgr, plate_text):
    x, y = int(x), int(y)
    cv2.rectangle(frame, (x, y), (x + VEHICLE_W, y + VEHICLE_H - PLATE_H), color_bgr, -1)
    cv2.rectangle(frame, (x, y), (x + VEHICLE_W, y + VEHICLE_H - PLATE_H), (0, 0, 0), 1)
    plate_y0 = y + VEHICLE_H - PLATE_H
    cv2.rectangle(frame, (x + 8, plate_y0), (x + VEHICLE_W - 8, plate_y0 + PLATE_H), (255, 255, 255), -1)
    cv2.rectangle(frame, (x + 8, plate_y0), (x + VEHICLE_W - 8, plate_y0 + PLATE_H), (0, 0, 0), 1)
    cv2.putText(frame, plate_text, (x + 10, plate_y0 + PLATE_H - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (0, 0, 0), 1, cv2.LINE_AA)


def _make_background():
    bg = np.full((config.FRAME_H, config.FRAME_W, 3), (70, 70, 70), dtype=np.uint8)
    cv2.rectangle(bg, (0, ROAD_Y - 55), (config.FRAME_W, ROAD_Y + 90), (60, 60, 60), -1)
    for lx in range(0, config.FRAME_W, 40):
        cv2.line(bg, (lx, ROAD_Y + 20), (lx + 20, ROAD_Y + 20), (200, 200, 0), 2)
    return bg


def generate_random_plate(rng):
    letters = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2))
    digits1 = rng.randint(1, 99)
    letters2 = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2))
    digits2 = rng.randint(1000, 9999)
    return f"JK{digits1:02d}{letters2}{digits2}"


def build_scenario():
    """Defines the ground-truth vehicles and every camera crossing event."""
    rng = random.Random(7)
    color_names = list(VEHICLE_COLORS.keys())

    real_vehicles = []
    speeds_kmh = [28, 34, 22, 40]
    for i, speed in enumerate(speeds_kmh):
        real_vehicles.append({
            "vehicle_id": f"V{i+1}",
            "plate": generate_random_plate(rng),
            "color": color_names[i % len(color_names)],
            "speed_kmh": speed,
            "departure_offset_s": i * 4.0,  # stagger so they don't all overlap at C1
            "blacklisted": (i == 2),  # V3 is flagged as a watch-listed vehicle
        })

    events = []  # (camera, t_enter_sec, t_exit_sec, vehicle_id, plate, color)
    CROSS_DURATION = 2.4  # seconds to cross a camera's field of view
    buffer = config.STARTUP_BUFFER_S
    for v in real_vehicles:
        speed_mps = v["speed_kmh"] * 1000 / 3600
        for cam in config.CAMERA_ORDER:
            t_center = buffer + v["departure_offset_s"] + config.CAMERAS[cam]["position_m"] / speed_mps
            t_center += rng.uniform(-0.3, 0.3)  # small real-world jitter
            events.append({
                "camera": cam,
                "t_enter": max(0.0, t_center - CROSS_DURATION / 2),
                "t_exit": t_center + CROSS_DURATION / 2,
                "vehicle_id": v["vehicle_id"],
                "plate": v["plate"],
                "color": v["color"],
                "blacklisted": v["blacklisted"],
            })

    # Distractor vehicles: appear at exactly one camera, never reappear.
    distractors = []
    for cam in config.CAMERA_ORDER:
        for _ in range(2):
            t_center = buffer + rng.uniform(3, 40)
            plate = generate_random_plate(rng)
            color = rng.choice(color_names)
            events.append({
                "camera": cam,
                "t_enter": t_center,
                "t_exit": t_center + CROSS_DURATION,
                "vehicle_id": f"NOISE-{cam}-{plate}",
                "plate": plate,
                "color": color,
                "blacklisted": False,
            })
            distractors.append(plate)

    video_duration = max(e["t_exit"] for e in events) + 3
    return real_vehicles, events, video_duration


def generate_dataset(out_dir=None):
    out_dir = out_dir or config.DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    real_vehicles, events, duration = build_scenario()

    events_by_cam = {cam: [] for cam in config.CAMERA_ORDER}
    for e in events:
        events_by_cam[e["camera"]].append(e)

    n_frames = int(duration * config.FPS)
    bg = _make_background()

    for cam in config.CAMERA_ORDER:
        path = os.path.join(out_dir, f"{cam}.avi")
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"),
                                  config.FPS, (config.FRAME_W, config.FRAME_H))
        cam_events = events_by_cam[cam]
        for f in range(n_frames):
            t = f / config.FPS
            frame = bg.copy()
            for e in cam_events:
                if e["t_enter"] <= t <= e["t_exit"]:
                    prog = (t - e["t_enter"]) / max(1e-6, (e["t_exit"] - e["t_enter"]))
                    x = -VEHICLE_W + prog * (config.FRAME_W + VEHICLE_W)
                    _draw_vehicle(frame, x, ROAD_Y - VEHICLE_H // 2,
                                  VEHICLE_COLORS[e["color"]], e["plate"])
            writer.write(frame)
        writer.release()

    ground_truth = {
        "real_vehicles": real_vehicles,
        "events": events,
        "fps": config.FPS,
        "frame_size": [config.FRAME_W, config.FRAME_H],
    }
    with open(os.path.join(out_dir, "ground_truth.json"), "w") as fh:
        json.dump(ground_truth, fh, indent=2)

    return ground_truth


if __name__ == "__main__":
    gt = generate_dataset()
    print(f"Generated {len(config.CAMERA_ORDER)} camera videos, "
          f"{len(gt['real_vehicles'])} real vehicles, {len(gt['events'])} total crossing events.")
