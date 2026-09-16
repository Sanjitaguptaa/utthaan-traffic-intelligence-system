"""
Runs the full single-camera stack (Input -> AI Processing Layer) on one
camera's video and produces a list of `CameraObservation` objects: one per
vehicle track that was confidently seen by that camera, carrying its best
read plate, appearance embedding, and dwell-time window. These observations
are what feed the Cross-Camera Intelligence stage (association.py).
"""
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from . import config
from .detection import VehicleDetector
from .plate_ocr import read_plate
from .reid import embed
from .tracker import SortTracker


@dataclass
class CameraObservation:
    camera: str
    track_id: int
    plate: Optional[str]
    plate_confidence: float
    appearance_vec: np.ndarray
    first_frame: int
    last_frame: int
    fps: float
    color_bgr_mean: tuple = field(default=(0, 0, 0))

    @property
    def t_enter(self):
        return self.first_frame / self.fps

    @property
    def t_exit(self):
        return self.last_frame / self.fps

    @property
    def t_mid(self):
        return (self.t_enter + self.t_exit) / 2


def process_camera_video(camera_id, video_path, sample_every=2, verbose=False):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or config.FPS

    detector = VehicleDetector()
    tracker = SortTracker()

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return []
    detector.warm_up(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    track_plate_votes = {}
    track_embeddings = {}
    track_colors = {}
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame, frame_idx)
        tracks = tracker.update(detections)

        if frame_idx % sample_every == 0:
            for t in tracks:
                x, y, w, h = t.get_bbox()
                x, y = max(0, x), max(0, y)
                crop = frame[y:y + h, x:x + w]
                if crop.size == 0:
                    continue
                plate, conf = read_plate(crop)
                if plate:
                    track_plate_votes.setdefault(t.id, Counter())[plate] += conf + 0.01
                track_embeddings.setdefault(t.id, []).append(embed(crop))
                track_colors.setdefault(t.id, []).append(crop.reshape(-1, 3).mean(axis=0))

        frame_idx += 1

    cap.release()

    observations = []
    for t in tracker.all_tracks:
        if t.hits < tracker.min_hits:
            continue
        votes = track_plate_votes.get(t.id)
        plate, plate_conf = None, 0.0
        if votes:
            plate, score = votes.most_common(1)[0]
            plate_conf = min(1.0, score / max(1, sum(votes.values())))
        embeds = track_embeddings.get(t.id, [])
        mean_embed = np.mean(embeds, axis=0) if embeds else embed(None)
        colors = track_colors.get(t.id, [])
        mean_color = tuple(np.mean(colors, axis=0)) if colors else (0, 0, 0)

        observations.append(CameraObservation(
            camera=camera_id, track_id=t.id, plate=plate, plate_confidence=plate_conf,
            appearance_vec=mean_embed, first_frame=t.first_frame, last_frame=t.last_frame,
            fps=fps, color_bgr_mean=mean_color,
        ))

    if verbose:
        print(f"[{camera_id}] {len(observations)} vehicle track(s), "
              f"{sum(1 for o in observations if o.plate)} with a readable plate")
    return observations
