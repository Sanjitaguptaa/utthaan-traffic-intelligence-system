"""
Single-camera multi-object tracking ("Single-Camera Tracking (ByteTrack /
BGI-SHORT / DeepSORT)" in the deck).

Implements a compact SORT-style tracker: a Kalman filter predicts each
track's next position, and detections are assigned to tracks by IOU (via the
Hungarian algorithm). This is the same family of algorithm the deck names
(ByteTrack/DeepSORT are refinements of SORT); it keeps a stable per-camera
track ID for each vehicle across frames so plate reads can be aggregated
per-track rather than per-frame.
"""
from dataclasses import dataclass, field

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def iou(bb1, bb2):
    x1, y1, w1, h1 = bb1
    x2, y2, w2, h2 = bb2
    xa, ya = max(x1, x2), max(y1, y2)
    xb, yb = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


class _Track:
    _next_id = 1

    def __init__(self, detection):
        self.id = _Track._next_id
        _Track._next_id += 1
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self._init_kalman(detection.bbox)
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        self.plate_votes = {}
        self.best_frames = []  # (frame_idx, bbox) for OCR sampling
        self.first_frame = detection.frame_idx
        self.last_frame = detection.frame_idx
        self.color_samples = []

    def _init_kalman(self, bbox):
        x, y, w, h = bbox
        cx, cy, s, r = x + w / 2, y + h / 2, w * h, w / max(h, 1)
        self.kf.x[:4] = np.array([[cx], [cy], [s], [r]])
        self.kf.F = np.eye(7)
        for i in range(3):
            self.kf.F[i, i + 4] = 1
        self.kf.H = np.zeros((4, 7))
        self.kf.H[:4, :4] = np.eye(4)
        self.kf.P *= 10.0
        self.kf.R *= 5.0
        self.kf.Q *= 0.01

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.get_bbox()

    def update(self, detection):
        x, y, w, h = detection.bbox
        cx, cy, s, r = x + w / 2, y + h / 2, w * h, w / max(h, 1)
        self.kf.update(np.array([[cx], [cy], [s], [r]]))
        self.hits += 1
        self.time_since_update = 0
        self.last_frame = detection.frame_idx
        self.best_frames.append((detection.frame_idx, detection.bbox))

    def get_bbox(self):
        cx, cy, s, r = self.kf.x[:4, 0]
        w = np.sqrt(max(s * r, 1))
        h = s / max(w, 1)
        return (int(cx - w / 2), int(cy - h / 2), int(w), int(h))


class SortTracker:
    """Note: `tracks` holds only currently-active tracks (used for frame-to-
    frame matching). `all_tracks` additionally retains tracks that have since
    aged out, so a completed vehicle passage isn't lost once the vehicle
    leaves frame — callers that want every track seen during the whole video
    (e.g. camera_pipeline.py) should read `all_tracks` after processing."""

    def __init__(self, max_age=8, min_hits=2, iou_threshold=0.25):
        self.tracks = []
        self.all_tracks = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def update(self, detections):
        for t in self.tracks:
            t.predict()

        if self.tracks and detections:
            cost = np.zeros((len(self.tracks), len(detections)))
            for i, t in enumerate(self.tracks):
                for j, d in enumerate(detections):
                    cost[i, j] = 1 - iou(t.get_bbox(), d.bbox)
            row_idx, col_idx = linear_sum_assignment(cost)
            matched_tracks, matched_dets = set(), set()
            for r, c in zip(row_idx, col_idx):
                if cost[r, c] <= (1 - self.iou_threshold):
                    self.tracks[r].update(detections[c])
                    matched_tracks.add(r)
                    matched_dets.add(c)
            unmatched_dets = [d for j, d in enumerate(detections) if j not in matched_dets]
        else:
            unmatched_dets = detections

        for d in unmatched_dets:
            new_track = _Track(d)
            self.tracks.append(new_track)
            self.all_tracks.append(new_track)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        return [t for t in self.tracks if t.hits >= self.min_hits]
