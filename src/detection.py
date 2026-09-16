"""
Vehicle detection layer.

The SIH deck specifies a YOLO-class detector ("Vehicle Detection (VLO or
equivalent)"). To keep this prototype runnable anywhere with no GPU, no model
download, and no internet access to a weights host, detection here uses
OpenCV background subtraction (MOG2) + contour extraction, which is a
legitimate classical CV vehicle detector for a fixed camera view (exactly the
CCTV setup in the deck).

Swapping in a real detector is a one-function change: replace `VehicleDetector
.detect(frame)` with a YOLOv8 (ultralytics) inference call that returns the
same `Detection` objects. Everything downstream (tracking, OCR, Re-ID,
association) is detector-agnostic.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int
    frame_idx: int

    @property
    def bbox(self):
        return (self.x, self.y, self.w, self.h)

    @property
    def centroid(self):
        return (self.x + self.w / 2, self.y + self.h / 2)


class VehicleDetector:
    def __init__(self, min_area=900, max_area=20000):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=50, varThreshold=25, detectShadows=False
        )
        self.min_area = min_area
        self.max_area = max_area
        self._warmed_up = False

    def warm_up(self, frame):
        """Feed a background-only frame so the subtractor learns the static scene."""
        for _ in range(5):
            self.bg_subtractor.apply(frame)
        self._warmed_up = True

    def detect(self, frame, frame_idx=0):
        fg_mask = self.bg_subtractor.apply(frame, learningRate=0.001)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg_mask = cv2.dilate(fg_mask, np.ones((5, 5), np.uint8), iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(c)
                detections.append(Detection(x, y, w, h, frame_idx))
        return detections
