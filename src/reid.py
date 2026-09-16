"""
Vehicle Re-Identification ("Vehicle RE-ID (Appearance features, OSNet)" in
the deck).

A full OSNet embedding network needs a GPU and a pretrained-weights download.
This prototype instead builds a lightweight, dependency-free appearance
descriptor from an HSV colour histogram + edge-density signature of the
vehicle crop, and compares vehicles with cosine similarity. It plugs into the
exact same slot a real Re-ID network would occupy: `embed(vehicle_crop) ->
vector`, `similarity(vec_a, vec_b) -> 0..1`.
"""
import cv2
import numpy as np


def embed(vehicle_crop):
    if vehicle_crop is None or vehicle_crop.size == 0:
        return np.zeros(48 + 16)
    hsv = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 3], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()

    gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_hist, _ = np.histogram(edges.flatten(), bins=16, range=(0, 255))
    edge_hist = edge_hist / (edge_hist.sum() + 1e-8)

    return np.concatenate([hist, edge_hist])


def similarity(vec_a, vec_b):
    na, nb = np.linalg.norm(vec_a), np.linalg.norm(vec_b)
    if na == 0 or nb == 0:
        return 0.0
    cos_sim = float(np.dot(vec_a, vec_b) / (na * nb))
    return max(0.0, min(1.0, cos_sim))
