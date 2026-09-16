"""
Central configuration for the UTTHAAN City-Wide Traffic Intelligence Engine.

Defines the simulated city road network, camera placements (with real-world-style
lat/lon so the GIS layer is meaningful), and tunable pipeline parameters.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_W, FRAME_H = 640, 480
FPS = 15

# Four cameras placed along a simple corridor (mirrors the C1->C2->C3->C4 example
# in the SIH deck). Coordinates are illustrative lat/lon points (Jammu region,
# near Katra, so the digital twin plots on a real map). position_m is kept
# short so synthetic demo videos stay a manageable length.
CAMERAS = {
    "C1": {"name": "Camera 1 - Entry Point",  "lat": 32.9916, "lon": 74.8480, "position_m": 0},
    "C2": {"name": "Camera 2 - Mid Route",     "lat": 32.9944, "lon": 74.8555, "position_m": 100},
    "C3": {"name": "Camera 3 - Near Exit",     "lat": 32.9978, "lon": 74.8631, "position_m": 220},
    "C4": {"name": "Camera 4 - Exit Point",    "lat": 33.0010, "lon": 74.8702, "position_m": 350},
}
CAMERA_ORDER = ["C1", "C2", "C3", "C4"]

# Clean, vehicle-free seconds at the start of every camera video so the
# background-subtraction detector always learns a clean static scene first.
STARTUP_BUFFER_S = 2.5

# Expected free-flow speed range (km/h) used for travel-time plausibility scoring
EXPECTED_SPEED_MIN_KMH = 15
EXPECTED_SPEED_MAX_KMH = 80

# Cross-camera association confidence weighting (mirrors the deck's
# "Dynamic Trajectory Confidence Score" signal list)
ASSOCIATION_WEIGHTS = {
    "plate_similarity": 0.45,
    "appearance_similarity": 0.20,
    "timestamp_compatibility": 0.20,
    "direction_consistency": 0.15,
}
HIGH_CONFIDENCE_THRESHOLD = 0.80
PROBABLE_THRESHOLD = 0.50

# Simple congestion thresholds (vehicles / minute passing a camera)
CONGESTION_LOW_MAX = 4
CONGESTION_NORMAL_MAX = 8  # above this => High congestion
