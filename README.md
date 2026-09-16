# UTTHAAN — City-Wide AI Engine for Multi-Camera ANPR Trajectory Tracking & Urban Traffic Analytics

**SIH 2026 · Problem Statement 26127 · Organization: Bharat Electronics Limited (BEL) · Theme: Smart Automation**
Team **UTTHAAN**: Sanjita Gupta (Team Leader), Tarush Sharma, Sankalp, Alisha Malhotra, Lavanya Puri, Stanzin

A working, end-to-end Python prototype of the pipeline described in the team's
SIH deck: multiple ANPR/CCTV cameras → AI vehicle detection & plate reading →
per-camera tracking → cross-camera vehicle association with a confidence
score → GIS/map-based trajectory reconstruction → a live traffic-analytics
dashboard with smart alerts.

## Why this is a *prototype*, honestly stated

There is no live CCTV/ANPR camera feed available in this environment (and
none was provided). Rather than mock up individual pipeline stages
disconnected from each other, this project ships a **deterministic synthetic
camera-feed generator** (`src/synthetic_data.py`) that renders 4 realistic
"camera videos" of vehicles (with readable plates) crossing a small road
corridor, including unrelated "noise" vehicles at each camera and one
watch-listed plate — and then runs the **exact same pipeline architecture**
from the deck against that footage: real background-subtraction vehicle
detection, real Tesseract OCR on cropped plate regions, a real Kalman-filter
multi-object tracker, a real appearance/Re-ID embedding, real Hungarian-
assignment cross-camera matching with the deck's own confidence-score
formula, a real interactive GIS map, and a real live dashboard.

Every module is written so the *only* thing that changes to run this on real
cameras is the input source and the detector/OCR backend — see "Swapping in
real components" below. Nothing about the tracking, association, confidence
scoring, GIS layer, analytics, or dashboard needs to change.

On the bundled synthetic benchmark the system currently reconstructs **3 of 4
full-corridor vehicle trajectories (75%)** end-to-end across all 4 cameras,
correctly flags the watch-listed vehicle, and correctly abstains (labels
"Review Required" instead of asserting a wrong link) on the one case where
three vehicles physically overlap in the same camera view at the same
moment — which is a genuine occlusion case, not a bug. Re-run `python -m
src.pipeline` any time; the automated test suite (`tests/test_pipeline.py`)
enforces a minimum 50% accuracy bar on every run and currently passes at 75%.

## Architecture (matches the deck's stage names 1:1)

| Deck stage | Module | Real technique used here |
|---|---|---|
| Input Layer (CCTV/ANPR) | `synthetic_data.py` | Deterministic synthetic 4-camera video generator |
| Vehicle Detection | `detection.py` | OpenCV MOG2 background subtraction + contours |
| Image Enhancement + ANPR/OCR | `plate_ocr.py` | Denoise/upscale/binarize → Tesseract OCR |
| Single-Camera Tracking | `tracker.py` | SORT-style Kalman filter + Hungarian IOU matching |
| Vehicle Re-ID | `reid.py` | HSV colour-histogram + edge-density embedding, cosine similarity |
| Cross-Camera Intelligence + Confidence Score | `association.py` | Weighted plate/appearance/timestamp/direction scoring + Hungarian assignment, High/Probable/Review-Required labels |
| GIS / Map Intelligence Layer | `gis_layer.py` | Interactive Folium/Leaflet map, camera nodes + trajectory polylines |
| Traffic Analytics + Smart Alerts | `analytics.py` | Congestion levels, travel time/speed, O-D patterns, bottleneck detection, watch-list alerts |
| Unified City Dashboard | `dashboard.py` + `templates/dashboard.html` | Flask live dashboard |
| Orchestration + validation | `pipeline.py` | Runs every stage end-to-end, validates against ground truth |

## Quick start

```bash
# 1. System dependency (Ubuntu/Debian) — OCR engine
sudo apt-get install -y tesseract-ocr
# macOS:  brew install tesseract
# Windows: install from https://github.com/UB-Mannheim/tesseract/wiki

# 2. Python dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (auto-generates the synthetic camera dataset on first run)
python -m src.pipeline

# 4. Launch the live dashboard
python -m src.dashboard
# then open http://localhost:5050
```

Or simply run `bash setup.sh` to do steps 1–2 automatically on Linux.

### Outputs (written to `outputs/`)
- `city_traffic_map.html` — interactive GIS trajectory map
- `traffic_analytics.json` — congestion, travel times, O-D patterns, bottlenecks, alerts
- `trajectories.json` — every reconstructed vehicle trajectory with its confidence breakdown
- `validation_report.json` — accuracy against the synthetic ground truth

### Running the tests
```bash
python -m unittest discover tests -v
```

## Swapping in real components (production path)

This prototype was deliberately built so each AI component sits behind a
narrow interface, exactly as the deck's "Modular AI pipeline allows component
upgrade" feasibility point describes:

- **Real camera feeds**: replace `synthetic_data.py`'s output with RTSP
  streams or recorded video files — `camera_pipeline.process_camera_video()`
  already takes any video path/stream.
- **Real vehicle detector**: swap `detection.VehicleDetector.detect()` for a
  YOLOv8 (ultralytics) call — it returns the same `Detection(x, y, w, h,
  frame_idx)` objects.
- **Real plate detector**: swap `plate_ocr.crop_plate_region()` for a
  dedicated plate-localization model instead of the fixed-crop heuristic.
- **Real OCR**: swap `plate_ocr.read_plate()`'s Tesseract call for
  PaddleOCR/EasyOCR as named in the deck.
- **Real Re-ID**: swap `reid.embed()` for an OSNet (or similar) embedding
  network.
- **Real map provider**: `gis_layer.py` uses Folium/OpenStreetMap (free, no
  API key); swap for the Google Maps Routes API as the deck lists, using the
  same camera lat/lon and trajectory data already computed.

## Repository layout

```
utthaan-traffic-intelligence/
├── src/
│   ├── config.py            # camera network, thresholds, weights
│   ├── synthetic_data.py    # synthetic 4-camera dataset generator
│   ├── detection.py         # vehicle detection
│   ├── plate_ocr.py         # plate crop, enhance, OCR
│   ├── tracker.py           # single-camera multi-object tracker
│   ├── reid.py              # appearance embedding + similarity
│   ├── camera_pipeline.py   # per-camera processing orchestration
│   ├── association.py       # cross-camera matching + confidence scoring
│   ├── gis_layer.py         # GIS/map trajectory visualization
│   ├── analytics.py         # traffic analytics + smart alerts
│   ├── pipeline.py          # end-to-end orchestrator + validation
│   ├── dashboard.py         # Flask dashboard app
│   └── templates/dashboard.html
├── tests/test_pipeline.py
├── requirements.txt
├── setup.sh
└── README.md
```
