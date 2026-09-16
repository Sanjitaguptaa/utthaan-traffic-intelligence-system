"""
License plate cropping, enhancement and OCR ("ANPR / OCR" stage in the deck).

Uses pytesseract (Tesseract OCR) rather than PaddleOCR/EasyOCR so the
prototype has no large model downloads and runs fully offline/CPU-only.
Swapping the OCR backend is isolated to `read_plate()`.
"""
import re

import cv2
import numpy as np
import pytesseract

PLATE_PATTERN = re.compile(r"[A-Z0-9]{6,12}")
TESSERACT_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def crop_plate_region(vehicle_crop):
    """The plate sits in the bottom third of the vehicle bounding box (see
    detection.py). For a real detector, replace this with a dedicated plate
    detector/localizer (e.g. a small YOLO head) run on the vehicle crop."""
    h, w = vehicle_crop.shape[:2]
    y0 = int(h * 0.62)
    return vehicle_crop[y0:h, int(w * 0.05):int(w * 0.95)]


def enhance_plate(plate_img):
    """Image Enhancement stage: denoise, upscale, and binarize for OCR robustness
    under blur / low-light conditions, as called out in the deck's AI Processing Layer."""
    if plate_img.size == 0:
        return plate_img
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if plate_img.ndim == 3 else plate_img
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.fastNlMeansDenoising(gray, h=15)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return gray


def read_plate(vehicle_crop):
    """Returns (plate_text, ocr_confidence 0-1) or (None, 0.0) if nothing legible."""
    plate_crop = crop_plate_region(vehicle_crop)
    if plate_crop.size == 0:
        return None, 0.0
    processed = enhance_plate(plate_crop)

    try:
        data = pytesseract.image_to_data(
            processed, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DICT
        )
    except pytesseract.TesseractError:
        return None, 0.0

    texts, confs = [], []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        txt = txt.strip().upper()
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = -1.0
        if txt and conf > 0:
            texts.append(txt)
            confs.append(conf)

    if not texts:
        return None, 0.0

    joined = "".join(texts)
    match = PLATE_PATTERN.search(joined)
    if not match:
        return None, 0.0

    avg_conf = float(np.mean(confs)) / 100.0
    return match.group(0), min(1.0, max(0.0, avg_conf))
