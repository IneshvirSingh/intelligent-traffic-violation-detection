"""
Centralized configuration for the Traffic Violation Detection System.
All parameters are defined here for easy tuning and deployment.
"""

import os
from pathlib import Path


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
VIOLATION_IMG_DIR = OUTPUT_DIR / "violations"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"
DB_PATH = BASE_DIR / "violations.db"

# Ensure output directories exist
VIOLATION_IMG_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# YOLOv8 Detection
# ──────────────────────────────────────────────
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL", "yolov8n.pt")
DETECTION_CONFIDENCE = 0.4
DETECTION_IOU_THRESHOLD = 0.45

# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# License plate detector (optional secondary model)
PLATE_MODEL_PATH = os.environ.get("PLATE_MODEL", None)
PLATE_CONFIDENCE = 0.3

# ──────────────────────────────────────────────
# Tracking (ByteTrack-style)
# ──────────────────────────────────────────────
TRACK_HIGH_THRESH = 0.5          # detection score above this → first association
TRACK_LOW_THRESH = 0.1           # detection score above this → second association
TRACK_MATCH_THRESH = 0.8         # IoU threshold for matching
TRACK_BUFFER = 30                # frames to keep lost tracks alive
MAX_TIME_LOST = 30               # synonym used in tracker

# ──────────────────────────────────────────────
# Violation Detection
# ──────────────────────────────────────────────

# -- Red-light violation --
# Stop-line defined as a horizontal y-coordinate (pixels from top)
STOP_LINE_Y = 400
# Width span of the stop-line (x_start, x_end)
STOP_LINE_X_RANGE = (100, 1100)
# Default traffic light state: "RED", "GREEN", "YELLOW"
DEFAULT_SIGNAL_STATE = "RED"

# -- Lane violation --
# Allowed lane polygons as lists of (x, y) vertices
ALLOWED_LANES = [
    [(200, 720), (200, 400), (500, 400), (500, 720)],
    [(520, 720), (520, 400), (820, 400), (820, 720)],
]

# -- Speed violation --
SPEED_LIMIT_KMH = 60
PIXELS_PER_METER = 8.0       # calibration factor (scene-dependent)
FPS = 30                     # default video FPS

# ──────────────────────────────────────────────
# Violation Cooldown
# ──────────────────────────────────────────────
VIOLATION_COOLDOWN_SECONDS = 5   # same vehicle won't be flagged twice within this window

# ──────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────
DEFAULT_CAMERA_ID = "CAM_001"
DEFAULT_LOCATION = "Main St & 1st Ave"

# ──────────────────────────────────────────────
# Performance
# ──────────────────────────────────────────────
FRAME_SKIP = 2                   # process every Nth frame (1 = no skip)
INFERENCE_SIZE = 640             # YOLO input size
try:
    import torch
    USE_GPU = torch.cuda.is_available()
except ImportError:
    USE_GPU = False

# ──────────────────────────────────────────────
# API
# ──────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000

# ──────────────────────────────────────────────
# Heatmap
# ──────────────────────────────────────────────
HEATMAP_WIDTH = 1280
HEATMAP_HEIGHT = 720
HEATMAP_RADIUS = 40              # Gaussian kernel radius
