# 🚦 Intelligent Traffic Violation Detection System

A production-quality computer vision system that detects traffic violations from video feeds using **YOLOv8**, **ByteTrack** multi-object tracking, and a **FastAPI** REST backend with real-time analytics and heatmap generation.

---

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────────────┐
│  Video Feed │────▶│ Vehicle Detection│────▶│ Multi-Object Tracking│
│  (Camera)   │     │   (YOLOv8)       │     │    (ByteTrack)       │
└─────────────┘     └──────────────────┘     └──────────┬───────────┘
                                                        │
                    ┌──────────────────┐     ┌──────────▼───────────┐
                    │ Evidence Capture │◀────│ Violation Detection  │
                    │ (Frame + Bbox)   │     │  Engine (Event Bus)  │
                    └────────┬─────────┘     └──────────────────────┘
                             │               ┌──────────────────────┐
                    ┌────────▼─────────┐     │    Violation Types   │
                    │  SQLite Database │     │  • Red-light crossing│
                    │  (SQLAlchemy)    │     │  • Lane departure    │
                    └────────┬─────────┘     │  • Speed exceedance  │
                             │               └──────────────────────┘
                    ┌────────▼─────────┐
                    │  FastAPI Backend  │
                    │  + Analytics +    │
                    │  Heatmaps         │
                    └──────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **YOLOv8 Detection** | Detects cars, buses, trucks, and motorcycles with configurable confidence |
| **ByteTrack Tracking** | Persistent vehicle IDs across frames with Kalman prediction |
| **Red-Light Violation** | Detects vehicles crossing a stop-line during a RED signal |
| **Lane Violation** | Flags vehicles leaving allowed-lane polygon regions |
| **Speed Violation** | Estimates speed from pixel displacement; flags over-limit |
| **License Plate Detection** | Crops plate regions with model or heuristic; OCR-ready |
| **Evidence Capture** | Saves annotated violation frames with timestamps |
| **SQLite Database** | Stores all violations with full metadata (SQLAlchemy ORM) |
| **FastAPI REST API** | Query violations, analytics, and heatmaps via JSON endpoints |
| **Heatmap Analytics** | Gaussian-smoothed violation density maps for hotspot identification |
| **Multi-Camera** | Threaded pipeline for simultaneous camera feeds |
| **Real-time Visualization** | Bounding boxes, track IDs, lane overlays, violation alerts |

---

## 📁 Project Structure

```
traffic_violation_detector/
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
├── detection/
│   ├── vehicle_detector.py          # YOLOv8 inference wrapper
│   └── plate_detector.py           # License plate detection + OCR hook
├── tracking/
│   └── tracker.py                   # ByteTrack multi-object tracker
├── violations/
│   ├── red_light_violation.py       # Stop-line crossing detector
│   ├── lane_violation.py            # Lane departure detector
│   └── speed_violation.py           # Speed estimation & violation
├── pipeline/
│   ├── video_processor.py           # Main video processing loop
│   └── event_engine.py              # Violation orchestrator + cooldowns
├── backend/
│   ├── api.py                       # FastAPI REST endpoints
│   ├── database.py                  # SQLAlchemy session + CRUD
│   └── models.py                    # ORM models (Violation table)
├── analytics/
│   ├── statistics.py                # Aggregated violation stats
│   └── heatmap.py                   # Gaussian heatmap generation
├── utils/
│   ├── config.py                    # Central configuration
│   └── visualization.py            # OpenCV drawing utilities
├── tests/
│   ├── test_tracker.py              # Tracker unit tests
│   └── test_violations.py          # Violation detector tests
├── data/
│   └── sample_videos/               # Place input videos here
└── outputs/
    ├── violations/                   # Saved evidence images
    └── heatmaps/                     # Generated heatmap PNGs
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Install

```bash
cd traffic_violation_detector
pip install -r requirements.txt
```

YOLOv8 weights (`yolov8n.pt`) are downloaded automatically on first run.

### Add a sample video

Place a traffic video in:
```
data/sample_videos/traffic.mp4
```

---

## 🎬 Usage

### Process a Video

```bash
# With visualization window
python main.py --video data/sample_videos/traffic.mp4 --show

# Custom camera ID and location
python main.py --video traffic.mp4 --camera-id CAM_002 --location "Main St" --show

# Skip frames for performance
python main.py --video traffic.mp4 --frame-skip 3
```

### Start the REST API

```bash
python main.py --api
```

Opens at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Both simultaneously

```bash
python main.py --video traffic.mp4 --api --show
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `GET` | `/violations` | List violations (paginated, filterable) |
| `GET` | `/violations/{id}` | Get single violation |
| `GET` | `/analytics` | Aggregated statistics |
| `GET` | `/heatmap` | Violation heatmap image |
| `GET` | `/cameras` | Camera list with counts |

### Example

```bash
curl http://localhost:8000/violations?limit=10&violation_type=red_light
```

---

## 🧪 Running Tests

```bash
# With pytest
python -m pytest tests/ -v

# Standalone
python tests/test_tracker.py
python tests/test_violations.py
```

---

## ⚙️ Configuration

All parameters are in `utils/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `DETECTION_CONFIDENCE` | 0.4 | YOLO detection threshold |
| `STOP_LINE_Y` | 400 | Red-light stop-line position |
| `SPEED_LIMIT_KMH` | 60 | Speed violation threshold |
| `PIXELS_PER_METER` | 8.0 | Camera calibration ratio |
| `FRAME_SKIP` | 2 | Process every Nth frame |
| `VIOLATION_COOLDOWN_SECONDS` | 5 | Duplicate suppression window |

---

## 📊 Expected Outputs

After processing a video, the system generates:

1. **Evidence images** in `outputs/violations/` — violation frames with annotated bounding boxes
2. **Heatmaps** in `outputs/heatmaps/` — density maps of violation hotspots
3. **SQLite database** `violations.db` — structured violation records

---

## 🔮 Future Improvements

- **OCR Integration** — Tesseract / EasyOCR for license plate text recognition
- **Real-time Dashboard** — React/Vue frontend with live violation feed
- **Cloud Deployment** — Docker compose with GPU support for production
- **Kafka Streaming** — Event-driven pipeline for high-throughput deployments
- **Edge Deployment** — TensorRT / ONNX optimization for Jetson Nano
- **Traffic Signal Detection** — Automatic RED/GREEN state detection from video
- **Night-mode Enhancement** — Adaptive preprocessing for low-light conditions

---

## 🛠 Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.9+ | Core language |
| YOLOv8 (Ultralytics) | Vehicle detection |
| OpenCV | Frame processing & visualization |
| ByteTrack | Multi-object tracking |
| NumPy / SciPy | Numerical computation |
| FastAPI + Uvicorn | REST API backend |
| SQLAlchemy + SQLite | Database ORM |
| Matplotlib | Heatmap rendering |
| PyTorch | Deep learning backend |

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
