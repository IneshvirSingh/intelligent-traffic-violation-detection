"""
Visualization utilities for drawing overlays on video frames.
Provides helpers for bounding boxes, track IDs, lane boundaries,
stop-line markers, and violation alert banners.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# ── Color palette (BGR) ──────────────────────
COLOR_GREEN = (0, 220, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 230, 255)
COLOR_BLUE = (255, 160, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ORANGE = (0, 140, 255)
COLOR_CYAN = (255, 255, 0)

VIOLATION_COLOR = COLOR_RED
NORMAL_COLOR = COLOR_GREEN
TRACK_COLOR = COLOR_CYAN


def draw_bounding_box(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    label: str = "",
    color: Tuple[int, int, int] = NORMAL_COLOR,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a labeled bounding box on the frame."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if label:
        font_scale = 0.55
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_WHITE, 1, cv2.LINE_AA,
        )
    return frame


def draw_track_id(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    track_id: int,
    color: Tuple[int, int, int] = TRACK_COLOR,
) -> np.ndarray:
    """Draw a track ID label above the bounding box."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    label = f"ID:{track_id}"
    cv2.putText(
        frame, label, (x1, y1 - 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
    )
    return frame


def draw_stop_line(
    frame: np.ndarray,
    y: int,
    x_range: Tuple[int, int],
    color: Tuple[int, int, int] = COLOR_RED,
    thickness: int = 3,
) -> np.ndarray:
    """Draw the stop-line as a horizontal line across the frame."""
    cv2.line(frame, (x_range[0], y), (x_range[1], y), color, thickness)
    cv2.putText(
        frame, "STOP LINE", (x_range[0] + 5, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
    )
    return frame


def draw_lane_polygons(
    frame: np.ndarray,
    lanes: List[List[Tuple[int, int]]],
    color: Tuple[int, int, int] = COLOR_BLUE,
    alpha: float = 0.15,
) -> np.ndarray:
    """Draw semi-transparent lane polygon overlays."""
    overlay = frame.copy()
    for lane in lanes:
        pts = np.array(lane, dtype=np.int32)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(frame, [pts], True, color, 2)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_violation_alert(
    frame: np.ndarray,
    text: str,
    position: Tuple[int, int] = (10, 40),
    color: Tuple[int, int, int] = VIOLATION_COLOR,
) -> np.ndarray:
    """Draw a prominent violation alert banner on the frame."""
    (tw, th), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
    )
    x, y = position
    cv2.rectangle(frame, (x - 5, y - th - 10), (x + tw + 10, y + 5), (0, 0, 0), -1)
    cv2.rectangle(frame, (x - 5, y - th - 10), (x + tw + 10, y + 5), color, 2)
    cv2.putText(
        frame, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA,
    )
    return frame


def draw_info_bar(
    frame: np.ndarray,
    camera_id: str,
    frame_count: int,
    violation_count: int,
    fps: float = 0.0,
) -> np.ndarray:
    """Draw an info bar at the top of the frame with camera and stats."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 35), (30, 30, 30), -1)
    info = f"Camera: {camera_id}  |  Frame: {frame_count}  |  Violations: {violation_count}  |  FPS: {fps:.1f}"
    cv2.putText(
        frame, info, (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA,
    )
    return frame


def annotate_frame(
    frame: np.ndarray,
    detections: list,
    tracks: list,
    violations: list,
    stop_line_y: int,
    stop_line_x_range: Tuple[int, int],
    lanes: List[List[Tuple[int, int]]],
    camera_id: str = "",
    frame_count: int = 0,
    fps: float = 0.0,
) -> np.ndarray:
    """
    Full annotation pipeline for a single frame.
    Draws lanes, stop-line, tracked boxes, and violation alerts.
    """
    # Draw lanes and stop-line
    frame = draw_lane_polygons(frame, lanes)
    frame = draw_stop_line(frame, stop_line_y, stop_line_x_range)

    # Draw tracked objects
    for track in tracks:
        bbox = track["bbox"]
        track_id = track["track_id"]
        class_name = track.get("class_name", "vehicle")
        is_violating = any(
            v["track_id"] == track_id for v in violations
        )
        color = VIOLATION_COLOR if is_violating else NORMAL_COLOR
        label = f"{class_name} #{track_id}"
        frame = draw_bounding_box(frame, bbox, label, color)

    # Draw violation alerts
    alert_y = 50
    for v in violations:
        text = f"⚠ {v['type'].upper()}: Vehicle #{v['track_id']}"
        frame = draw_violation_alert(frame, text, (10, alert_y))
        alert_y += 45

    # Info bar
    frame = draw_info_bar(frame, camera_id, frame_count, len(violations), fps)

    return frame
