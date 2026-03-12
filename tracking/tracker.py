"""
Multi-Object Tracker — ByteTrack-style implementation.

Provides persistent integer IDs for vehicles across frames using
IoU-based association with a simple Kalman-filter prediction step.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of boxes."""
    n, m = len(boxes_a), len(boxes_b)
    iou_mat = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            iou_mat[i, j] = _iou(boxes_a[i], boxes_b[j])
    return iou_mat


def _greedy_assignment(iou_mat: np.ndarray, threshold: float):
    """Simple greedy IoU-based assignment (no Hungarian needed)."""
    matched_indices = []
    unmatched_a = list(range(iou_mat.shape[0]))
    unmatched_b = list(range(iou_mat.shape[1]))

    if iou_mat.size == 0:
        return matched_indices, unmatched_a, unmatched_b

    # Sort by IoU descending
    flat = iou_mat.flatten()
    sorted_idx = np.argsort(-flat)

    used_a = set()
    used_b = set()

    for idx in sorted_idx:
        i = idx // iou_mat.shape[1]
        j = idx % iou_mat.shape[1]
        if i in used_a or j in used_b:
            continue
        if iou_mat[i, j] < threshold:
            break
        matched_indices.append((i, j))
        used_a.add(i)
        used_b.add(j)

    unmatched_a = [i for i in range(iou_mat.shape[0]) if i not in used_a]
    unmatched_b = [j for j in range(iou_mat.shape[1]) if j not in used_b]

    return matched_indices, unmatched_a, unmatched_b


@dataclass
class KalmanBoxState:
    """Minimal Kalman-like state for a bounding box."""
    bbox: np.ndarray          # [x1, y1, x2, y2]
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))
    age: int = 0
    hits: int = 0
    time_since_update: int = 0

    def predict(self) -> np.ndarray:
        """Predict the next position."""
        self.bbox = self.bbox + self.velocity
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update(self, new_bbox: np.ndarray):
        """Update state with a new measurement."""
        self.velocity = 0.5 * self.velocity + 0.5 * (new_bbox - self.bbox)
        self.bbox = new_bbox
        self.hits += 1
        self.time_since_update = 0


@dataclass
class Track:
    """Represents a tracked object."""
    track_id: int
    state: KalmanBoxState
    class_name: str = "vehicle"
    class_id: int = -1
    confidence: float = 0.0
    is_active: bool = True
    centroid_history: list = field(default_factory=list)

    @property
    def bbox(self) -> np.ndarray:
        return self.state.bbox

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.state.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def speed_pixels(self) -> float:
        """Estimate speed as pixel displacement per frame."""
        if len(self.centroid_history) < 2:
            return 0.0
        c1 = np.array(self.centroid_history[-2])
        c2 = np.array(self.centroid_history[-1])
        return float(np.linalg.norm(c2 - c1))

    def to_dict(self) -> dict:
        x1, y1, x2, y2 = self.state.bbox
        return {
            "track_id": self.track_id,
            "bbox": (float(x1), float(y1), float(x2), float(y2)),
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": self.confidence,
            "center": self.center,
            "speed_pixels": self.speed_pixels,
        }


class ByteTracker:
    """
    ByteTrack-style multi-object tracker.

    Two-stage association:
      1. High-confidence detections matched to existing tracks via IoU.
      2. Low-confidence detections matched to remaining tracks.

    Tracks that aren't matched for `max_age` frames are removed.
    New tracks are created from unmatched high-confidence detections.
    """

    def __init__(
        self,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.3,
        max_age: int = 30,
    ):
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self.max_age = max_age

        self._tracks: List[Track] = []
        self._next_id = 1

    @property
    def active_tracks(self) -> List[Track]:
        return [t for t in self._tracks if t.is_active]

    def update(self, detections: list) -> List[Track]:
        """
        Update tracks with new detections.

        Args:
            detections: List of Detection objects with .bbox, .confidence,
                        .class_id, .class_name attributes.

        Returns:
            List of active Track objects after this update.
        """
        # Predict step for all existing tracks
        for track in self._tracks:
            track.state.predict()

        if not detections:
            self._age_tracks()
            return self.active_tracks

        # Split detections into high and low confidence
        det_bboxes = np.array([d.bbox for d in detections])
        det_scores = np.array([d.confidence for d in detections])

        high_mask = det_scores >= self.high_thresh
        low_mask = (det_scores >= self.low_thresh) & (~high_mask)

        high_indices = np.where(high_mask)[0]
        low_indices = np.where(low_mask)[0]

        # ── First association: high-confidence detections ──
        active = self.active_tracks
        if len(active) > 0 and len(high_indices) > 0:
            track_bboxes = np.array([t.state.bbox for t in active])
            high_det_bboxes = det_bboxes[high_indices]
            iou_mat = _iou_matrix(track_bboxes, high_det_bboxes)
            matched, unmatched_tracks_1, unmatched_dets_1 = _greedy_assignment(
                iou_mat, self.match_thresh
            )

            for t_idx, d_idx in matched:
                det = detections[high_indices[d_idx]]
                active[t_idx].state.update(np.array(det.bbox))
                active[t_idx].confidence = det.confidence
                active[t_idx].class_name = det.class_name
                active[t_idx].class_id = det.class_id
                active[t_idx].centroid_history.append(active[t_idx].center)
                # Keep history bounded
                if len(active[t_idx].centroid_history) > 60:
                    active[t_idx].centroid_history = active[t_idx].centroid_history[-60:]

            remaining_tracks = [active[i] for i in unmatched_tracks_1]
            remaining_high_dets = [high_indices[i] for i in unmatched_dets_1]
        else:
            remaining_tracks = list(active)
            remaining_high_dets = list(high_indices)

        # ── Second association: low-confidence detections ──
        if len(remaining_tracks) > 0 and len(low_indices) > 0:
            track_bboxes = np.array([t.state.bbox for t in remaining_tracks])
            low_det_bboxes = det_bboxes[low_indices]
            iou_mat = _iou_matrix(track_bboxes, low_det_bboxes)
            matched2, unmatched_tracks_2, _ = _greedy_assignment(
                iou_mat, self.match_thresh
            )

            for t_idx, d_idx in matched2:
                det = detections[low_indices[d_idx]]
                remaining_tracks[t_idx].state.update(np.array(det.bbox))
                remaining_tracks[t_idx].confidence = det.confidence
                remaining_tracks[t_idx].centroid_history.append(
                    remaining_tracks[t_idx].center
                )
        else:
            remaining_high_dets = list(remaining_high_dets) if not isinstance(remaining_high_dets, list) else remaining_high_dets

        # ── Create new tracks from unmatched high-confidence detections ──
        for d_idx in remaining_high_dets:
            det = detections[d_idx]
            new_track = Track(
                track_id=self._next_id,
                state=KalmanBoxState(bbox=np.array(det.bbox)),
                class_name=det.class_name,
                class_id=det.class_id,
                confidence=det.confidence,
            )
            new_track.centroid_history.append(new_track.center)
            self._tracks.append(new_track)
            self._next_id += 1

        self._age_tracks()
        return self.active_tracks

    def _age_tracks(self):
        """Deactivate tracks that haven't been updated recently."""
        for track in self._tracks:
            if track.state.time_since_update > self.max_age:
                track.is_active = False

        # Periodically prune dead tracks to free memory
        self._tracks = [
            t for t in self._tracks
            if t.is_active or t.state.time_since_update <= self.max_age * 2
        ]

    def reset(self):
        """Clear all tracks."""
        self._tracks.clear()
        self._next_id = 1
