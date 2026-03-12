"""
Lane Violation Detector.

Checks whether a tracked vehicle's centroid exits the set of
allowed-lane polygon regions.
"""

import cv2
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from utils.config import ALLOWED_LANES


class LaneViolationDetector:
    """
    Detects lane violations by checking if a vehicle centroid
    is outside all defined allowed-lane polygons.

    Uses OpenCV's pointPolygonTest for robust inside/outside checks.
    """

    def __init__(self, lanes: Optional[List[List[Tuple[int, int]]]] = None):
        self.lanes = lanes or ALLOWED_LANES
        self._lane_contours = [
            np.array(lane, dtype=np.int32) for lane in self.lanes
        ]
        # Track which vehicles have already been inside a lane
        self._was_inside: Dict[int, bool] = {}

    def _is_inside_any_lane(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside any of the allowed lane polygons."""
        pt = (float(point[0]), float(point[1]))
        for contour in self._lane_contours:
            dist = cv2.pointPolygonTest(contour, pt, measureDist=False)
            if dist >= 0:  # inside or on edge
                return True
        return False

    def check(self, track) -> Optional[Dict[str, Any]]:
        """
        Check if a tracked vehicle is committing a lane violation.

        A lane violation is flagged when a vehicle that was previously
        inside an allowed lane moves outside of all lanes.

        Args:
            track: Track object with .track_id, .center, .bbox, .class_name

        Returns:
            Violation dict if violation detected, else None.
        """
        cx, cy = track.center
        track_id = track.track_id
        inside = self._is_inside_any_lane((cx, cy))

        was_inside = self._was_inside.get(track_id, False)

        if inside:
            self._was_inside[track_id] = True
            return None

        # Vehicle is outside — only flag if it was previously inside
        if was_inside:
            self._was_inside[track_id] = False  # Reset to avoid duplicate flags
            x1, y1, x2, y2 = track.bbox
            return {
                "type": "lane_violation",
                "track_id": track_id,
                "class_name": track.class_name,
                "location": (float(cx), float(cy)),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "confidence": track.confidence,
                "details": f"Vehicle #{track_id} left allowed lane region at ({cx:.0f}, {cy:.0f})",
            }

        return None

    def cleanup(self, active_ids: set):
        """Remove state for tracks that are no longer active."""
        self._was_inside = {
            tid: v for tid, v in self._was_inside.items()
            if tid in active_ids
        }
