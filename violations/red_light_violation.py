"""
Red Light Violation Detector.

Checks whether a tracked vehicle crosses a defined stop-line
while the traffic signal state is RED.
"""

from typing import Optional, Dict, Any
from utils.config import STOP_LINE_Y, STOP_LINE_X_RANGE, DEFAULT_SIGNAL_STATE


class RedLightViolationDetector:
    """
    Detects red-light violations by monitoring when a vehicle's centroid
    crosses below the stop-line y-coordinate while the signal is RED.

    The detector tracks previous centroid positions to determine
    crossing direction (vehicles moving downward past the line).
    """

    def __init__(
        self,
        stop_line_y: int = STOP_LINE_Y,
        stop_line_x_range: tuple = STOP_LINE_X_RANGE,
    ):
        self.stop_line_y = stop_line_y
        self.x_start, self.x_end = stop_line_x_range
        self._signal_state: str = DEFAULT_SIGNAL_STATE
        self._previous_positions: Dict[int, float] = {}  # track_id → last y

    @property
    def signal_state(self) -> str:
        return self._signal_state

    @signal_state.setter
    def signal_state(self, state: str):
        self._signal_state = state.upper()

    def check(self, track) -> Optional[Dict[str, Any]]:
        """
        Check if a tracked vehicle is committing a red-light violation.

        Args:
            track: Track object with .track_id, .center, .bbox, .class_name

        Returns:
            Violation dict if violation detected, else None.
        """
        if self._signal_state != "RED":
            return None

        cx, cy = track.center
        track_id = track.track_id

        # Check x-range (vehicle must be in the controlled area)
        if not (self.x_start <= cx <= self.x_end):
            self._previous_positions[track_id] = cy
            return None

        prev_y = self._previous_positions.get(track_id)
        self._previous_positions[track_id] = cy

        if prev_y is None:
            return None

        # Vehicle crossed the stop line (moving downward past it)
        if prev_y <= self.stop_line_y < cy:
            x1, y1, x2, y2 = track.bbox
            return {
                "type": "red_light",
                "track_id": track_id,
                "class_name": track.class_name,
                "location": (float(cx), float(cy)),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "confidence": track.confidence,
                "details": f"Vehicle #{track_id} crossed stop-line at y={self.stop_line_y} during RED signal",
            }

        return None

    def cleanup(self, active_ids: set):
        """Remove position history for tracks that are no longer active."""
        self._previous_positions = {
            tid: y for tid, y in self._previous_positions.items()
            if tid in active_ids
        }
