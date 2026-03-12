"""
Speed Violation Detector.

Estimates vehicle speed from frame-to-frame pixel displacement
and flags vehicles exceeding the configured speed limit.
"""

from typing import Optional, Dict, Any
from utils.config import SPEED_LIMIT_KMH, PIXELS_PER_METER, FPS


class SpeedViolationDetector:
    """
    Estimates vehicle speed based on centroid displacement between
    consecutive frames.

    Speed formula:
        displacement_pixels / PIXELS_PER_METER = displacement_meters
        displacement_meters * FPS = speed_m_per_s
        speed_m_per_s * 3.6 = speed_km_per_h

    This is a simplified estimation — accuracy depends on camera
    calibration (pixels-per-meter ratio).
    """

    def __init__(
        self,
        speed_limit: float = SPEED_LIMIT_KMH,
        pixels_per_meter: float = PIXELS_PER_METER,
        fps: float = FPS,
    ):
        self.speed_limit = speed_limit
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self._flagged: Dict[int, bool] = {}

    def _estimate_speed_kmh(self, speed_pixels: float) -> float:
        """Convert pixel displacement per frame to km/h."""
        if self.pixels_per_meter <= 0:
            return 0.0
        speed_meters_per_frame = speed_pixels / self.pixels_per_meter
        speed_meters_per_sec = speed_meters_per_frame * self.fps
        speed_kmh = speed_meters_per_sec * 3.6
        return speed_kmh

    def check(self, track) -> Optional[Dict[str, Any]]:
        """
        Check if a tracked vehicle exceeds the speed limit.

        Args:
            track: Track object with .track_id, .speed_pixels, .center,
                   .bbox, .class_name

        Returns:
            Violation dict if speeding detected, else None.
        """
        track_id = track.track_id
        speed_px = track.speed_pixels
        speed_kmh = self._estimate_speed_kmh(speed_px)

        # Only flag once per vehicle (avoid flooding)
        if speed_kmh > self.speed_limit and not self._flagged.get(track_id, False):
            self._flagged[track_id] = True
            cx, cy = track.center
            x1, y1, x2, y2 = track.bbox
            return {
                "type": "speed_violation",
                "track_id": track_id,
                "class_name": track.class_name,
                "location": (float(cx), float(cy)),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "confidence": track.confidence,
                "speed_kmh": round(speed_kmh, 1),
                "speed_limit": self.speed_limit,
                "details": f"Vehicle #{track_id} speed={speed_kmh:.1f} km/h exceeds limit={self.speed_limit} km/h",
            }

        return None

    def cleanup(self, active_ids: set):
        """Remove state for tracks that are no longer active."""
        self._flagged = {
            tid: v for tid, v in self._flagged.items()
            if tid in active_ids
        }
