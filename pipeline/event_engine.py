"""
Event Engine — Violation Orchestrator.

Coordinates all violation detectors, manages cooldowns to prevent
duplicate flagging, and emits structured violation events.
"""

import time
import cv2
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from violations.red_light_violation import RedLightViolationDetector
from violations.lane_violation import LaneViolationDetector
from violations.speed_violation import SpeedViolationDetector
from utils.config import (
    VIOLATION_COOLDOWN_SECONDS,
    VIOLATION_IMG_DIR,
    STOP_LINE_Y,
    STOP_LINE_X_RANGE,
    ALLOWED_LANES,
    SPEED_LIMIT_KMH,
    PIXELS_PER_METER,
    FPS,
)


class EventEngine:
    """
    Orchestrates violation detection across all checkers.

    Maintains per-vehicle cooldowns so that the same vehicle/violation-type
    pair is not flagged repeatedly within a short window.
    """

    def __init__(self, fps: float = FPS):
        self.red_light = RedLightViolationDetector(
            stop_line_y=STOP_LINE_Y,
            stop_line_x_range=STOP_LINE_X_RANGE,
        )
        self.lane = LaneViolationDetector(lanes=ALLOWED_LANES)
        self.speed = SpeedViolationDetector(
            speed_limit=SPEED_LIMIT_KMH,
            pixels_per_meter=PIXELS_PER_METER,
            fps=fps,
        )

        # Cooldown tracking: (track_id, violation_type) → last_flagged_time
        self._cooldowns: Dict[tuple, float] = {}
        self.cooldown_seconds = VIOLATION_COOLDOWN_SECONDS

    def process_tracks(
        self,
        tracks: list,
        frame=None,
        camera_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Run all violation checkers against the current set of tracks.

        Args:
            tracks: List of Track objects from the tracker.
            frame: Current video frame (for evidence capture).
            camera_id: Identifier for the camera source.

        Returns:
            List of violation event dicts.
        """
        violations = []
        current_time = time.time()
        active_ids = {t.track_id for t in tracks}

        for track in tracks:
            # Check each violation type
            for checker in [self.red_light, self.lane, self.speed]:
                result = checker.check(track)
                if result is None:
                    continue

                key = (result["track_id"], result["type"])

                # Apply cooldown
                last_time = self._cooldowns.get(key, 0)
                if current_time - last_time < self.cooldown_seconds:
                    continue

                self._cooldowns[key] = current_time

                # Capture evidence
                evidence_path = None
                if frame is not None:
                    evidence_path = self._capture_evidence(
                        frame, result, camera_id
                    )

                result["timestamp"] = datetime.now().isoformat()
                result["camera_id"] = camera_id
                result["evidence_path"] = evidence_path
                violations.append(result)

        # Cleanup old state
        self.red_light.cleanup(active_ids)
        self.lane.cleanup(active_ids)
        self.speed.cleanup(active_ids)
        self._cleanup_cooldowns(current_time)

        return violations

    def _capture_evidence(
        self,
        frame,
        violation: Dict[str, Any],
        camera_id: str,
    ) -> Optional[str]:
        """Capture and save evidence image for a violation."""
        import numpy as np

        evidence = frame.copy()
        bbox = violation.get("bbox")
        if bbox:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(evidence, (x1, y1), (x2, y2), (0, 0, 255), 3)

            label = f"{violation['type'].upper()} - Vehicle #{violation['track_id']}"
            cv2.putText(
                evidence, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2,
            )

        # Add timestamp watermark
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(
            evidence, f"{ts} | {camera_id}", (10, evidence.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
        )

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"violation_{violation['type']}_{violation['track_id']}_{timestamp_str}.jpg"
        filepath = os.path.join(str(VIOLATION_IMG_DIR), filename)
        cv2.imwrite(filepath, evidence)
        return filepath

    def _cleanup_cooldowns(self, current_time: float):
        """Remove expired cooldown entries."""
        expired = [
            key for key, t in self._cooldowns.items()
            if current_time - t > self.cooldown_seconds * 3
        ]
        for key in expired:
            del self._cooldowns[key]

    def set_signal_state(self, state: str):
        """Update the traffic signal state for red-light detection."""
        self.red_light.signal_state = state
