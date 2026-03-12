"""
Unit tests for the violation detection modules.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from violations.red_light_violation import RedLightViolationDetector
from violations.lane_violation import LaneViolationDetector
from violations.speed_violation import SpeedViolationDetector


class MockTrack:
    """Lightweight mock for Track objects."""
    def __init__(self, track_id, center, bbox=None, class_name="car",
                 confidence=0.9, speed_pixels=0.0):
        self.track_id = track_id
        self._center = center
        self.bbox = bbox or (
            center[0] - 30, center[1] - 20,
            center[0] + 30, center[1] + 20,
        )
        self.class_name = class_name
        self.confidence = confidence
        self.speed_pixels = speed_pixels

    @property
    def center(self):
        return self._center


# ── Red Light Tests ──────────────────────────────

def test_red_light_no_violation_when_green():
    detector = RedLightViolationDetector(stop_line_y=400, stop_line_x_range=(100, 1100))
    detector.signal_state = "GREEN"
    track = MockTrack(1, (500, 410))
    assert detector.check(track) is None


def test_red_light_violation_on_crossing():
    detector = RedLightViolationDetector(stop_line_y=400, stop_line_x_range=(100, 1100))
    detector.signal_state = "RED"

    # Frame 1: before stop line
    track1 = MockTrack(1, (500, 390))
    assert detector.check(track1) is None

    # Frame 2: after stop line
    track2 = MockTrack(1, (500, 410))
    result = detector.check(track2)
    assert result is not None
    assert result["type"] == "red_light"
    assert result["track_id"] == 1


def test_red_light_no_violation_outside_x_range():
    detector = RedLightViolationDetector(stop_line_y=400, stop_line_x_range=(100, 500))
    detector.signal_state = "RED"

    track1 = MockTrack(1, (600, 390))
    detector.check(track1)
    track2 = MockTrack(1, (600, 410))
    assert detector.check(track2) is None  # Outside x-range


# ── Lane Violation Tests ─────────────────────────

def test_lane_no_violation_when_inside():
    lanes = [[(0, 0), (0, 100), (100, 100), (100, 0)]]
    detector = LaneViolationDetector(lanes=lanes)
    track = MockTrack(1, (50, 50))
    assert detector.check(track) is None


def test_lane_violation_when_leaving():
    lanes = [[(0, 0), (0, 100), (100, 100), (100, 0)]]
    detector = LaneViolationDetector(lanes=lanes)

    # First inside
    track_in = MockTrack(1, (50, 50))
    detector.check(track_in)

    # Then outside
    track_out = MockTrack(1, (200, 200))
    result = detector.check(track_out)
    assert result is not None
    assert result["type"] == "lane_violation"


def test_lane_no_violation_if_never_inside():
    lanes = [[(0, 0), (0, 100), (100, 100), (100, 0)]]
    detector = LaneViolationDetector(lanes=lanes)
    # Never was inside, so no "leaving" event
    track = MockTrack(1, (200, 200))
    assert detector.check(track) is None


# ── Speed Violation Tests ────────────────────────

def test_speed_no_violation_under_limit():
    detector = SpeedViolationDetector(speed_limit=60, pixels_per_meter=8, fps=30)
    track = MockTrack(1, (100, 100), speed_pixels=2.0)
    assert detector.check(track) is None  # 2px → low speed


def test_speed_violation_over_limit():
    detector = SpeedViolationDetector(speed_limit=60, pixels_per_meter=8, fps=30)
    # 100 pixels/frame → 100/8 = 12.5 m/frame → 12.5*30 = 375 m/s → 1350 km/h
    track = MockTrack(1, (100, 100), speed_pixels=100.0)
    result = detector.check(track)
    assert result is not None
    assert result["type"] == "speed_violation"
    assert result["speed_kmh"] > 60


def test_speed_only_flags_once():
    detector = SpeedViolationDetector(speed_limit=60, pixels_per_meter=8, fps=30)
    track = MockTrack(1, (100, 100), speed_pixels=100.0)
    result1 = detector.check(track)
    assert result1 is not None
    result2 = detector.check(track)
    assert result2 is None  # Already flagged


def test_cleanup():
    detector = SpeedViolationDetector(speed_limit=60, pixels_per_meter=8, fps=30)
    track = MockTrack(1, (100, 100), speed_pixels=100.0)
    detector.check(track)
    detector.cleanup(active_ids=set())  # Track 1 no longer active
    # After cleanup, it should be able to flag again (new track with same ID)
    result = detector.check(track)
    assert result is not None


if __name__ == "__main__":
    test_red_light_no_violation_when_green()
    test_red_light_violation_on_crossing()
    test_red_light_no_violation_outside_x_range()
    test_lane_no_violation_when_inside()
    test_lane_violation_when_leaving()
    test_lane_no_violation_if_never_inside()
    test_speed_no_violation_under_limit()
    test_speed_violation_over_limit()
    test_speed_only_flags_once()
    test_cleanup()
    print("All violation tests passed! ✓")
