"""
Unit tests for the ByteTracker module.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tracking.tracker import ByteTracker, _iou, _iou_matrix, KalmanBoxState


class MockDetection:
    """Lightweight mock that mimics the Detection dataclass."""
    def __init__(self, bbox, confidence=0.9, class_id=2, class_name="car"):
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name


def test_iou_identical_boxes():
    box = np.array([100, 100, 200, 200])
    assert abs(_iou(box, box) - 1.0) < 1e-6


def test_iou_no_overlap():
    a = np.array([0, 0, 50, 50])
    b = np.array([100, 100, 150, 150])
    assert _iou(a, b) == 0.0


def test_iou_partial_overlap():
    a = np.array([0, 0, 100, 100])
    b = np.array([50, 50, 150, 150])
    iou = _iou(a, b)
    # Intersection: 50×50=2500, Union: 10000+10000-2500=17500
    assert abs(iou - 2500 / 17500) < 1e-4


def test_iou_matrix():
    a = np.array([[0, 0, 100, 100], [200, 200, 300, 300]])
    b = np.array([[50, 50, 150, 150]])
    mat = _iou_matrix(a, b)
    assert mat.shape == (2, 1)
    assert mat[0, 0] > 0  # overlaps
    assert mat[1, 0] == 0  # no overlap


def test_kalman_box_state():
    state = KalmanBoxState(bbox=np.array([10, 10, 50, 50]))
    assert state.age == 0
    state.predict()
    assert state.age == 1
    state.update(np.array([12, 12, 52, 52]))
    assert state.hits == 1
    assert state.time_since_update == 0


def test_tracker_creates_new_tracks():
    tracker = ByteTracker(high_thresh=0.5, max_age=5)
    dets = [
        MockDetection(bbox=(100, 100, 200, 200), confidence=0.9),
        MockDetection(bbox=(300, 300, 400, 400), confidence=0.8),
    ]
    tracks = tracker.update(dets)
    assert len(tracks) == 2
    ids = {t.track_id for t in tracks}
    assert len(ids) == 2  # unique IDs


def test_tracker_maintains_ids():
    tracker = ByteTracker(high_thresh=0.5, max_age=5)

    # Frame 1
    dets1 = [MockDetection(bbox=(100, 100, 200, 200), confidence=0.9)]
    tracks1 = tracker.update(dets1)
    id1 = tracks1[0].track_id

    # Frame 2 — same position
    dets2 = [MockDetection(bbox=(102, 102, 202, 202), confidence=0.9)]
    tracks2 = tracker.update(dets2)
    assert tracks2[0].track_id == id1  # Same ID maintained


def test_tracker_removes_lost_tracks():
    tracker = ByteTracker(high_thresh=0.5, max_age=3)

    dets = [MockDetection(bbox=(100, 100, 200, 200), confidence=0.9)]
    tracker.update(dets)

    # Update with no detections for max_age + 1 frames
    for _ in range(5):
        tracks = tracker.update([])

    assert len(tracks) == 0


def test_tracker_reset():
    tracker = ByteTracker()
    dets = [MockDetection(bbox=(100, 100, 200, 200), confidence=0.9)]
    tracker.update(dets)
    assert len(tracker.active_tracks) > 0
    tracker.reset()
    assert len(tracker.active_tracks) == 0


if __name__ == "__main__":
    test_iou_identical_boxes()
    test_iou_no_overlap()
    test_iou_partial_overlap()
    test_iou_matrix()
    test_kalman_box_state()
    test_tracker_creates_new_tracks()
    test_tracker_maintains_ids()
    test_tracker_removes_lost_tracks()
    test_tracker_reset()
    print("All tracker tests passed! ✓")
