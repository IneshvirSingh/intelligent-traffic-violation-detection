"""
Unit tests for the database module.
"""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Use a temporary database for testing
import utils.config as config
_original_db = config.DB_PATH
_tmpdir = tempfile.mkdtemp()
config.DB_PATH = os.path.join(_tmpdir, "test_violations.db")

from backend.models import Base
from backend import database as db_mod
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Override the engine/session for tests
_test_engine = create_engine(
    f"sqlite:///{config.DB_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},
)
db_mod._engine = _test_engine
db_mod.SessionLocal = sessionmaker(bind=_test_engine, autocommit=False, autoflush=False)

from backend.database import (
    create_tables,
    add_violation,
    get_violations,
    get_violation_by_id,
    get_violation_count,
    get_violations_by_type,
    get_violations_by_camera,
    get_violations_by_hour,
    get_violation_locations,
)


def setup():
    """Initialize test database."""
    create_tables()


def test_add_and_retrieve_violation():
    create_tables()
    violation_data = {
        "track_id": 1,
        "type": "red_light",
        "camera_id": "CAM_001",
        "location": (500.0, 400.0),
        "location_name": "Main St & 1st Ave",
        "confidence": 0.92,
        "class_name": "car",
        "details": "Vehicle #1 crossed stop-line at y=400 during RED signal",
    }
    v = add_violation(violation_data)
    assert v.id is not None
    assert v.vehicle_id == 1
    assert v.violation_type == "red_light"


def test_get_violations():
    violations = get_violations(limit=10)
    assert isinstance(violations, list)
    assert len(violations) >= 1


def test_get_violation_by_id():
    v = get_violation_by_id(1)
    assert v is not None
    assert v["vehicle_id"] == 1


def test_get_violation_count():
    count = get_violation_count()
    assert count >= 1


def test_get_violations_by_type():
    by_type = get_violations_by_type()
    assert "red_light" in by_type
    assert by_type["red_light"] >= 1


def test_get_violations_by_camera():
    by_camera = get_violations_by_camera()
    assert "CAM_001" in by_camera


def test_get_violations_by_hour():
    by_hour = get_violations_by_hour()
    assert isinstance(by_hour, dict)


def test_get_violation_locations():
    locations = get_violation_locations()
    assert isinstance(locations, list)
    assert len(locations) >= 1
    assert "x" in locations[0]
    assert "y" in locations[0]


def test_multiple_violations():
    for i in range(5):
        add_violation({
            "track_id": 10 + i,
            "type": "lane_violation" if i % 2 == 0 else "speed_violation",
            "camera_id": "CAM_002",
            "location": (300.0 + i * 10, 500.0 + i * 5),
            "speed_kmh": 80.0 + i * 10 if i % 2 != 0 else None,
            "confidence": 0.85,
            "class_name": "truck",
            "details": f"Test violation {i}",
        })
    total = get_violation_count()
    assert total >= 6  # 1 from first test + 5 new

    # Check filtering
    lane = get_violations(violation_type="lane_violation")
    assert len(lane) >= 1

    speed = get_violations(violation_type="speed_violation")
    assert len(speed) >= 1


def cleanup():
    """Remove test database."""
    import shutil
    config.DB_PATH = _original_db
    try:
        shutil.rmtree(_tmpdir)
    except Exception:
        pass


if __name__ == "__main__":
    setup()
    test_add_and_retrieve_violation()
    test_get_violations()
    test_get_violation_by_id()
    test_get_violation_count()
    test_get_violations_by_type()
    test_get_violations_by_camera()
    test_get_violations_by_hour()
    test_get_violation_locations()
    test_multiple_violations()
    cleanup()
    print("All database tests passed! ✓")
