"""
Database access layer for the Traffic Violation Detection System.

Provides session management and CRUD operations for violation records.
Uses SQLite for zero-configuration deployment.
"""

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker, Session

from backend.models import Base, Violation
from utils.config import DB_PATH


# ── Engine & Session Factory ──────────────────────

_engine = create_engine(
    f"sqlite:///{DB_PATH}",
    echo=False,
    connect_args={"check_same_thread": False},  # For multi-thread access
)

SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)


def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(_engine)


def get_session() -> Session:
    """Get a new database session."""
    return SessionLocal()


# ── CRUD Operations ──────────────────────────────

def add_violation(violation_data: Dict[str, Any]) -> Violation:
    """
    Insert a new violation record into the database.

    Args:
        violation_data: Dict with keys matching Violation model fields.

    Returns:
        The created Violation ORM instance.
    """
    session = get_session()
    try:
        # Parse timestamp if string
        ts = violation_data.get("timestamp")
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                ts = datetime.now(timezone.utc)
        elif ts is None:
            ts = datetime.now(timezone.utc)

        location = violation_data.get("location", (None, None))
        if isinstance(location, (list, tuple)) and len(location) == 2:
            loc_x, loc_y = location
        else:
            loc_x = violation_data.get("location_x")
            loc_y = violation_data.get("location_y")

        violation = Violation(
            vehicle_id=violation_data.get("track_id", 0),
            violation_type=violation_data.get("type", "unknown"),
            timestamp=ts,
            camera_id=violation_data.get("camera_id", ""),
            evidence_image_path=violation_data.get("evidence_path"),
            plate_image_path=violation_data.get("plate_image_path"),
            location_x=float(loc_x) if loc_x is not None else None,
            location_y=float(loc_y) if loc_y is not None else None,
            location_name=violation_data.get("location_name"),
            speed=violation_data.get("speed_kmh"),
            confidence=violation_data.get("confidence"),
            class_name=violation_data.get("class_name"),
            details=violation_data.get("details"),
        )
        session.add(violation)
        session.commit()
        session.refresh(violation)
        return violation
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_violations(
    limit: int = 100,
    offset: int = 0,
    violation_type: Optional[str] = None,
    camera_id: Optional[str] = None,
) -> List[dict]:
    """Retrieve violation records with optional filters."""
    session = get_session()
    try:
        query = session.query(Violation)
        if violation_type:
            query = query.filter(Violation.violation_type == violation_type)
        if camera_id:
            query = query.filter(Violation.camera_id == camera_id)
        query = query.order_by(Violation.timestamp.desc())
        query = query.offset(offset).limit(limit)
        return [v.to_dict() for v in query.all()]
    finally:
        session.close()


def get_violation_by_id(violation_id: int) -> Optional[dict]:
    """Retrieve a single violation by its ID."""
    session = get_session()
    try:
        v = session.query(Violation).filter(Violation.id == violation_id).first()
        return v.to_dict() if v else None
    finally:
        session.close()


def get_violation_count() -> int:
    """Get total number of violations."""
    session = get_session()
    try:
        return session.query(func.count(Violation.id)).scalar() or 0
    finally:
        session.close()


def get_violation_locations() -> List[Dict[str, float]]:
    """Get all violation locations for heatmap generation."""
    session = get_session()
    try:
        results = session.query(
            Violation.location_x, Violation.location_y, Violation.violation_type
        ).filter(
            Violation.location_x.isnot(None),
            Violation.location_y.isnot(None),
        ).all()
        return [
            {"x": r.location_x, "y": r.location_y, "type": r.violation_type}
            for r in results
        ]
    finally:
        session.close()


def get_violations_by_type() -> Dict[str, int]:
    """Get violation counts grouped by type."""
    session = get_session()
    try:
        results = session.query(
            Violation.violation_type, func.count(Violation.id)
        ).group_by(Violation.violation_type).all()
        return {r[0]: r[1] for r in results}
    finally:
        session.close()


def get_violations_by_camera() -> Dict[str, int]:
    """Get violation counts grouped by camera."""
    session = get_session()
    try:
        results = session.query(
            Violation.camera_id, func.count(Violation.id)
        ).group_by(Violation.camera_id).all()
        return {r[0]: r[1] for r in results}
    finally:
        session.close()


def get_violations_by_hour() -> Dict[int, int]:
    """Get violation counts grouped by hour of day."""
    session = get_session()
    try:
        violations = session.query(Violation.timestamp).all()
        hourly: Dict[int, int] = {}
        for (ts,) in violations:
            if ts:
                h = ts.hour
                hourly[h] = hourly.get(h, 0) + 1
        return hourly
    finally:
        session.close()
