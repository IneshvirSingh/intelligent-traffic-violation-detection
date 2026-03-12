"""
SQLAlchemy ORM models for the Traffic Violation Detection System.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Text,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Violation(Base):
    """
    Stores a single traffic violation event.

    Fields:
        id:                  Auto-increment primary key.
        vehicle_id:          Tracker-assigned vehicle ID.
        violation_type:      Type of violation (red_light, lane_violation, speed_violation).
        timestamp:           When the violation occurred.
        camera_id:           Which camera captured the violation.
        evidence_image_path: Path to the saved evidence frame.
        plate_image_path:    Path to the cropped license plate image (nullable).
        location_x:          X coordinate where the violation occurred.
        location_y:          Y coordinate where the violation occurred.
        location_name:       Human-readable location name.
        speed:               Estimated speed in km/h (for speed violations).
        confidence:          Detection confidence score.
        class_name:          Vehicle class (car, bus, truck, motorcycle).
        details:             Human-readable description of the violation.
    """
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    vehicle_id = Column(Integer, nullable=False, index=True)
    violation_type = Column(String(50), nullable=False, index=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    camera_id = Column(String(50), nullable=False, index=True)
    evidence_image_path = Column(String(500), nullable=True)
    plate_image_path = Column(String(500), nullable=True)
    location_x = Column(Float, nullable=True)
    location_y = Column(Float, nullable=True)
    location_name = Column(String(200), nullable=True)
    speed = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    class_name = Column(String(50), nullable=True)
    details = Column(Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "vehicle_id": self.vehicle_id,
            "violation_type": self.violation_type,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "camera_id": self.camera_id,
            "evidence_image_path": self.evidence_image_path,
            "plate_image_path": self.plate_image_path,
            "location_x": self.location_x,
            "location_y": self.location_y,
            "location_name": self.location_name,
            "speed": self.speed,
            "confidence": self.confidence,
            "class_name": self.class_name,
            "details": self.details,
        }
