"""
Statistics module for violation analytics.

Queries the database and compiles summary statistics for the API.
"""

from typing import Dict, Any

from backend.database import (
    get_violation_count,
    get_violations_by_type,
    get_violations_by_camera,
    get_violations_by_hour,
)


def get_summary_statistics() -> Dict[str, Any]:
    """
    Compile comprehensive analytics summary.

    Returns:
        Dictionary with total count, breakdown by type,
        by camera, and by hour of day.
    """
    total = get_violation_count()
    by_type = get_violations_by_type()
    by_camera = get_violations_by_camera()
    by_hour = get_violations_by_hour()

    # Format hourly data for chart consumption
    hourly_data = [
        {"hour": h, "count": by_hour.get(h, 0)}
        for h in range(24)
    ]

    # Most common violation type
    most_common = max(by_type, key=by_type.get) if by_type else None

    # Peak hour
    peak_hour = max(by_hour, key=by_hour.get) if by_hour else None

    return {
        "total_violations": total,
        "by_type": by_type,
        "by_camera": by_camera,
        "hourly_distribution": hourly_data,
        "most_common_violation": most_common,
        "peak_hour": peak_hour,
        "summary": {
            "red_light": by_type.get("red_light", 0),
            "lane_violation": by_type.get("lane_violation", 0),
            "speed_violation": by_type.get("speed_violation", 0),
        },
    }
