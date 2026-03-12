"""
FastAPI Backend for the Traffic Violation Detection System.

Provides REST endpoints for querying violations, analytics, and heatmaps.
"""

import os
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.database import (
    create_tables,
    get_violations,
    get_violation_by_id,
    get_violation_count,
    get_violations_by_type,
    get_violations_by_camera,
    get_violations_by_hour,
    get_violation_locations,
)
from analytics.heatmap import generate_heatmap
from analytics.statistics import get_summary_statistics
from utils.config import VIOLATION_IMG_DIR, HEATMAP_DIR


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Traffic Violation Detection API",
        description=(
            "REST API for querying traffic violations, analytics, "
            "and heatmap data from the intelligent detection system."
        ),
        version="1.0.0",
    )

    # Ensure DB tables exist
    create_tables()

    # Serve evidence images as static files
    if os.path.isdir(str(VIOLATION_IMG_DIR)):
        app.mount(
            "/evidence",
            StaticFiles(directory=str(VIOLATION_IMG_DIR)),
            name="evidence",
        )
    if os.path.isdir(str(HEATMAP_DIR)):
        app.mount(
            "/heatmap-images",
            StaticFiles(directory=str(HEATMAP_DIR)),
            name="heatmap-images",
        )

    # ── Endpoints ────────────────────────────────────

    @app.get("/")
    async def root():
        """API health check."""
        return {
            "service": "Traffic Violation Detection API",
            "version": "1.0.0",
            "status": "running",
        }

    @app.get("/violations")
    async def list_violations(
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        violation_type: Optional[str] = None,
        camera_id: Optional[str] = None,
    ):
        """
        List violation records with optional filtering.

        Query parameters:
            - limit: Max number of records (default: 50)
            - offset: Pagination offset
            - violation_type: Filter by type (red_light, lane_violation, speed_violation)
            - camera_id: Filter by camera ID
        """
        violations = get_violations(
            limit=limit,
            offset=offset,
            violation_type=violation_type,
            camera_id=camera_id,
        )
        total = get_violation_count()
        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "violations": violations,
        }

    @app.get("/violations/{violation_id}")
    async def get_single_violation(violation_id: int):
        """Retrieve a single violation by its ID."""
        violation = get_violation_by_id(violation_id)
        if violation is None:
            raise HTTPException(status_code=404, detail="Violation not found")
        return violation

    @app.get("/analytics")
    async def analytics():
        """
        Get aggregated analytics data.

        Returns violation counts by type, camera, and hour.
        """
        stats = get_summary_statistics()
        return stats

    @app.get("/heatmap")
    async def heatmap(regenerate: bool = False):
        """
        Get the violation heatmap.

        Query parameters:
            - regenerate: If true, regenerate the heatmap from current data.

        Returns the heatmap image file path and metadata.
        """
        heatmap_path = os.path.join(str(HEATMAP_DIR), "violation_heatmap.png")

        if regenerate or not os.path.exists(heatmap_path):
            locations = get_violation_locations()
            if locations:
                generate_heatmap(locations, output_path=heatmap_path)
            else:
                return {"message": "No violation data available for heatmap"}

        if os.path.exists(heatmap_path):
            return FileResponse(
                heatmap_path,
                media_type="image/png",
                filename="violation_heatmap.png",
            )
        return {"message": "Heatmap not yet generated"}

    @app.get("/cameras")
    async def cameras():
        """
        List cameras and their violation statistics.
        """
        by_camera = get_violations_by_camera()
        camera_list = [
            {"camera_id": cam_id, "violation_count": count}
            for cam_id, count in by_camera.items()
        ]
        return {"cameras": camera_list}

    return app


# Module-level app instance for `uvicorn backend.api:app`
app = create_app()
