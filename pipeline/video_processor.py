"""
Video Processor — Main Processing Pipeline.

Reads frames from video sources, runs the detection → tracking →
violation pipeline, and manages multi-camera support via threading.
"""

import cv2
import time
import logging
import threading
from queue import Queue
from typing import Optional, Callable

from detection.vehicle_detector import VehicleDetector
from detection.plate_detector import PlateDetector
from tracking.tracker import ByteTracker
from pipeline.event_engine import EventEngine
from utils.config import (
    FRAME_SKIP,
    DEFAULT_CAMERA_ID,
    DEFAULT_LOCATION,
    TRACK_HIGH_THRESH,
    TRACK_LOW_THRESH,
    TRACK_MATCH_THRESH,
    MAX_TIME_LOST,
    FPS,
)
from utils.visualization import annotate_frame
from utils.config import STOP_LINE_Y, STOP_LINE_X_RANGE, ALLOWED_LANES

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Main video processing pipeline.

    Pipeline per frame:
        1. Read frame from source
        2. Run YOLOv8 vehicle detection
        3. Update ByteTracker with detections
        4. Run violation detection engine
        5. Save violations to DB (via callback)
        6. Annotate frame for display
    """

    def __init__(
        self,
        video_source: str,
        camera_id: str = DEFAULT_CAMERA_ID,
        location: str = DEFAULT_LOCATION,
        frame_skip: int = FRAME_SKIP,
        show: bool = False,
        on_violation: Optional[Callable] = None,
    ):
        self.video_source = video_source
        self.camera_id = camera_id
        self.location = location
        self.frame_skip = max(1, frame_skip)
        self.show = show
        self.on_violation = on_violation  # callback(violation_dict)

        # Pipeline components
        self.detector = VehicleDetector()
        self.plate_detector = PlateDetector()
        self.tracker = ByteTracker(
            high_thresh=TRACK_HIGH_THRESH,
            low_thresh=TRACK_LOW_THRESH,
            match_thresh=TRACK_MATCH_THRESH,
            max_age=MAX_TIME_LOST,
        )
        self.event_engine = EventEngine(fps=FPS)

        # State
        self._running = False
        self._frame_count = 0
        self._violation_count = 0
        self._fps = 0.0

    def run(self):
        """Start processing the video feed."""
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logger.error(f"Cannot open video source: {self.video_source}")
            return

        # Try to get FPS from video
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps > 0:
            self.event_engine.speed.fps = video_fps

        self._running = True
        frame_time = time.time()

        logger.info(
            f"Starting pipeline for camera={self.camera_id}, "
            f"source={self.video_source}"
        )

        try:
            while self._running:
                ret, frame = cap.read()
                if not ret:
                    logger.info("End of video stream.")
                    break

                self._frame_count += 1

                # Frame skipping for performance
                if self._frame_count % self.frame_skip != 0:
                    continue

                # ── 1. Detection ──
                detections = self.detector.detect(frame)

                # ── 2. Tracking ──
                tracks = self.tracker.update(detections)

                # ── 3. Violation Detection ──
                track_dicts = [t.to_dict() for t in tracks]
                violations = self.event_engine.process_tracks(
                    tracks, frame, self.camera_id
                )

                # ── 4. Handle violations ──
                for v in violations:
                    self._violation_count += 1
                    v["location_name"] = self.location
                    logger.warning(f"VIOLATION: {v['details']}")

                    # Try plate detection on violation vehicles
                    if v.get("bbox"):
                        plate_img = self.plate_detector.detect_plate(
                            frame, v["bbox"]
                        )
                        if plate_img is not None:
                            plate_path = self.plate_detector.save_plate_image(
                                plate_img, v["track_id"]
                            )
                            v["plate_image_path"] = plate_path

                    # Callback to save to DB
                    if self.on_violation:
                        self.on_violation(v)

                # ── 5. FPS calculation ──
                now = time.time()
                elapsed = now - frame_time
                self._fps = 1.0 / elapsed if elapsed > 0 else 0.0
                frame_time = now

                # ── 6. Visualization ──
                if self.show:
                    annotated = annotate_frame(
                        frame.copy(),
                        detections=[],
                        tracks=track_dicts,
                        violations=violations,
                        stop_line_y=STOP_LINE_Y,
                        stop_line_x_range=STOP_LINE_X_RANGE,
                        lanes=ALLOWED_LANES,
                        camera_id=self.camera_id,
                        frame_count=self._frame_count,
                        fps=self._fps,
                    )
                    cv2.imshow(f"Camera: {self.camera_id}", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User pressed 'q' — stopping.")
                        break

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt — stopping pipeline.")
        finally:
            cap.release()
            if self.show:
                cv2.destroyAllWindows()
            self._running = False

        logger.info(
            f"Pipeline finished. Frames: {self._frame_count}, "
            f"Violations: {self._violation_count}"
        )

    def stop(self):
        """Signal the pipeline to stop."""
        self._running = False

    @property
    def stats(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "frames_processed": self._frame_count,
            "violations_detected": self._violation_count,
            "current_fps": round(self._fps, 1),
        }


class MultiCameraProcessor:
    """
    Manages multiple VideoProcessor instances across threads.

    Each camera runs in its own thread with an independent pipeline.
    Violations are collected via a shared callback.
    """

    def __init__(self, on_violation: Optional[Callable] = None):
        self._processors: dict = {}
        self._threads: dict = {}
        self.on_violation = on_violation

    def add_camera(
        self,
        video_source: str,
        camera_id: str,
        location: str = "",
        frame_skip: int = FRAME_SKIP,
        show: bool = False,
    ):
        """Register a camera source."""
        processor = VideoProcessor(
            video_source=video_source,
            camera_id=camera_id,
            location=location,
            frame_skip=frame_skip,
            show=show,
            on_violation=self.on_violation,
        )
        self._processors[camera_id] = processor

    def start_all(self):
        """Start all camera processors in separate threads."""
        for cam_id, proc in self._processors.items():
            thread = threading.Thread(
                target=proc.run, name=f"cam-{cam_id}", daemon=True
            )
            self._threads[cam_id] = thread
            thread.start()
            logger.info(f"Started camera thread: {cam_id}")

    def stop_all(self):
        """Stop all camera processors."""
        for proc in self._processors.values():
            proc.stop()
        for thread in self._threads.values():
            thread.join(timeout=5)

    def get_stats(self) -> dict:
        """Get stats for all cameras."""
        return {
            cam_id: proc.stats
            for cam_id, proc in self._processors.items()
        }
