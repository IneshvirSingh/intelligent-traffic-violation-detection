"""
Vehicle Detection module using YOLOv8 (Ultralytics).

Loads a YOLOv8 model, runs inference on individual frames,
and filters detections to vehicle classes (car, bus, truck, motorcycle).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # Graceful fallback for environments without ultralytics

from utils.config import (
    YOLO_MODEL_PATH,
    DETECTION_CONFIDENCE,
    DETECTION_IOU_THRESHOLD,
    VEHICLE_CLASSES,
    INFERENCE_SIZE,
    USE_GPU,
)


@dataclass
class Detection:
    """Represents a single vehicle detection."""
    bbox: tuple           # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> tuple:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def to_tlwh(self) -> tuple:
        """Convert bbox to (top-left-x, top-left-y, width, height)."""
        x1, y1, x2, y2 = self.bbox
        return (x1, y1, x2 - x1, y2 - y1)


class VehicleDetector:
    """
    YOLOv8-based vehicle detector.

    Loads the model once and provides a `detect()` method to run
    inference on individual frames, returning filtered vehicle
    detections.
    """

    def __init__(
        self,
        model_path: str = YOLO_MODEL_PATH,
        confidence: float = DETECTION_CONFIDENCE,
        iou_threshold: float = DETECTION_IOU_THRESHOLD,
        img_size: int = INFERENCE_SIZE,
        use_gpu: bool = USE_GPU,
    ):
        if YOLO is None:
            raise ImportError(
                "ultralytics is not installed. "
                "Install it via: pip install ultralytics"
            )
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = "cuda" if use_gpu else "cpu"

        # Pre-compute the set of vehicle class IDs for fast lookup
        self._vehicle_class_ids = set(VEHICLE_CLASSES.keys())

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv8 inference on a single frame.

        Args:
            frame: BGR image as a numpy array (H, W, 3).

        Returns:
            List of Detection objects filtered to vehicle classes.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )

        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                class_id = int(boxes.cls[i].item())
                if class_id not in self._vehicle_class_ids:
                    continue

                conf = float(boxes.conf[i].item())
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()

                detections.append(
                    Detection(
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        confidence=conf,
                        class_id=class_id,
                        class_name=VEHICLE_CLASSES[class_id],
                    )
                )

        return detections

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        """Run detection on a batch of frames for GPU efficiency."""
        all_detections = []
        results = self.model.predict(
            source=frames,
            conf=self.confidence,
            iou=self.iou_threshold,
            imgsz=self.img_size,
            device=self.device,
            verbose=False,
        )

        for result in results:
            frame_detections: List[Detection] = []
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())
                    if class_id not in self._vehicle_class_ids:
                        continue
                    conf = float(boxes.conf[i].item())
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    frame_detections.append(
                        Detection(
                            bbox=(float(x1), float(y1), float(x2), float(y2)),
                            confidence=conf,
                            class_id=class_id,
                            class_name=VEHICLE_CLASSES[class_id],
                        )
                    )
            all_detections.append(frame_detections)

        return all_detections
