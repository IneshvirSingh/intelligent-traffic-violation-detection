"""
License Plate Detection module.

Crops vehicle regions from detections and runs a secondary detector
to locate license plates. Provides a hook point for future OCR integration.
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Tuple

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from utils.config import (
    PLATE_MODEL_PATH,
    PLATE_CONFIDENCE,
    VIOLATION_IMG_DIR,
)


class PlateDetector:
    """
    License plate detector.

    Uses a secondary YOLO model (if available) to detect license plates
    within vehicle bounding-box crops. Falls back to a simple region
    heuristic if no model is provided.

    Future enhancement: integrate Tesseract / EasyOCR for plate text extraction.
    """

    def __init__(self, model_path: Optional[str] = PLATE_MODEL_PATH):
        self.model = None
        if model_path and YOLO is not None:
            try:
                self.model = YOLO(model_path)
            except Exception:
                pass  # Plate detection is optional

    def detect_plate(
        self,
        frame: np.ndarray,
        vehicle_bbox: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        """
        Detect and crop the license plate region from a vehicle ROI.

        Args:
            frame: Full frame image (BGR).
            vehicle_bbox: (x1, y1, x2, y2) of the vehicle.

        Returns:
            Cropped plate image (numpy array) or None if not found.
        """
        x1, y1, x2, y2 = [int(v) for v in vehicle_bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        vehicle_crop = frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            return None

        if self.model is not None:
            return self._detect_with_model(vehicle_crop)
        else:
            return self._heuristic_crop(vehicle_crop)

    def _detect_with_model(self, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        """Run YOLO plate detector on the vehicle crop."""
        results = self.model.predict(
            source=vehicle_crop,
            conf=PLATE_CONFIDENCE,
            verbose=False,
        )
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                # Take the highest-confidence plate
                best_idx = result.boxes.conf.argmax()
                px1, py1, px2, py2 = result.boxes.xyxy[best_idx].cpu().numpy()
                px1, py1, px2, py2 = int(px1), int(py1), int(px2), int(py2)
                plate_crop = vehicle_crop[py1:py2, px1:px2]
                if plate_crop.size > 0:
                    return plate_crop
        return None

    def _heuristic_crop(self, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Heuristic: license plates are typically in the bottom-center
        third of the vehicle bounding box.
        """
        vh, vw = vehicle_crop.shape[:2]
        # Bottom 30%, center 60%
        y_start = int(vh * 0.65)
        x_start = int(vw * 0.2)
        x_end = int(vw * 0.8)
        plate_region = vehicle_crop[y_start:vh, x_start:x_end]
        if plate_region.size > 0:
            return plate_region
        return None

    def save_plate_image(
        self,
        plate_img: np.ndarray,
        vehicle_id: int,
        timestamp: Optional[str] = None,
    ) -> str:
        """Save the plate crop to disk and return the file path."""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plate_{vehicle_id}_{timestamp}.jpg"
        filepath = os.path.join(str(VIOLATION_IMG_DIR), filename)
        cv2.imwrite(filepath, plate_img)
        return filepath

    @staticmethod
    def integrate_ocr(plate_img: np.ndarray) -> str:
        """
        Placeholder for OCR integration.

        To integrate OCR, install one of:
            - pytesseract: pip install pytesseract
            - easyocr:     pip install easyocr

        Example with EasyOCR:
            import easyocr
            reader = easyocr.Reader(['en'])
            results = reader.readtext(plate_img)
            text = ' '.join([r[1] for r in results])
            return text

        Example with Tesseract:
            import pytesseract
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray, config='--psm 7')
            return text.strip()
        """
        return "OCR_NOT_CONFIGURED"
