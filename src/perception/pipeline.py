"""Unified perception pipeline combining vehicle detection and Egyptian plate OCR."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence, TypedDict

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.perception.detector import VehicleDetection, VehicleDetector
    from src.perception.ocr_reader import EgyptianPlateReader
else:
    from .detector import VehicleDetection, VehicleDetector
    from .ocr_reader import EgyptianPlateReader


class PerceptionResult(TypedDict):
    """Unified output payload returned by PerceptionPipeline.process_frame."""

    is_emergency: bool
    plate_text: str | None
    confidence: float


class PerceptionPipeline:
    """Pipeline Pattern wrapper chaining VehicleDetector and EgyptianPlateReader."""

    def __init__(
        self,
        detector: VehicleDetector | None = None,
        ocr_reader: EgyptianPlateReader | None = None,
    ) -> None:
        """Initialize or inject vehicle detection and LPR OCR components."""
        self.detector = detector if detector is not None else VehicleDetector()
        self.ocr_reader = (
            ocr_reader if ocr_reader is not None else EgyptianPlateReader()
        )

    @staticmethod
    def _extract_plate_candidate(vehicle_crop: np.ndarray) -> np.ndarray:
        """Focus on the lower vehicle bumper region where license plates typically sit."""
        height = vehicle_crop.shape[0]
        if height < 40:
            return vehicle_crop
        start_y = int(height * 0.45)
        return np.ascontiguousarray(vehicle_crop[start_y:, :].copy())

    def localize_plate_crop(
        self,
        frame: np.ndarray,
        vehicle_bbox: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Crop a vehicle ROI and localize the tightest Egyptian license plate region inside it."""
        vehicle_crop = self.detector.crop_region(frame, vehicle_bbox)
        if hasattr(self.detector, "detect_plate_bbox"):
            local_plate_bbox = self.detector.detect_plate_bbox(vehicle_crop)
            plate_crop = self.detector.crop_region(vehicle_crop, local_plate_bbox)
        else:
            height, width = vehicle_crop.shape[:2]
            start_y = int(height * 0.45) if height >= 40 else 0
            local_plate_bbox = [0, start_y, width, height]
            plate_crop = self._extract_plate_candidate(vehicle_crop)

        frame_plate_bbox = VehicleDetector.translate_bbox_to_frame(
            vehicle_bbox,
            local_plate_bbox,
            frame.shape,
        )
        return vehicle_crop, plate_crop, frame_plate_bbox

    def process_frame(self, frame: np.ndarray) -> PerceptionResult:
        """Detect vehicles, fast-track emergencies, and extract Egyptian plate text."""
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim < 2
        ):
            raise ValueError("frame must be a non-empty NumPy BGR image array.")

        # Stage 1: Vehicle and emergency detection
        detections: list[VehicleDetection] = self.detector.detect_vehicles(frame)
        if not detections:
            return {
                "is_emergency": False,
                "plate_text": None,
                "confidence": 0.0,
            }

        # Stage 2: Zero-latency emergency bypass (emergency detections are sorted first)
        primary = detections[0]
        if primary["is_emergency"]:
            return {
                "is_emergency": True,
                "plate_text": None,
                "confidence": float(primary["confidence"]),
            }

        # Stage 3: Two-stage plate localization + Egyptian license plate OCR
        for det in detections:
            vehicle_crop, plate_crop, _ = self.localize_plate_crop(frame, det["bbox"])

            plate_text, ocr_conf = self.ocr_reader.read_plate(plate_crop)
            if plate_text is None and plate_crop.shape != vehicle_crop.shape:
                plate_text, ocr_conf = self.ocr_reader.read_plate(vehicle_crop)

            if plate_text is not None:
                return {
                    "is_emergency": False,
                    "plate_text": plate_text,
                    "confidence": float(ocr_conf),
                }

        return {
            "is_emergency": False,
            "plate_text": None,
            "confidence": 0.0,
        }

    def __call__(self, frame: np.ndarray) -> PerceptionResult:
        """Allow invoking the pipeline instance directly on a frame."""
        return self.process_frame(frame)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        input_frame = cv2.imread(str(img_path))
        if input_frame is None:
            raise FileNotFoundError(f"Unable to read image at: {img_path}")
    else:
        input_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    pipeline = PerceptionPipeline()
    payload = pipeline.process_frame(input_frame)
    print(f"Pipeline output payload: {payload}")
    print("PerceptionPipeline verification completed without errors.")

