"""Vehicle and emergency vehicle detection module using YOLOv8 and OpenCV."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Sequence, TypedDict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"
DEFAULT_PLATE_MODEL_PATH = PROJECT_ROOT / "models" / "egyptian_plate_yolov8n.pt"
DEFAULT_TEST_IMAGES_DIR = PROJECT_ROOT / "data" / "test_images"

VEHICLE_CLASSES: frozenset[str] = frozenset({"car", "motorcycle", "bus", "truck"})
EMERGENCY_CLASSES: frozenset[str] = frozenset(
    {
        "ambulance",
        "fire_truck",
        "fire truck",
        "firetruck",
        "police",
        "police_car",
        "police car",
        "emergency",
        "emergency_vehicle",
    }
)
TARGET_CLASSES: frozenset[str] = VEHICLE_CLASSES | EMERGENCY_CLASSES


class VehicleDetection(TypedDict):
    """Structured vehicle detection returned by VehicleDetector."""

    class_name: str
    confidence: float
    bbox: list[int]
    is_emergency: bool


class VehicleDetector:
    """Detects vehicles and identifies emergency vehicles using YOLOv8."""

    ROOF_BAND_RATIO = 0.30
    DUAL_BEACON_MIN_RATIO = 0.015
    LARGE_VEHICLE_WHITE_MIN_RATIO = 0.35
    SINGLE_BEACON_MIN_RATIO = 0.035

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        plate_model_path: str | Path = DEFAULT_PLATE_MODEL_PATH,
        conf_threshold: float = 0.25,
        plate_conf_threshold: float = 0.45,
        device: str | None = None,
        enable_emergency_heuristic: bool = True,
    ) -> None:
        """Load vehicle and custom plate YOLO models and select CUDA GPU when available."""
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.conf_threshold = conf_threshold
        self.plate_conf_threshold = float(plate_conf_threshold)
        self.enable_emergency_heuristic = enable_emergency_heuristic

        self.model_path = Path(model_path)
        if self.model_path.parent != Path("."):
            self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.model = YOLO(str(self.model_path))
        self.model.to(self.device)

        self.plate_model_path = Path(plate_model_path)
        if self.plate_model_path.is_file():
            self.plate_model: YOLO | None = YOLO(str(self.plate_model_path))
            self.plate_model.to(self.device)
        else:
            logger.warning(
                "Custom plate model not found at %s; using bumper fallback heuristic.",
                self.plate_model_path,
            )
            self.plate_model = None

    def detect_vehicles(self, frame: np.ndarray) -> list[VehicleDetection]:
        """Run inference on an OpenCV BGR frame and return structured vehicle detections."""
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim < 2
        ):
            raise ValueError("frame must be a non-empty NumPy BGR image array.")

        height, width = frame.shape[:2]
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )

        if not results or results[0].boxes is None:
            return []

        result = results[0]
        detections: list[VehicleDetection] = []

        for box in result.boxes:
            class_id = int(box.cls.item())
            raw_name = result.names[class_id]
            class_name = str(raw_name).strip().lower()

            if class_name not in TARGET_CLASSES:
                continue

            confidence = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [
                max(0, min(width, int(round(x1)))),
                max(0, min(height, int(round(y1)))),
                max(0, min(width, int(round(x2)))),
                max(0, min(height, int(round(y2)))),
            ]
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue

            is_emergency = class_name in EMERGENCY_CLASSES
            if not is_emergency and self.enable_emergency_heuristic:
                is_emergency = self._evaluate_emergency_heuristic(
                    frame, bbox, class_name
                )

            detections.append(
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": bbox,
                    "is_emergency": is_emergency,
                }
            )

        # Prioritize emergency vehicles first, then higher confidence detections
        detections.sort(
            key=lambda d: (d["is_emergency"], d["confidence"]),
            reverse=True,
        )
        return detections

    def _evaluate_emergency_heuristic(
        self,
        frame: np.ndarray,
        bbox: Sequence[int],
        class_name: str,
    ) -> bool:
        """Check roof lightbar beacons and ambulance body heuristics inside a vehicle ROI."""
        roi = self.crop_region(frame, bbox)
        if roi.shape[0] < 10 or roi.shape[1] < 10:
            return False

        roof_height = max(1, int(roi.shape[0] * self.ROOF_BAND_RATIO))
        roof_roi = roi[:roof_height, :]
        roof_hsv = cv2.cvtColor(roof_roi, cv2.COLOR_BGR2HSV)

        red_lower = cv2.inRange(
            roof_hsv,
            np.array([0, 160, 180], dtype=np.uint8),
            np.array([10, 255, 255], dtype=np.uint8),
        )
        red_upper = cv2.inRange(
            roof_hsv,
            np.array([170, 160, 180], dtype=np.uint8),
            np.array([180, 255, 255], dtype=np.uint8),
        )
        red_mask = cv2.bitwise_or(red_lower, red_upper)
        blue_mask = cv2.inRange(
            roof_hsv,
            np.array([100, 160, 180], dtype=np.uint8),
            np.array([130, 255, 255], dtype=np.uint8),
        )

        roof_pixels = float(roof_roi.shape[0] * roof_roi.shape[1])
        red_ratio = float(cv2.countNonZero(red_mask)) / roof_pixels
        blue_ratio = float(cv2.countNonZero(blue_mask)) / roof_pixels

        if (
            red_ratio >= self.DUAL_BEACON_MIN_RATIO
            and blue_ratio >= self.DUAL_BEACON_MIN_RATIO
        ):
            return True

        if class_name in {"truck", "bus"} and (
            red_ratio >= self.SINGLE_BEACON_MIN_RATIO
            or blue_ratio >= self.SINGLE_BEACON_MIN_RATIO
        ):
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            white_mask = cv2.inRange(
                roi_hsv,
                np.array([0, 0, 190], dtype=np.uint8),
                np.array([180, 45, 255], dtype=np.uint8),
            )
            white_ratio = float(cv2.countNonZero(white_mask)) / float(
                roi.shape[0] * roi.shape[1]
            )
            if white_ratio >= self.LARGE_VEHICLE_WHITE_MIN_RATIO:
                return True

        return False

    @staticmethod
    def _bumper_fallback_bbox(height: int, width: int) -> list[int]:
        """Compute lower-center bumper fallback [x1, y1, x2, y2] inside a vehicle crop."""
        x1 = max(0, min(width - 1, int(round(width * 0.15))))
        x2 = max(x1 + 1, min(width, int(round(width * 0.85))))
        y1 = max(0, min(height - 1, int(round(height * 0.50))))
        y2 = max(y1 + 1, min(height, int(round(height * 0.95))))
        return [x1, y1, x2, y2]

    @staticmethod
    def translate_bbox_to_frame(
        vehicle_bbox: Sequence[int],
        local_bbox: Sequence[int],
        frame_shape: tuple[int, ...] | None = None,
    ) -> list[int]:
        """Translate vehicle-crop coordinates [x1, y1, x2, y2] into full-frame coordinates."""
        vx1, vy1 = int(vehicle_bbox[0]), int(vehicle_bbox[1])
        fx1 = vx1 + int(local_bbox[0])
        fy1 = vy1 + int(local_bbox[1])
        fx2 = vx1 + int(local_bbox[2])
        fy2 = vy1 + int(local_bbox[3])

        if frame_shape is not None and len(frame_shape) >= 2:
            height, width = int(frame_shape[0]), int(frame_shape[1])
            fx1 = max(0, min(width, fx1))
            fy1 = max(0, min(height, fy1))
            fx2 = max(0, min(width, fx2))
            fy2 = max(0, min(height, fy2))

        return [fx1, fy1, fx2, fy2]

    def detect_plate_bbox(
        self,
        vehicle_crop: np.ndarray,
        conf_threshold: float | None = None,
    ) -> list[int]:
        """Localize the tightest Egyptian license plate [x1, y1, x2, y2] inside a vehicle crop."""
        if (
            vehicle_crop is None
            or not isinstance(vehicle_crop, np.ndarray)
            or vehicle_crop.size == 0
            or vehicle_crop.ndim < 2
        ):
            raise ValueError("vehicle_crop must be a non-empty NumPy BGR image array.")

        height, width = vehicle_crop.shape[:2]
        threshold = (
            self.plate_conf_threshold
            if conf_threshold is None
            else float(conf_threshold)
        )

        if self.plate_model is not None:
            results = self.plate_model.predict(
                source=vehicle_crop,
                conf=threshold,
                device=self.device,
                verbose=False,
            )

            candidates: list[tuple[float, int, list[int]]] = []
            if results and results[0].boxes is not None:
                result = results[0]
                names = getattr(result, "names", {}) or {}
                multi_class = len(names) > 1

                for box in result.boxes:
                    conf_val = float(box.conf.item())
                    if conf_val < threshold:
                        continue

                    if multi_class:
                        class_id = int(box.cls.item())
                        label = str(names.get(class_id, "")).strip().lower()
                        if "plate" not in label:
                            continue

                    px1, py1, px2, py2 = box.xyxy[0].tolist()
                    bbox = [
                        max(0, min(width, int(round(px1)))),
                        max(0, min(height, int(round(py1)))),
                        max(0, min(width, int(round(px2)))),
                        max(0, min(height, int(round(py2)))),
                    ]
                    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                        continue

                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    candidates.append((conf_val, area, bbox))

            if candidates:
                max_conf = max(item[0] for item in candidates)
                high_conf = [
                    item for item in candidates if (max_conf - item[0]) <= 0.05
                ]
                _, _, best_bbox = min(
                    high_conf,
                    key=lambda item: (item[1], -item[0]),
                )
                return best_bbox

        return self._bumper_fallback_bbox(height, width)

    @staticmethod
    def crop_region(
        frame: np.ndarray,
        bbox: Sequence[int | float],
        padding: int = 0,
    ) -> np.ndarray:
        """Crop and return a contiguous BGR ROI from frame using [x1, y1, x2, y2] coordinates."""
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim < 2
        ):
            raise ValueError("frame must be a non-empty NumPy image array.")
        if len(bbox) != 4:
            raise ValueError("bbox must contain exactly 4 coordinates: [x1, y1, x2, y2].")
        if padding < 0:
            raise ValueError("padding must be non-negative.")

        height, width = frame.shape[:2]
        x1 = max(0, min(width, int(round(bbox[0])) - padding))
        y1 = max(0, min(height, int(round(bbox[1])) - padding))
        x2 = max(0, min(width, int(round(bbox[2])) + padding))
        y2 = max(0, min(height, int(round(bbox[3])) + padding))

        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box resulting in an empty crop region.")

        return np.ascontiguousarray(frame[y1:y2, x1:x2].copy())


def _load_test_frame(cli_path: str | None = None) -> tuple[np.ndarray, str]:
    """Load a test frame from CLI path, data/test_images, webcam, or synthetic fallback."""
    if cli_path:
        image_path = Path(cli_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Unable to read image at: {image_path}")
        return frame, f"image file ({image_path})"

    if DEFAULT_TEST_IMAGES_DIR.is_dir():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            matches = sorted(DEFAULT_TEST_IMAGES_DIR.glob(ext))
            if matches:
                frame = cv2.imread(str(matches[0]))
                if frame is not None:
                    return frame, f"sample image ({matches[0]})"

    cap = cv2.VideoCapture(0)
    try:
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                return frame, "webcam (device 0)"
    finally:
        cap.release()

    synthetic = np.zeros((480, 640, 3), dtype=np.uint8)
    return synthetic, "synthetic 640x480 BGR frame (no sample image or webcam available)"


if __name__ == "__main__":
    cli_arg = sys.argv[1] if len(sys.argv) > 1 else None
    test_frame, frame_source = _load_test_frame(cli_arg)

    detector = VehicleDetector()
    device_label = (
        f"cuda ({torch.cuda.get_device_name(0)})"
        if detector.device.startswith("cuda") and torch.cuda.is_available()
        else detector.device
    )

    print(f"Hardware device: {device_label}")
    print(f"Vehicle model loaded: {detector.model_path}")
    print(
        "Plate model loaded: "
        f"{detector.plate_model_path if detector.plate_model is not None else 'fallback heuristic only'}"
    )
    print(f"Input source: {frame_source}")

    detections = detector.detect_vehicles(test_frame)
    print(f"Detected vehicles: {len(detections)}")
    for idx, det in enumerate(detections, start=1):
        roi = detector.crop_region(test_frame, det["bbox"])
        local_plate = detector.detect_plate_bbox(roi)
        frame_plate = detector.translate_bbox_to_frame(
            det["bbox"], local_plate, test_frame.shape
        )
        print(
            f"  {idx}. class={det['class_name']}, "
            f"conf={det['confidence']:.3f}, "
            f"bbox={det['bbox']}, "
            f"is_emergency={det['is_emergency']}, "
            f"roi_shape={roi.shape}, "
            f"plate_bbox={frame_plate}"
        )

    if not detections:
        sample_roi = detector.crop_region(test_frame, [100, 100, 300, 250])
        sample_plate = detector.detect_plate_bbox(sample_roi)
        print(f"Sample crop_region ROI shape: {sample_roi.shape}, plate_bbox={sample_plate}")

    print("VehicleDetector verification completed without errors.")

