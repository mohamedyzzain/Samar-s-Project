"""Egyptian License Plate Recognition (LPR) module using OpenCV and EasyOCR."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import easyocr
import numpy as np
import torch

# Standardize Arabic character variants (alef forms, alef maksura, teh marbuta, Persian forms)
_ARABIC_LETTER_MAP = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ی": "ي",
        "ة": "ه",
        "ک": "ك",
    }
)

# Convert Western (0-9) and Eastern Persian numerals to Arabic-Indic numerals (٠-٩)
_DIGIT_TO_ARABIC_INDIC = str.maketrans(
    "0123456789۰۱۲۳۴۵۶۷۸۹",
    "٠١٢٣٤٥٦٧٨٩٠١٢٣٤٥٦٧٨٩",
)

# Strip Arabic diacritics (tashkeel) and elongation (tatweel)
_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670\u0640]")
_ARABIC_LETTER_RE = re.compile(r"[\u0621-\u063a\u0641-\u064a]")
_ARABIC_INDIC_DIGIT_RE = re.compile(r"[\u0660-\u0669]")


class EgyptianPlateReader:
    """Preprocesses license plate crops and reads/normalizes Egyptian Arabic plates."""

    def __init__(
        self,
        languages: Sequence[str] = ("ar",),
        gpu: bool = True,
        min_confidence: float = 0.4,
        use_adaptive_threshold: bool = False,
    ) -> None:
        """Initialize EasyOCR with GPU when CUDA is available, falling back to CPU."""
        self.languages = list(languages)
        self.min_confidence = min_confidence
        self.use_adaptive_threshold = use_adaptive_threshold

        requested_gpu = bool(gpu and torch.cuda.is_available())
        if requested_gpu:
            try:
                self.reader = easyocr.Reader(self.languages, gpu=True, verbose=False)
                self.use_gpu = True
            except (RuntimeError, AssertionError):
                self.use_gpu = False
                self.reader = easyocr.Reader(self.languages, gpu=False, verbose=False)
        else:
            self.use_gpu = False
            self.reader = easyocr.Reader(self.languages, gpu=False, verbose=False)

        self.device = "cuda" if self.use_gpu else "cpu"

    def preprocess_plate(
        self,
        plate_img: np.ndarray,
        apply_threshold: bool | None = None,
    ) -> np.ndarray:
        """Enhance plate contrast and clarity with grayscale, bilateral filtering, and CLAHE."""
        if (
            plate_img is None
            or not isinstance(plate_img, np.ndarray)
            or plate_img.size == 0
            or plate_img.ndim < 2
        ):
            raise ValueError("plate_img must be a non-empty NumPy image array.")

        if plate_img.ndim == 2:
            gray = plate_img.copy()
        elif plate_img.ndim == 3 and plate_img.shape[2] == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        elif plate_img.ndim == 3 and plate_img.shape[2] == 4:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGRA2GRAY)
        elif plate_img.ndim == 3 and plate_img.shape[2] == 1:
            gray = plate_img[:, :, 0].copy()
        else:
            raise ValueError("Unsupported channel count in plate_img.")

        # Upscale small crops so Arabic dots and diacritical features remain legible
        if gray.shape[0] < 64:
            gray = cv2.resize(
                gray,
                None,
                fx=2.0,
                fy=2.0,
                interpolation=cv2.INTER_CUBIC,
            )

        denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        do_threshold = (
            self.use_adaptive_threshold
            if apply_threshold is None
            else apply_threshold
        )
        if do_threshold:
            return cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                15,
                5,
            )

        return enhanced

    @staticmethod
    def _clean_text(raw_text: str) -> str:
        """Strip diacritics and standardize Arabic letters and Arabic-Indic digits."""
        cleaned = _DIACRITICS_RE.sub("", raw_text)
        return cleaned.translate(_ARABIC_LETTER_MAP).translate(_DIGIT_TO_ARABIC_INDIC)

    @staticmethod
    def _format_plate_parts(letters: Sequence[str], digits: Sequence[str]) -> str:
        """Join space-separated Arabic letters followed by contiguous Arabic-Indic digits."""
        letters_part = " ".join(letters)
        digits_part = "".join(digits)
        return " ".join(part for part in (letters_part, digits_part) if part)

    @classmethod
    def normalize_plate_text(cls, raw_text: str) -> str:
        """Normalize raw OCR text to standard Egyptian plate format (e.g. 'ا ب ج ١٢٣')."""
        if not raw_text:
            return ""

        cleaned = cls._clean_text(raw_text)
        letters = _ARABIC_LETTER_RE.findall(cleaned)
        digits = _ARABIC_INDIC_DIGIT_RE.findall(cleaned)
        return cls._format_plate_parts(letters, digits)

    def read_plate(self, plate_img: np.ndarray) -> tuple[str | None, float]:
        """Run OCR on a plate crop, filter noise (< min_confidence), and normalize text."""
        processed = self.preprocess_plate(plate_img)
        raw_results: list[Any] = self.reader.readtext(processed, detail=1)

        letter_boxes: list[tuple[float, list[str]]] = []
        digit_boxes: list[tuple[float, list[str]]] = []
        confs: list[float] = []

        for entry in raw_results:
            bbox, text, conf = entry
            conf_val = float(conf)
            if conf_val < self.min_confidence:
                continue

            cleaned = self._clean_text(str(text))
            letters = _ARABIC_LETTER_RE.findall(cleaned)
            digits = _ARABIC_INDIC_DIGIT_RE.findall(cleaned)
            if not letters and not digits:
                continue

            cx = (
                float(sum(pt[0] for pt in bbox)) / len(bbox)
                if bbox
                else 0.0
            )
            if letters:
                letter_boxes.append((cx, letters))
            if digits:
                digit_boxes.append((cx, digits))
            confs.append(conf_val)

        if not confs:
            return None, 0.0

        # Egyptian plates place Arabic letters on the right (RTL) and digits on the left (LTR)
        letter_boxes.sort(key=lambda item: item[0], reverse=True)
        digit_boxes.sort(key=lambda item: item[0], reverse=False)

        all_letters = [ch for _, chars in letter_boxes for ch in chars]
        all_digits = [ch for _, chars in digit_boxes for ch in chars]
        normalized = self._format_plate_parts(all_letters, all_digits)
        if not normalized:
            return None, 0.0

        avg_confidence = float(sum(confs) / len(confs))
        return normalized, avg_confidence


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    reader = EgyptianPlateReader()
    print(f"OCR hardware device: {reader.device}")

    samples = ["أ ب ج 123", "سمر4567", "٧٨٩ ح-ب-ى !@#"]
    for sample in samples:
        print(f"Normalized '{sample}' -> '{reader.normalize_plate_text(sample)}'")

    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        plate_frame = cv2.imread(str(img_path))
        if plate_frame is None:
            raise FileNotFoundError(f"Unable to read plate image at: {img_path}")
    else:
        plate_frame = np.full((120, 320, 3), 240, dtype=np.uint8)

    preprocessed = reader.preprocess_plate(plate_frame)
    plate_text, confidence = reader.read_plate(plate_frame)
    print(f"Preprocessed shape: {preprocessed.shape}")
    print(f"OCR output: plate_text={plate_text!r}, confidence={confidence:.3f}")
    print("EgyptianPlateReader verification completed without errors.")

