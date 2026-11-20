"""Live video stream processor and real-time OpenCV/Pillow HUD monitor for SafeGate AI."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, features as pil_features

try:
    import arabic_reshaper
except ImportError:
    arabic_reshaper = None

try:
    from bidi.algorithm import get_display as bidi_get_display
except ImportError:
    bidi_get_display = None

from src.core import AccessDecision, GateController
from src.perception import PerceptionPipeline, VehicleDetection, VehicleDetector

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "logs" / "snapshots"
WINDOW_TITLE = "SafeGate AI - Live Gate Monitor"

# OpenCV BGR bounding box & overlay colors
COLOR_OPEN_BGR = (0, 200, 0)
COLOR_DENY_BGR = (0, 0, 220)
COLOR_SCAN_BGR = (0, 215, 255)
COLOR_PLATE_BGR = (255, 255, 0)

# Pillow RGB HUD text colors
COLOR_OPEN_RGB = (0, 220, 90)
COLOR_DENY_RGB = (235, 65, 65)
COLOR_SCAN_RGB = (255, 210, 40)
COLOR_TEXT_WHITE = (245, 245, 245)
COLOR_TEXT_MUTED = (200, 205, 215)

_HUD_FONT_CANDIDATES = (
    Path(r"C:\Windows\Fonts\tahomabd.ttf"),
    Path(r"C:\Windows\Fonts\tahoma.ttf"),
    Path(r"C:\Windows\Fonts\arialbd.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
    Path(r"C:\Windows\Fonts\segoeuib.ttf"),
    Path(r"C:\Windows\Fonts\segoeui.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
)


def parse_video_source(source: int | str) -> int | str:
    """Convert numeric strings (e.g. '0') to webcam ints while keeping stream URLs as strings."""
    if isinstance(source, int):
        return source
    cleaned = source.strip()
    if cleaned.isdigit():
        return int(cleaned)
    return cleaned


def shape_arabic_text(text: str) -> str:
    """Prepare Arabic text for Pillow rendering without double-reversing if libraqm is active."""
    if not text:
        return ""
    if pil_features.check("raqm"):
        return text
    if arabic_reshaper is not None and bidi_get_display is not None:
        return bidi_get_display(arabic_reshaper.reshape(text))
    return text


def load_hud_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load the first available Arabic-capable TrueType font, or default bitmap font."""
    for candidate in _HUD_FONT_CANDIDATES:
        if candidate.is_file():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except OSError:
                continue
    return ImageFont.load_default()


@dataclass
class MonitorState:
    """Latest vehicle detections, localized plates, gate decision, and HUD status."""

    detections: list[VehicleDetection] = field(default_factory=list)
    plate_bboxes: list[list[int]] = field(default_factory=list)
    decision: AccessDecision | None = None
    status_label: str = "SCANNING..."
    updated_at: float = 0.0


class LiveGateMonitor:
    """Streams webcam or IP camera frames, throttles OCR, debounces DB logs, and renders HUD."""

    def __init__(
        self,
        source: int | str = 0,
        controller: GateController | None = None,
        ocr_interval: int = 10,
        cooldown_seconds: float = 5.0,
        max_frame_failures: int = 5,
        reconnect_delay: float = 2.0,
        max_reconnect_attempts: int = 10,
        snapshot_dir: str | Path = DEFAULT_SNAPSHOT_DIR,
        window_name: str = WINDOW_TITLE,
        async_inference: bool = True,
        capture_factory: Optional[Callable[[int | str], cv2.VideoCapture]] = None,
        clock: Callable[[], float] = time.monotonic,
        display_fn: Optional[Callable[[str, np.ndarray], None]] = None,
        wait_key_fn: Optional[Callable[[int], int]] = None,
    ) -> None:
        """Configure stream source, throttling, cooldown debounce, and HUD rendering."""
        self.source = parse_video_source(source)
        self._owns_controller = controller is None
        self.controller = controller if controller is not None else GateController()

        self.ocr_interval = max(1, int(ocr_interval))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.max_frame_failures = max(1, int(max_frame_failures))
        self.reconnect_delay = max(0.0, float(reconnect_delay))
        self.max_reconnect_attempts = max(0, int(max_reconnect_attempts))
        self.snapshot_dir = Path(snapshot_dir)
        self.window_name = window_name
        self.async_inference = async_inference

        self._capture_factory = capture_factory or self._default_capture_factory
        self._clock = clock
        self._uses_native_gui = display_fn is None
        self._display_fn = display_fn if display_fn is not None else cv2.imshow
        self._wait_key_fn = wait_key_fn if wait_key_fn is not None else cv2.waitKey

        self._state = MonitorState()
        self._state_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._cooldown_lock = threading.Lock()
        self._stop_event = threading.Event()

        self._latest_frame: np.ndarray | None = None
        self._frame_available = threading.Event()
        self._processed_frames = 0

        self._last_logged_at: dict[str, float] = {}
        self._cached_decisions: dict[str, AccessDecision] = {}

        # Cache TrueType fonts once so per-frame HUD drawing has minimal overhead
        self._font_status = load_hud_font(26)
        self._font_plate = load_hud_font(24)
        self._font_info = load_hud_font(16)

    @staticmethod
    def _default_capture_factory(source: int | str) -> cv2.VideoCapture:
        """Open a cv2.VideoCapture stream and minimize buffering on network URLs."""
        cap = cv2.VideoCapture(source)
        if isinstance(source, str):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _source_label(self) -> str:
        """Return a concise source label for the HUD."""
        if isinstance(self.source, int):
            return f"Webcam #{self.source}"
        if len(self.source) > 36:
            return f"{self.source[:33]}..."
        return self.source

    def is_in_cooldown(self, key: str, now: float | None = None) -> bool:
        """Return True if key was evaluated within cooldown_seconds."""
        current = self._clock() if now is None else now
        with self._cooldown_lock:
            last_seen = self._last_logged_at.get(key)
            if last_seen is None:
                return False
            return (current - last_seen) < self.cooldown_seconds

    def _evaluate_with_cooldown(
        self,
        key: str,
        perception_data: dict[str, object],
    ) -> AccessDecision:
        """Return a cached decision within cooldown_seconds or evaluate and log via GateController."""
        now = self._clock()
        with self._cooldown_lock:
            last_seen = self._last_logged_at.get(key)
            if (
                last_seen is not None
                and (now - last_seen) < self.cooldown_seconds
                and key in self._cached_decisions
            ):
                return self._cached_decisions[key]

            decision = self.controller.evaluate_perception(perception_data)
            self._last_logged_at[key] = now
            self._cached_decisions[key] = decision
            return decision

    def _hold_or_reset_state(
        self,
        detections: list[VehicleDetection],
        now: float,
        plate_bboxes: list[list[int]] | None = None,
    ) -> MonitorState:
        """Hold recent gate decision during cooldown window, otherwise return SCANNING."""
        with self._state_lock:
            prev_decision = self._state.decision
            prev_status = self._state.status_label
            prev_time = self._state.updated_at

        plates = list(plate_bboxes) if plate_bboxes is not None else []
        if prev_decision is not None and (now - prev_time) < self.cooldown_seconds:
            return MonitorState(
                detections=detections,
                plate_bboxes=plates,
                decision=prev_decision,
                status_label=prev_status,
                updated_at=prev_time,
            )

        return MonitorState(
            detections=detections,
            plate_bboxes=plates,
            decision=None,
            status_label="SCANNING...",
            updated_at=now,
        )

    def analyze_frame(
        self,
        frame: np.ndarray,
        run_ocr: bool = True,
    ) -> MonitorState:
        """Run detection every call and throttle heavy OCR according to run_ocr."""
        now = self._clock()
        pipeline = self.controller.pipeline

        if not (hasattr(pipeline, "detector") and hasattr(pipeline, "ocr_reader")):
            # Fallback for custom PerceptionProcessor stubs without split stages
            if not run_ocr:
                return self._hold_or_reset_state([], now)
            perception = pipeline.process_frame(frame)
            if perception["is_emergency"]:
                decision = self._evaluate_with_cooldown("EMERGENCY", dict(perception))
                return MonitorState([], [], decision, "GATE OPEN", now)
            plate = perception["plate_text"]
            if plate:
                decision = self._evaluate_with_cooldown(plate, dict(perception))
                status = "GATE OPEN" if decision["action"] == "OPEN" else "ACCESS DENIED"
                return MonitorState([], [], decision, status, now)
            return self._hold_or_reset_state([], now)

        detections: list[VehicleDetection] = pipeline.detector.detect_vehicles(frame)
        if not detections:
            return self._hold_or_reset_state([], now)

        # Stage 1: Zero-latency emergency bypass executes immediately on any frame
        primary = detections[0]
        if primary["is_emergency"]:
            decision = self._evaluate_with_cooldown(
                "EMERGENCY",
                {
                    "is_emergency": True,
                    "plate_text": None,
                    "confidence": float(primary["confidence"]),
                },
            )
            return MonitorState(
                detections=detections,
                plate_bboxes=[],
                decision=decision,
                status_label="GATE OPEN",
                updated_at=now,
            )

        # Stage 2: Two-stage plate localization every frame + throttled OCR
        plate_bboxes: list[list[int]] = []
        matched_decision: AccessDecision | None = None
        matched_status: str | None = None

        for det in detections:
            vehicle_crop = pipeline.detector.crop_region(frame, det["bbox"])
            if hasattr(pipeline.detector, "detect_plate_bbox"):
                local_plate_bbox = pipeline.detector.detect_plate_bbox(vehicle_crop)
                plate_crop = pipeline.detector.crop_region(vehicle_crop, local_plate_bbox)
                frame_plate_bbox = VehicleDetector.translate_bbox_to_frame(
                    det["bbox"],
                    local_plate_bbox,
                    frame.shape,
                )
                plate_bboxes.append(frame_plate_bbox)
            else:
                plate_crop = PerceptionPipeline._extract_plate_candidate(vehicle_crop)

            if run_ocr and matched_decision is None:
                plate_text, ocr_conf = pipeline.ocr_reader.read_plate(plate_crop)
                if plate_text is None and plate_crop.shape != vehicle_crop.shape:
                    plate_text, ocr_conf = pipeline.ocr_reader.read_plate(vehicle_crop)

                if plate_text is not None:
                    matched_decision = self._evaluate_with_cooldown(
                        plate_text,
                        {
                            "is_emergency": False,
                            "plate_text": plate_text,
                            "confidence": float(ocr_conf),
                        },
                    )
                    matched_status = (
                        "GATE OPEN"
                        if matched_decision["action"] == "OPEN"
                        else "ACCESS DENIED"
                    )

        if matched_decision is not None and matched_status is not None:
            return MonitorState(
                detections=detections,
                plate_bboxes=plate_bboxes,
                decision=matched_decision,
                status_label=matched_status,
                updated_at=now,
            )

        return self._hold_or_reset_state(detections, now, plate_bboxes=plate_bboxes)

    def _status_colors(
        self,
        status_label: str,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Map status label to (BGR color, RGB color)."""
        if status_label == "GATE OPEN":
            return COLOR_OPEN_BGR, COLOR_OPEN_RGB
        if status_label == "ACCESS DENIED":
            return COLOR_DENY_BGR, COLOR_DENY_RGB
        return COLOR_SCAN_BGR, COLOR_SCAN_RGB

    def render_hud(
        self,
        frame: np.ndarray,
        state: MonitorState,
        fps: float = 0.0,
        snapshot_notice: str | None = None,
    ) -> np.ndarray:
        """Draw vehicle bounding boxes, status banner, and Arabic plate text onto frame."""
        annotated = frame.copy()
        height, width = annotated.shape[:2]
        active_bgr, active_rgb = self._status_colors(state.status_label)

        for det in state.detections:
            x1, y1, x2, y2 = det["bbox"]
            box_bgr = COLOR_OPEN_BGR if det["is_emergency"] else active_bgr
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_bgr, 2)

            tag = f"{det['class_name']} {det['confidence']:.2f}"
            if det["is_emergency"]:
                tag = f"EMERGENCY ({tag})"
            label_y = max(20, y1 - 8)
            cv2.putText(
                annotated,
                tag,
                (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_bgr,
                2,
                cv2.LINE_AA,
            )

        # Draw distinct Cyan bounding boxes around localized license plates
        for px1, py1, px2, py2 in state.plate_bboxes:
            cv2.rectangle(annotated, (px1, py1), (px2, py2), COLOR_PLATE_BGR, 2)
            plate_tag_y = max(20, py1 - 6)
            cv2.putText(
                annotated,
                "PLATE",
                (px1, plate_tag_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                COLOR_PLATE_BGR,
                2,
                cv2.LINE_AA,
            )

        # Semi-transparent top status banner for clear readability over live video
        bar_height = min(height, 108)
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (width, bar_height), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.72, annotated, 0.28, 0, annotated)
        cv2.line(annotated, (0, bar_height), (width, bar_height), active_bgr, 2)

        pil_img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        draw.text((16, 10), state.status_label, font=self._font_status, fill=active_rgb)

        reason = (
            state.decision["reason"]
            if state.decision is not None
            else "Monitoring gate camera..."
        )
        draw.text((16, 44), reason, font=self._font_info, fill=COLOR_TEXT_MUTED)

        plate_value = state.decision["plate"] if state.decision is not None else None
        if plate_value:
            prefix = "Plate: "
            draw.text((16, 68), prefix, font=self._font_plate, fill=COLOR_TEXT_WHITE)
            prefix_width = int(draw.textlength(prefix, font=self._font_plate))
            draw.text(
                (16 + prefix_width + 4, 68),
                shape_arabic_text(plate_value),
                font=self._font_plate,
                fill=active_rgb,
            )
        else:
            draw.text((16, 70), "Plate: ---", font=self._font_info, fill=COLOR_TEXT_MUTED)

        telemetry = f"FPS: {fps:.1f} | {self._source_label()}"
        tel_width = int(draw.textlength(telemetry, font=self._font_info))
        draw.text(
            (max(16, width - tel_width - 16), 14),
            telemetry,
            font=self._font_info,
            fill=COLOR_TEXT_WHITE,
        )

        controls_hint = "[S] Snapshot  |  [Q] Quit"
        hint_width = int(draw.textlength(controls_hint, font=self._font_info))
        draw.text(
            (max(16, width - hint_width - 16), 44),
            controls_hint,
            font=self._font_info,
            fill=COLOR_TEXT_MUTED,
        )

        if snapshot_notice:
            notice_width = int(draw.textlength(snapshot_notice, font=self._font_info))
            draw.text(
                (max(16, width - notice_width - 16), 72),
                snapshot_notice,
                font=self._font_info,
                fill=COLOR_OPEN_RGB,
            )

        return cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)

    def save_snapshot(self, frame: np.ndarray) -> Path:
        """Save the current frame with a timestamped filename into logs/snapshots/."""
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = self.snapshot_dir / f"snapshot_{timestamp}.jpg"
        saved = cv2.imwrite(str(output_path), frame)
        if not saved:
            raise RuntimeError(f"Failed to write snapshot image to: {output_path}")
        logger.info("Saved snapshot: %s", output_path)
        return output_path

    def _open_capture_with_retry(self) -> cv2.VideoCapture | None:
        """Attempt to open the configured video source, retrying on transient failures."""
        attempts = 0
        while not self._stop_event.is_set():
            cap = self._capture_factory(self.source)
            if cap is not None and cap.isOpened():
                return cap
            if cap is not None:
                cap.release()

            attempts += 1
            if attempts > self.max_reconnect_attempts:
                logger.error(
                    "Unable to open video source %r after %d attempts.",
                    self.source,
                    self.max_reconnect_attempts,
                )
                return None

            logger.warning(
                "Video source %r unavailable (attempt %d/%d). Retrying in %.1fs...",
                self.source,
                attempts,
                self.max_reconnect_attempts,
                self.reconnect_delay,
            )
            if self._stop_event.wait(self.reconnect_delay):
                return None
        return None

    def _inference_worker(self) -> None:
        """Process the most recent queued frame in the background to prevent UI lag."""
        while not self._stop_event.is_set():
            if not self._frame_available.wait(timeout=0.1):
                continue

            with self._frame_lock:
                frame = self._latest_frame
                self._latest_frame = None
                self._frame_available.clear()

            if frame is None:
                continue

            self._processed_frames += 1
            run_ocr = (self._processed_frames % self.ocr_interval) == 0
            try:
                new_state = self.analyze_frame(frame, run_ocr=run_ocr)
                with self._state_lock:
                    self._state = new_state
            except Exception:
                logger.exception("Error during background frame inference.")

    def _submit_frame(self, frame: np.ndarray) -> None:
        """Submit the latest frame to the inference worker, dropping any stale frame."""
        with self._frame_lock:
            self._latest_frame = frame.copy()
            self._frame_available.set()

    def run(self) -> None:
        """Start the live video loop with automatic reconnection, HUD overlay, and hotkeys."""
        self._stop_event.clear()
        self._processed_frames = 0
        worker: threading.Thread | None = None
        cap: cv2.VideoCapture | None = None

        try:
            cap = self._open_capture_with_retry()
            if cap is None:
                return

            if self.async_inference:
                worker = threading.Thread(
                    target=self._inference_worker,
                    name="safegate-inference",
                    daemon=True,
                )
                worker.start()

            consecutive_failures = 0
            reconnect_attempts = 0
            last_tick = time.perf_counter()
            fps = 0.0
            sync_frames = 0
            snapshot_notice: str | None = None
            snapshot_notice_until = 0.0

            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    consecutive_failures += 1
                    if consecutive_failures < self.max_frame_failures:
                        continue

                    logger.warning(
                        "Stream frame drop threshold (%d) reached. Reconnecting to %r...",
                        self.max_frame_failures,
                        self.source,
                    )
                    cap.release()
                    cap = None
                    consecutive_failures = 0
                    reconnect_attempts += 1

                    if reconnect_attempts > self.max_reconnect_attempts:
                        logger.error(
                            "Exceeded max reconnect attempts (%d). Stopping monitor.",
                            self.max_reconnect_attempts,
                        )
                        break

                    if self._stop_event.wait(self.reconnect_delay):
                        break

                    cap = self._open_capture_with_retry()
                    if cap is None:
                        break
                    continue

                consecutive_failures = 0
                reconnect_attempts = 0

                now_perf = time.perf_counter()
                dt = now_perf - last_tick
                last_tick = now_perf
                if dt > 0:
                    instant_fps = 1.0 / dt
                    fps = instant_fps if fps == 0.0 else (0.9 * fps + 0.1 * instant_fps)

                if self.async_inference:
                    self._submit_frame(frame)
                    with self._state_lock:
                        current_state = self._state
                else:
                    sync_frames += 1
                    run_ocr = (sync_frames % self.ocr_interval) == 0
                    current_state = self.analyze_frame(frame, run_ocr=run_ocr)
                    with self._state_lock:
                        self._state = current_state

                now_mono = self._clock()
                active_notice = (
                    snapshot_notice if now_mono < snapshot_notice_until else None
                )
                annotated = self.render_hud(
                    frame,
                    current_state,
                    fps=fps,
                    snapshot_notice=active_notice,
                )
                self._display_fn(self.window_name, annotated)

                key = self._wait_key_fn(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    logger.info("Quit key pressed. Shutting down live monitor.")
                    break
                if key in (ord("s"), ord("S")):
                    saved_path = self.save_snapshot(annotated)
                    snapshot_notice = f"Saved: {saved_path.name}"
                    snapshot_notice_until = self._clock() + 2.0

                if self._uses_native_gui:
                    try:
                        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                            logger.info("Monitor window closed. Shutting down.")
                            break
                    except cv2.error:
                        break
        finally:
            self._stop_event.set()
            self._frame_available.set()
            if worker is not None and worker.is_alive():
                worker.join(timeout=3.0)
            if cap is not None:
                cap.release()
            if self._uses_native_gui:
                cv2.destroyAllWindows()
            self.close()

    def close(self) -> None:
        """Close the underlying GateController if owned by this monitor."""
        if self._owns_controller and self.controller is not None:
            self.controller.close()

    def __enter__(self) -> LiveGateMonitor:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

