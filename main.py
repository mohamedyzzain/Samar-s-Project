"""SafeGate AI entrypoint for live gate monitoring and end-to-end simulation."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from typing import Sequence

import numpy as np

from src.core import GateController
from src.database import GateRepository
from src.perception.pipeline import PerceptionResult
from src.ui import LiveGateMonitor, parse_video_source


class SimulatedPerceptionPipeline:
    """Deterministic perception pipeline stub yielding pre-built test payloads."""

    def __init__(self, payloads: Sequence[PerceptionResult]) -> None:
        self._payloads: deque[PerceptionResult] = deque(payloads)

    def process_frame(self, frame: np.ndarray) -> PerceptionResult:
        """Return the next simulated perception payload for an incoming frame."""
        if not self._payloads:
            raise RuntimeError("No more simulated perception payloads remaining.")
        return self._payloads.popleft()


def _configure_runtime() -> None:
    """Ensure UTF-8 stdout encoding and standard application logging."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def run_integration_test() -> None:
    """Simulate emergency and resident traffic flowing through GateController into data/safegate.db."""
    _configure_runtime()

    payloads: list[PerceptionResult] = [
        {"is_emergency": True, "plate_text": None, "confidence": 0.96},
        {"is_emergency": False, "plate_text": "ا ب ج ١٢٣", "confidence": 0.92},
        {"is_emergency": False, "plate_text": "م ج ه ٩٩٩", "confidence": 0.87},
    ]
    expected_decisions = [
        {"action": "OPEN", "reason": "Emergency Vehicle Bypass", "plate": "EMERGENCY"},
        {"action": "OPEN", "reason": "Authorized Resident", "plate": "ا ب ج ١٢٣"},
        {"action": "DENY", "reason": "Unknown/Visitor", "plate": "م ج ه ٩٩٩"},
    ]

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with GateRepository() as repo:
        repo.seed_dummy_data()
        pipeline = SimulatedPerceptionPipeline(payloads)

        with GateController(repository=repo, pipeline=pipeline) as controller:
            for idx, expected in enumerate(expected_decisions, start=1):
                decision = controller.process_frame(dummy_frame)
                print(f"Scenario {idx} Decision: {decision}")
                assert decision == expected, (
                    f"Scenario {idx} mismatch: expected {expected}, got {decision}"
                )

        recent_logs = repo.get_recent_logs(limit=5)
        print("Recent Access_Logs entries in data/safegate.db:")
        for log_id, plate, timestamp, status in recent_logs:
            print(f"  [{log_id}] {timestamp} | plate={plate} | status={status}")

        persisted_pairs = {(plate, status) for _, plate, _, status in recent_logs}
        assert ("EMERGENCY", "Emergency Vehicle Bypass") in persisted_pairs
        assert ("ا ب ج ١٢٣", "Authorized Resident") in persisted_pairs
        assert ("م ج ه ٩٩٩", "Unknown/Visitor") in persisted_pairs

    print("End-to-end SafeGate AI integration test passed.")


def run_live_monitor(
    source: int | str,
    ocr_interval: int = 10,
    cooldown: float = 5.0,
) -> None:
    """Launch LiveGateMonitor connected to data/safegate.db."""
    _configure_runtime()
    logger = logging.getLogger(__name__)
    logger.info("Starting SafeGate AI Live Monitor on source=%r", source)

    with GateRepository() as repo:
        repo.seed_dummy_data()
        controller = GateController(repository=repo)
        monitor = LiveGateMonitor(
            source=source,
            controller=controller,
            ocr_interval=ocr_interval,
            cooldown_seconds=cooldown,
        )
        try:
            monitor.run()
        except KeyboardInterrupt:
            logger.info("Interrupted by user. Closing SafeGate AI monitor.")
        finally:
            monitor.close()
            controller.close()


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for SafeGate AI."""
    parser = argparse.ArgumentParser(
        description="SafeGate AI Smart Gate Access Control & Live Monitor",
    )
    parser.add_argument(
        "--source",
        default="0",
        help=(
            "Video source: local webcam index (e.g. 0) or IP/RTSP stream URL "
            "(e.g. http://192.168.1.15:8080/video). Default: 0."
        ),
    )
    parser.add_argument(
        "--ocr-interval",
        type=int,
        default=10,
        help="Run heavy OCR every N processed frames (default: 10).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=5.0,
        help="Seconds to debounce duplicate DB logging per unique plate (default: 5.0).",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run the offline DB integration simulation instead of live camera monitoring.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint dispatching live monitoring or offline simulation."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.simulate:
        run_integration_test()
        return

    parsed_source = parse_video_source(args.source)
    run_live_monitor(
        source=parsed_source,
        ocr_interval=args.ocr_interval,
        cooldown=args.cooldown,
    )


if __name__ == "__main__":
    main()
