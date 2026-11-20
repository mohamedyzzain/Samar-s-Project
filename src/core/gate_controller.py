"""Core gate controller orchestrating perception, strategy evaluation, and logging."""

from __future__ import annotations

import logging
from typing import Any, Mapping, Protocol

import numpy as np

from src.core.access_strategy import (
    AccessDecision,
    AccessStrategy,
    EmergencyPriorityStrategy,
    ResidentVerificationStrategy,
)
from src.database.repository import GateRepository
from src.perception.pipeline import PerceptionPipeline, PerceptionResult

logger = logging.getLogger(__name__)


class PerceptionProcessor(Protocol):
    """Structural interface for real or simulated perception pipelines."""

    def process_frame(self, frame: np.ndarray) -> PerceptionResult:
        """Process an image frame and return a unified perception payload."""
        ...


class GateController:
    """Orchestrates PerceptionPipeline, AccessStrategy selection, and gate audit logging."""

    def __init__(
        self,
        repository: GateRepository | None = None,
        pipeline: PerceptionProcessor | None = None,
        emergency_strategy: AccessStrategy | None = None,
        resident_strategy: AccessStrategy | None = None,
    ) -> None:
        """Initialize repository, optional perception pipeline, and decision strategies."""
        self._owns_repository = repository is None
        self.repository = repository if repository is not None else GateRepository()
        self._pipeline: PerceptionProcessor | None = pipeline
        self.emergency_strategy: AccessStrategy = (
            emergency_strategy
            if emergency_strategy is not None
            else EmergencyPriorityStrategy()
        )
        self.resident_strategy: AccessStrategy = (
            resident_strategy
            if resident_strategy is not None
            else ResidentVerificationStrategy()
        )

    @property
    def pipeline(self) -> PerceptionProcessor:
        """Lazily initialize the perception pipeline on first frame processing call."""
        if self._pipeline is None:
            self._pipeline = PerceptionPipeline()
        return self._pipeline

    def select_strategy(self, perception_data: Mapping[str, Any]) -> AccessStrategy:
        """Choose the emergency bypass strategy or standard resident verification strategy."""
        if bool(perception_data.get("is_emergency", False)):
            return self.emergency_strategy
        return self.resident_strategy

    def evaluate_perception(
        self,
        perception_data: Mapping[str, Any],
    ) -> AccessDecision:
        """Select a strategy, evaluate access, execute gate action, and log the event."""
        strategy = self.select_strategy(perception_data)
        decision = strategy.evaluate(perception_data, self.repository)
        self._execute_gate_action(decision)
        self._log_event(decision, strategy, perception_data)
        return decision

    def process_frame(self, frame: np.ndarray) -> AccessDecision:
        """Run the perception pipeline on a frame and evaluate gate access."""
        perception_data = self.pipeline.process_frame(frame)
        return self.evaluate_perception(perception_data)

    def _execute_gate_action(self, decision: AccessDecision) -> None:
        """Execute (simulate) hardware gate actuator command based on decision."""
        logger.info(
            "Gate command executed: action=%s reason=%r plate=%r",
            decision["action"],
            decision["reason"],
            decision["plate"],
        )

    def _log_event(
        self,
        decision: AccessDecision,
        strategy: AccessStrategy,
        perception_data: Mapping[str, Any],
    ) -> None:
        """Persist emergency bypass audit record post-decision and emit runtime audit log."""
        if (
            isinstance(strategy, EmergencyPriorityStrategy)
            and decision["action"] == "OPEN"
        ):
            self.repository.log_access(
                decision["plate"] or "EMERGENCY",
                decision["reason"],
            )

        logger.info(
            "Access event logged: strategy=%s action=%s reason=%r plate=%r confidence=%.3f",
            type(strategy).__name__,
            decision["action"],
            decision["reason"],
            decision["plate"],
            float(perception_data.get("confidence", 0.0)),
        )

    def close(self) -> None:
        """Close the repository connection if owned by this controller."""
        if self._owns_repository and self.repository is not None:
            self.repository.close()

    def __enter__(self) -> GateController:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

