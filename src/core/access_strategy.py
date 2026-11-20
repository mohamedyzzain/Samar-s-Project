"""Strategy Design Pattern definitions for SafeGate AI gate access decisions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal, Mapping, TypedDict

from src.database.repository import GateRepository


class AccessDecision(TypedDict):
    """Structured decision payload returned by AccessStrategy evaluations."""

    action: Literal["OPEN", "DENY"]
    reason: str
    plate: str | None


class AccessStrategy(ABC):
    """Abstract base strategy for evaluating gate access requests."""

    @abstractmethod
    def evaluate(
        self,
        perception_data: Mapping[str, Any],
        repository: GateRepository,
    ) -> AccessDecision:
        """Evaluate perception output against policy/persistence and return a decision."""


class EmergencyPriorityStrategy(AccessStrategy):
    """Zero-latency bypass strategy that opens immediately for emergency vehicles."""

    def evaluate(
        self,
        perception_data: Mapping[str, Any],
        repository: GateRepository,
    ) -> AccessDecision:
        """Return OPEN immediately when is_emergency is True without OCR or DB lookup."""
        if bool(perception_data.get("is_emergency", False)):
            return {
                "action": "OPEN",
                "reason": "Emergency Vehicle Bypass",
                "plate": "EMERGENCY",
            }

        return {
            "action": "DENY",
            "reason": "Not an Emergency Vehicle",
            "plate": perception_data.get("plate_text"),
        }


class ResidentVerificationStrategy(AccessStrategy):
    """Standard verification strategy querying GateRepository for resident plates."""

    def evaluate(
        self,
        perception_data: Mapping[str, Any],
        repository: GateRepository,
    ) -> AccessDecision:
        """Verify plate_text against Residents, log the attempt, and return OPEN or DENY."""
        raw_plate = perception_data.get("plate_text")
        if raw_plate is None or not str(raw_plate).strip():
            return {
                "action": "DENY",
                "reason": "No Readable License Plate",
                "plate": None,
            }

        plate_text = str(raw_plate).strip()
        is_allowed = repository.check_and_log_vehicle(plate_text)
        if is_allowed:
            return {
                "action": "OPEN",
                "reason": "Authorized Resident",
                "plate": plate_text,
            }

        return {
            "action": "DENY",
            "reason": "Unknown/Visitor",
            "plate": plate_text,
        }

