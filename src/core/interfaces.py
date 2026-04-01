"""
Core interfaces for PrintGuard.

All components depend on these abstractions, never on concrete implementations.
This allows testing without hardware and swapping implementations without
touching business logic (Dependency Inversion Principle).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """
    A single defect detection from one inference cycle.
    Immutable by design — created by IDetector, consumed by PrintStateTracker.
    """
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]   # x1, y1, x2, y2 in original image pixels
    is_active: bool                    # if False: logged but never triggers a pause


@dataclass(frozen=True)
class PrinterStatus:
    """
    Snapshot of printer state returned by IPrinterClient.get_status().
    Immutable — represents a point-in-time observation.
    """
    is_printing: bool
    is_paused: bool
    current_layer: int | None
    completion_pct: float | None
    job_name: str | None


@runtime_checkable
class IDetector(Protocol):
    """
    Runs defect detection on a single frame.
    Implementations: YoloDetector (production), MockDetector (tests).
    """
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Args:
            frame: BGR numpy array, any resolution.
        Returns:
            List of detections. Empty list if nothing found.
        """
        ...


@runtime_checkable
class IPrinterClient(Protocol):
    """
    Controls and queries the 3D printer.
    Implementations: OctoPrintClient (production), MockPrinterClient (tests).
    """
    def get_status(self) -> PrinterStatus | None:
        """Returns current printer state, or None if unreachable."""
        ...

    def pause(self) -> bool:
        """Pauses the current print. Returns True on success."""
        ...

    def resume(self) -> bool:
        """Resumes a paused print. Returns True on success."""
        ...


@runtime_checkable
class IFrameSource(Protocol):
    """
    Provides frames for inference.
    Implementations: WebcamSource (production), FileSource / MockSource (tests).
    """
    def capture(self) -> np.ndarray | None:
        """
        Captures one frame.
        Returns None if source is unavailable (camera disconnected etc).
        """
        ...


@runtime_checkable
class ILogger(Protocol):
    """
    Persists cycle data and alerts.
    Implementations: SQLiteLogger (production), InMemoryLogger (tests).
    """
    def log_cycle(
        self,
        layer: int | None,
        detections: list[Detection],
        action: str | None,
    ) -> None:
        """Records one inference cycle."""
        ...

    def log_alert(
        self,
        layer: int | None,
        defect_class: str,
        confidence: float,
    ) -> int:
        """
        Records a pause-triggering alert.
        Returns the alert ID for later verdict updates.
        """
        ...

    def set_verdict(self, alert_id: int, verdict: str) -> None:
        """
        Operator feedback: 'confirm' or 'dismiss'.
        Used for future fine-tuning data collection.
        """
        ...
