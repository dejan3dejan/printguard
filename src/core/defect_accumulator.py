from collections import deque
from typing import Dict, Optional, Tuple


class DefectAccumulator:
    """Pure, standalone defect persistence logic.
    
    Zero I/O, zero external dependencies → 100% testable in isolation.
    """

    def __init__(self, confidence_threshold: float, persistence_window: int) -> None:
        if persistence_window < 1:
            raise ValueError("persistence_window must be at least 1")
        if not (0.0 < confidence_threshold <= 1.0):
            raise ValueError("confidence_threshold must be in range (0, 1]")

        self.confidence_threshold = confidence_threshold
        self.persistence_window = persistence_window
        # class_name → deque of last N confidence scores
        self._history: Dict[str, deque[float]] = {}

    def update(self, class_name: str, confidence: float) -> None:
        """Call every frame for every class (use 0.0 if class not detected)."""
        if class_name not in self._history:
            self._history[class_name] = deque(maxlen=self.persistence_window)
        self._history[class_name].append(confidence)

    def should_pause(self) -> Tuple[bool, Optional[str]]:
        """Returns (pause_needed: bool, triggering_class: str | None)"""
        for class_name, hist in self._history.items():
            if (
                len(hist) == self.persistence_window
                and all(c >= self.confidence_threshold for c in hist)
            ):
                return True, class_name
        return False, None

    def reset(self, class_name: str) -> None:
        """Reset counter for one class (call after operator resumes)."""
        if class_name in self._history:
            self._history[class_name].clear()

    def reset_all(self) -> None:
        """Full reset (used on job start / state change)."""
        for hist in self._history.values():
            hist.clear()