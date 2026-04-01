"""
Application configuration.

Single source of truth for all runtime parameters.
Loaded once at startup from environment variables / .env file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


@dataclass(frozen=True)
class AppConfig:
    """
    All configuration in one place.
    Frozen dataclass — config is read-only after creation.

    Usage:
        config = AppConfig.from_env()
    """

    # --- OctoPrint ---
    octoprint_url: str
    octoprint_api_key: str

    # --- Model ---
    model_path: Path

    # --- Capture loop ---
    capture_interval_sec: int

    # --- Detection thresholds ---
    confidence_threshold: float    # minimum score to register a detection
    persistence_window: int        # consecutive detections needed to trigger pause
    iou_threshold: float           # NMS overlap threshold

    # --- Active classes ---
    # Only these classes can trigger a printer pause.
    # Others are detected and logged but never cause action.
    active_classes: frozenset[str]

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> "AppConfig":
        """
        Load config from environment variables.
        Reads .env file if present, then falls back to actual env vars.
        """
        if env_file:
            load_dotenv(env_file, override=False)

        active_raw = os.getenv(
            "ACTIVE_CLASSES",
            "spaghetti,stringing,warping,layer_shift"
        )

        return cls(
            octoprint_url=_require("OCTOPRINT_URL"),
            octoprint_api_key=_require("OCTOPRINT_API_KEY"),
            model_path=Path(os.getenv("MODEL_PATH", "models/model.onnx")),
            capture_interval_sec=int(os.getenv("CAPTURE_INTERVAL_SEC", "20")),
            confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.65")),
            persistence_window=int(os.getenv("PERSISTENCE_WINDOW", "3")),
            iou_threshold=float(os.getenv("IOU_THRESHOLD", "0.45")),
            active_classes=frozenset(
                c.strip() for c in active_raw.split(",") if c.strip()
            ),
        )


def _require(key: str) -> str:
    """Raise immediately if a required env var is missing."""
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill in the values."
        )
    return value
