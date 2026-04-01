"""
Smoke tests for interfaces and config.

Verifies that:
- Detection and PrinterStatus are properly immutable
- AppConfig loads from env correctly
- AppConfig raises on missing required vars
- Protocol implementations are detectable at runtime
"""
import pytest
import os
import numpy as np
from dataclasses import FrozenInstanceError

from src.core.interfaces import Detection, PrinterStatus, IDetector, IPrinterClient
from src.core.config import AppConfig


class TestDetection:
    def test_creation(self):
        d = Detection(
            class_name="spaghetti",
            confidence=0.82,
            bbox=(10, 20, 100, 200),
            is_active=True,
        )
        assert d.class_name == "spaghetti"
        assert d.confidence == 0.82
        assert d.is_active is True

    def test_immutable(self):
        d = Detection("warping", 0.7, (0, 0, 50, 50), True)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            d.confidence = 0.9  # type: ignore


class TestPrinterStatus:
    def test_creation(self):
        s = PrinterStatus(
            is_printing=True,
            is_paused=False,
            current_layer=12,
            completion_pct=34.5,
            job_name="benchy.gcode",
        )
        assert s.is_printing is True
        assert s.current_layer == 12

    def test_immutable(self):
        s = PrinterStatus(True, False, 5, 10.0, "test.gcode")
        with pytest.raises((FrozenInstanceError, AttributeError)):
            s.is_printing = False  # type: ignore


class TestAppConfig:
    def test_loads_from_env(self, monkeypatch):
        monkeypatch.setenv("OCTOPRINT_URL", "http://localhost:5000")
        monkeypatch.setenv("OCTOPRINT_API_KEY", "test-key-123")
        monkeypatch.setenv("CAPTURE_INTERVAL_SEC", "20")
        monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.65")
        monkeypatch.setenv("PERSISTENCE_WINDOW", "3")
        monkeypatch.setenv("IOU_THRESHOLD", "0.45")
        monkeypatch.setenv("ACTIVE_CLASSES", "spaghetti,warping")

        config = AppConfig.from_env(env_file=None)

        assert config.octoprint_url == "http://localhost:5000"
        assert config.capture_interval_sec == 20
        assert config.confidence_threshold == 0.65
        assert config.persistence_window == 3
        assert config.active_classes == frozenset({"spaghetti", "warping"})

    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("OCTOPRINT_URL", "http://localhost:5000")
        monkeypatch.setenv("OCTOPRINT_API_KEY", "key")

        config = AppConfig.from_env(env_file=None)

        assert config.capture_interval_sec == 20
        assert config.persistence_window == 3
        assert "spaghetti" in config.active_classes
        assert "layer_shift" in config.active_classes

    def test_raises_on_missing_url(self, monkeypatch):
        monkeypatch.delenv("OCTOPRINT_URL", raising=False)
        monkeypatch.delenv("OCTOPRINT_API_KEY", raising=False)

        with pytest.raises(EnvironmentError, match="OCTOPRINT_URL"):
            AppConfig.from_env(env_file=None)

    def test_active_classes_is_frozenset(self, monkeypatch):
        monkeypatch.setenv("OCTOPRINT_URL", "http://localhost:5000")
        monkeypatch.setenv("OCTOPRINT_API_KEY", "key")

        config = AppConfig.from_env(env_file=None)
        assert isinstance(config.active_classes, frozenset)

    def test_config_is_immutable(self, monkeypatch):
        monkeypatch.setenv("OCTOPRINT_URL", "http://localhost:5000")
        monkeypatch.setenv("OCTOPRINT_API_KEY", "key")

        config = AppConfig.from_env(env_file=None)
        with pytest.raises((FrozenInstanceError, AttributeError)):
            config.capture_interval_sec = 999  # type: ignore


class TestProtocolChecks:
    """Verify @runtime_checkable works for duck-typed implementations."""

    def test_detector_protocol(self):
        class MyDetector:
            def detect(self, frame: np.ndarray) -> list:
                return []

        assert isinstance(MyDetector(), IDetector)

    def test_printer_client_protocol(self):
        class MyClient:
            def get_status(self): return None
            def pause(self): return True
            def resume(self): return True

        assert isinstance(MyClient(), IPrinterClient)

    def test_missing_method_fails_protocol(self):
        class Incomplete:
            def detect(self, frame): return []
            # missing pause(), resume(), get_status()

        assert not isinstance(Incomplete(), IPrinterClient)
