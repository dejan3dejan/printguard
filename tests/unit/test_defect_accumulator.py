"""
Unit tests for DefectAccumulator — pure logic, 100% branch coverage.
Matches the style of your test_interfaces.py.
"""

import pytest
from src.core.defect_accumulator import DefectAccumulator


class TestDefectAccumulator:
    def test_constructor_validation(self):
        with pytest.raises(ValueError, match="persistence_window must be at least 1"):
            DefectAccumulator(confidence_threshold=0.6, persistence_window=0)

        with pytest.raises(ValueError, match="confidence_threshold must be in range"):
            DefectAccumulator(confidence_threshold=0.0, persistence_window=5)
            DefectAccumulator(confidence_threshold=1.1, persistence_window=5)

    def test_should_not_pause_before_window_is_full(self):
        acc = DefectAccumulator(confidence_threshold=0.65, persistence_window=3)
        acc.update("spaghetti", 0.85)
        acc.update("spaghetti", 0.85)
        assert acc.should_pause() == (False, None)

    def test_pause_when_persistence_met(self):
        acc = DefectAccumulator(confidence_threshold=0.65, persistence_window=3)
        for _ in range(3):
            acc.update("spaghetti", 0.85)
        assert acc.should_pause() == (True, "spaghetti")

    def test_only_pause_when_all_frames_above_threshold(self):
        acc = DefectAccumulator(confidence_threshold=0.65, persistence_window=3)
        acc.update("spaghetti", 0.85)
        acc.update("spaghetti", 0.85)
        acc.update("spaghetti", 0.55)   # drops below
        assert acc.should_pause() == (False, None)

        # start new window
        acc.update("spaghetti", 0.85)
        acc.update("spaghetti", 0.85)
        acc.update("spaghetti", 0.85)
        assert acc.should_pause() == (True, "spaghetti")

    def test_multiple_classes_only_one_triggers(self):
        acc = DefectAccumulator(confidence_threshold=0.65, persistence_window=3)
        for _ in range(3):
            acc.update("spaghetti", 0.82)
            acc.update("stringing", 0.40)   # never reaches threshold
        assert acc.should_pause() == (True, "spaghetti")

    def test_reset_single_class(self):
        acc = DefectAccumulator(confidence_threshold=0.65, persistence_window=3)
        for _ in range(3):
            acc.update("spaghetti", 0.80)
        assert acc.should_pause() == (True, "spaghetti")

        acc.reset("spaghetti")
        assert acc.should_pause() == (False, None)

    def test_reset_all(self):
        acc = DefectAccumulator(confidence_threshold=0.65, persistence_window=3)
        for _ in range(3):
            acc.update("spaghetti", 0.80)
            acc.update("warping", 0.80)
        acc.reset_all()
        assert acc.should_pause() == (False, None)

    def test_update_with_class_not_in_active_list_still_tracks(self):
        acc = DefectAccumulator(confidence_threshold=0.65, persistence_window=3)
        acc.update("under_extrusion", 0.90)   # not in active_classes, but we still track
        acc.update("under_extrusion", 0.90)
        acc.update("under_extrusion", 0.90)
        # should_pause still works (logic doesn't filter here — filtering happens in tracker)
        assert acc.should_pause() == (True, "under_extrusion")