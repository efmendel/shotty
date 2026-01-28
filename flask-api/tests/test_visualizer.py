"""Tests for services/visualizer.py - video annotation/rendering helpers."""

import pytest
from services.visualizer import (
    assign_phases_to_frames,
    get_phase_color,
)


class TestAssignPhasesToFrames:
    """Tests for assign_phases_to_frames."""

    def test_no_phases_detected(self):
        phases = {
            "unit_turn": None,
            "backswing": None,
            "forward_swing": None,
            "contact": None,
            "follow_through": None,
        }
        result = assign_phases_to_frames(phases, 10)
        for frame_num in range(1, 11):
            label, conf, reason = result[frame_num]
            assert conf == 0.0

    def test_all_phases_detected(self):
        phases = {
            "unit_turn": {"detected": True, "frame": 5, "confidence": 0.9, "reason": "Detected"},
            "backswing": {"detected": True, "frame": 15, "confidence": 0.85, "reason": "Detected"},
            "forward_swing": {"detected": True, "frame": 25, "confidence": 0.8, "reason": "Detected"},
            "contact": {"detected": True, "frame": 35, "confidence": 0.95, "reason": "Detected"},
            "follow_through": {"detected": True, "frame": 45, "confidence": 0.7, "reason": "Detected"},
        }
        result = assign_phases_to_frames(phases, 60)

        # Before unit turn -> Ready Position
        assert result[3][0] == "Ready Position"

        # At unit turn frame range
        assert "UNIT TURN" in result[5][0]

        # Backswing range
        assert "BACKSWING" in result[15][0]

        # Forward swing range
        assert "FORWARD SWING" in result[25][0]

        # Contact frame
        assert "CONTACT" in result[35][0]

        # After follow through
        assert "FINISH" in result[50][0]

    def test_partial_detection(self):
        phases = {
            "unit_turn": {"detected": True, "frame": 5, "confidence": 0.9, "reason": "Detected"},
            "backswing": {"detected": True, "frame": 15, "confidence": 0.85, "reason": "Detected"},
            "forward_swing": {"detected": False, "confidence": 0.0, "reason": "insufficient_velocity"},
            "contact": {"detected": False, "confidence": 0.0, "reason": "forward_swing_not_detected"},
            "follow_through": {"detected": False, "confidence": 0.0, "reason": "contact_not_detected"},
        }
        result = assign_phases_to_frames(phases, 30)
        # After backswing, should still show BACKSWING since no further phases detected
        assert "BACKSWING" in result[20][0]

    def test_contact_frame_exact(self):
        phases = {
            "unit_turn": {"detected": True, "frame": 5, "confidence": 0.9, "reason": "Detected"},
            "backswing": {"detected": True, "frame": 10, "confidence": 0.85, "reason": "Detected"},
            "forward_swing": {"detected": True, "frame": 15, "confidence": 0.8, "reason": "Detected"},
            "contact": {"detected": True, "frame": 20, "confidence": 0.95, "reason": "Detected"},
            "follow_through": {"detected": True, "frame": 25, "confidence": 0.7, "reason": "Detected"},
        }
        result = assign_phases_to_frames(phases, 30)
        assert "CONTACT" in result[20][0]
        assert result[20][1] == 0.95


class TestGetPhaseColor:
    """Tests for get_phase_color."""

    def test_analyzing_phase(self):
        color = get_phase_color("Analyzing...", 0.0)
        assert color == (128, 128, 128)

    def test_zero_confidence(self):
        color = get_phase_color("CONTACT", 0.0)
        assert color == (128, 128, 128)

    def test_backswing_high_confidence(self):
        color = get_phase_color("BACKSWING", 0.9)
        assert color == (255, 100, 0)

    def test_backswing_medium_confidence(self):
        color = get_phase_color("BACKSWING", 0.6)
        assert color == (200, 80, 0)

    def test_backswing_low_confidence(self):
        color = get_phase_color("BACKSWING", 0.3)
        assert color == (150, 60, 0)

    def test_forward_swing_high(self):
        color = get_phase_color("FORWARD SWING", 0.9)
        assert color == (0, 255, 0)

    def test_contact_high(self):
        color = get_phase_color("*** CONTACT ***", 0.9)
        assert color == (0, 0, 255)

    def test_follow_through_high(self):
        color = get_phase_color("FOLLOW THROUGH", 0.9)
        assert color == (0, 200, 255)

    def test_finish_high(self):
        color = get_phase_color("FINISH", 0.9)
        assert color == (0, 200, 255)

    def test_unit_turn(self):
        color = get_phase_color("UNIT TURN", 0.9)
        assert color == (255, 100, 0)

    def test_ready_position(self):
        color = get_phase_color("Ready Position", 0.9)
        assert color == (255, 100, 0)

    def test_unknown_phase_high_confidence(self):
        color = get_phase_color("Unknown Phase", 0.9)
        assert color == (0, 255, 0)

    def test_unknown_phase_medium_confidence(self):
        color = get_phase_color("Unknown Phase", 0.6)
        assert color == (0, 255, 255)

    def test_unknown_phase_low_confidence(self):
        color = get_phase_color("Unknown Phase", 0.3)
        assert color == (0, 0, 255)
