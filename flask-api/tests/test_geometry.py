"""Tests for utils/geometry.py - angle, velocity, and body position calculations."""

import math
import pytest
from utils.geometry import (
    calculate_angle,
    calculate_velocity,
    get_body_center_x,
    is_wrist_behind_body,
    calculate_shoulder_rotation,
)


class TestCalculateAngle:
    """Tests for calculate_angle function."""

    def test_right_angle(self):
        a = {"x": 0.0, "y": 1.0}
        b = {"x": 0.0, "y": 0.0}
        c = {"x": 1.0, "y": 0.0}
        assert abs(calculate_angle(a, b, c) - 90.0) < 0.1

    def test_straight_angle(self):
        a = {"x": 0.0, "y": 0.0}
        b = {"x": 0.5, "y": 0.0}
        c = {"x": 1.0, "y": 0.0}
        assert abs(calculate_angle(a, b, c) - 180.0) < 0.1

    def test_acute_angle(self):
        a = {"x": 0.0, "y": 1.0}
        b = {"x": 0.0, "y": 0.0}
        c = {"x": 1.0, "y": 1.0}
        angle = calculate_angle(a, b, c)
        assert 40 < angle < 50  # ~45 degrees

    def test_returns_degrees(self):
        a = {"x": 0.5, "y": 0.3}
        b = {"x": 0.6, "y": 0.5}
        c = {"x": 0.7, "y": 0.6}
        angle = calculate_angle(a, b, c)
        assert 0 <= angle <= 180


class TestCalculateVelocity:
    """Tests for calculate_velocity function."""

    def test_basic_velocity(self):
        pos1 = {"x": 0.0, "y": 0.0}
        pos2 = {"x": 3.0, "y": 4.0}
        vel = calculate_velocity(pos2, pos1, 1.0)
        assert abs(vel - 5.0) < 0.01

    def test_zero_time_delta(self):
        pos1 = {"x": 0.0, "y": 0.0}
        pos2 = {"x": 1.0, "y": 1.0}
        assert calculate_velocity(pos2, pos1, 0) == 0

    def test_no_movement(self):
        pos = {"x": 0.5, "y": 0.5}
        assert calculate_velocity(pos, pos, 1.0) == 0

    def test_frame_rate_velocity(self):
        pos1 = {"x": 0.5, "y": 0.5}
        pos2 = {"x": 0.6, "y": 0.6}
        vel = calculate_velocity(pos2, pos1, 1 / 30)  # 30fps
        expected = math.sqrt(0.1**2 + 0.1**2) / (1 / 30)
        assert abs(vel - expected) < 0.001


class TestGetBodyCenterX:
    """Tests for get_body_center_x function."""

    def test_symmetric_shoulders(self):
        left = {"x": 0.4}
        right = {"x": 0.6}
        assert abs(get_body_center_x(left, right) - 0.5) < 0.001

    def test_asymmetric_shoulders(self):
        left = {"x": 0.3}
        right = {"x": 0.7}
        assert abs(get_body_center_x(left, right) - 0.5) < 0.001


class TestIsWristBehindBody:
    """Tests for is_wrist_behind_body function."""

    def test_wrist_behind(self):
        wrist = {"x": 0.3}
        left_shoulder = {"x": 0.4}
        right_shoulder = {"x": 0.6}
        assert is_wrist_behind_body(wrist, left_shoulder, right_shoulder) is True

    def test_wrist_in_front(self):
        wrist = {"x": 0.7}
        left_shoulder = {"x": 0.4}
        right_shoulder = {"x": 0.6}
        assert is_wrist_behind_body(wrist, left_shoulder, right_shoulder) is False

    def test_wrist_at_center(self):
        wrist = {"x": 0.5}
        left_shoulder = {"x": 0.4}
        right_shoulder = {"x": 0.6}
        assert is_wrist_behind_body(wrist, left_shoulder, right_shoulder) is False


class TestCalculateShoulderRotation:
    """Tests for calculate_shoulder_rotation function."""

    def test_horizontal_shoulders(self):
        left = {"x": 0.4, "y": 0.5}
        right = {"x": 0.6, "y": 0.5}
        assert abs(calculate_shoulder_rotation(left, right)) < 0.1

    def test_tilted_shoulders(self):
        left = {"x": 0.4, "y": 0.5}
        right = {"x": 0.6, "y": 0.4}
        angle = calculate_shoulder_rotation(left, right)
        assert angle < 0  # Right shoulder higher = negative angle

    def test_returns_degrees(self):
        left = {"x": 0.4, "y": 0.5}
        right = {"x": 0.6, "y": 0.5}
        angle = calculate_shoulder_rotation(left, right)
        assert -180 <= angle <= 180