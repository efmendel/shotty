"""Tests for utils/biomechanics.py - kinematic chain calculations."""

import pytest
from utils.biomechanics import (
    calculate_hip_rotation,
    calculate_shoulder_rotation,
    calculate_knee_bend,
    calculate_trunk_lean,
    calculate_upper_arm_angle,
    create_sample_landmarks,
)


class TestHipRotation:
    """Tests for calculate_hip_rotation."""

    def test_neutral_position(self):
        landmarks = {
            "left_hip": {"x": 0.4, "y": 0.5, "z": 0.0},
            "right_hip": {"x": 0.6, "y": 0.5, "z": 0.0},
        }
        assert abs(calculate_hip_rotation(landmarks)) < 5

    def test_right_hip_forward(self):
        landmarks = {
            "left_hip": {"x": 0.4, "y": 0.5, "z": 0.1},
            "right_hip": {"x": 0.6, "y": 0.5, "z": -0.1},
        }
        assert calculate_hip_rotation(landmarks) < 0

    def test_left_hip_forward(self):
        landmarks = {
            "left_hip": {"x": 0.4, "y": 0.5, "z": -0.1},
            "right_hip": {"x": 0.6, "y": 0.5, "z": 0.1},
        }
        assert calculate_hip_rotation(landmarks) > 0

    def test_missing_landmarks(self):
        assert calculate_hip_rotation({"left_hip": {"x": 0.4, "y": 0.5, "z": 0}}) == 0.0

    def test_valid_range(self):
        landmarks = {
            "left_hip": {"x": 0.4, "y": 0.5, "z": 10.0},
            "right_hip": {"x": 0.6, "y": 0.5, "z": -10.0},
        }
        angle = calculate_hip_rotation(landmarks)
        assert -180 <= angle <= 180


class TestShoulderRotation:
    """Tests for calculate_shoulder_rotation."""

    def test_neutral_position(self):
        landmarks = {
            "left_shoulder": {"x": 0.4, "y": 0.3, "z": 0.0},
            "right_shoulder": {"x": 0.6, "y": 0.3, "z": 0.0},
        }
        assert abs(calculate_shoulder_rotation(landmarks)) < 5

    def test_right_shoulder_forward(self):
        landmarks = {
            "left_shoulder": {"x": 0.4, "y": 0.3, "z": 0.15},
            "right_shoulder": {"x": 0.6, "y": 0.3, "z": -0.15},
        }
        assert calculate_shoulder_rotation(landmarks) < 0

    def test_left_shoulder_forward(self):
        landmarks = {
            "left_shoulder": {"x": 0.4, "y": 0.3, "z": -0.15},
            "right_shoulder": {"x": 0.6, "y": 0.3, "z": 0.15},
        }
        assert calculate_shoulder_rotation(landmarks) > 0

    def test_missing_landmarks(self):
        assert calculate_shoulder_rotation({}) == 0.0


class TestKneeBend:
    """Tests for calculate_knee_bend."""

    def test_straight_leg(self):
        landmarks = {
            "right_hip": {"x": 0.6, "y": 0.5, "z": 0},
            "right_knee": {"x": 0.6, "y": 0.7, "z": 0},
            "right_ankle": {"x": 0.6, "y": 0.9, "z": 0},
        }
        assert 170 < calculate_knee_bend(landmarks, side="right") <= 180

    def test_90_degree_bend(self):
        landmarks = {
            "right_hip": {"x": 0.6, "y": 0.5, "z": 0},
            "right_knee": {"x": 0.6, "y": 0.7, "z": 0},
            "right_ankle": {"x": 0.7, "y": 0.7, "z": 0},
        }
        assert 85 < calculate_knee_bend(landmarks, side="right") < 95

    def test_left_leg(self):
        landmarks = {
            "left_hip": {"x": 0.4, "y": 0.5, "z": 0},
            "left_knee": {"x": 0.4, "y": 0.7, "z": 0},
            "left_ankle": {"x": 0.4, "y": 0.9, "z": 0},
        }
        assert 170 < calculate_knee_bend(landmarks, side="left") <= 180

    def test_missing_landmarks(self):
        landmarks = {"right_hip": {"x": 0.6, "y": 0.5, "z": 0}}
        assert calculate_knee_bend(landmarks, side="right") == 180.0

    def test_valid_range(self):
        landmarks = {
            "right_hip": {"x": 0.6, "y": 0.5, "z": 0},
            "right_knee": {"x": 0.6, "y": 0.7, "z": 0},
            "right_ankle": {"x": 0.7, "y": 0.7, "z": 0},
        }
        angle = calculate_knee_bend(landmarks, side="right")
        assert 0 <= angle <= 180


class TestTrunkLean:
    """Tests for calculate_trunk_lean."""

    def test_upright_posture(self):
        landmarks = {
            "left_hip": {"x": 0.4, "y": 0.5, "z": 0.0},
            "right_hip": {"x": 0.6, "y": 0.5, "z": 0.0},
            "left_shoulder": {"x": 0.4, "y": 0.3, "z": 0.0},
            "right_shoulder": {"x": 0.6, "y": 0.3, "z": 0.0},
        }
        assert abs(calculate_trunk_lean(landmarks)) < 10

    def test_forward_lean(self):
        landmarks = {
            "left_hip": {"x": 0.4, "y": 0.5, "z": 0.0},
            "right_hip": {"x": 0.6, "y": 0.5, "z": 0.0},
            "left_shoulder": {"x": 0.4, "y": 0.3, "z": -0.1},
            "right_shoulder": {"x": 0.6, "y": 0.3, "z": -0.1},
        }
        assert calculate_trunk_lean(landmarks) < 0

    def test_backward_lean(self):
        landmarks = {
            "left_hip": {"x": 0.4, "y": 0.5, "z": 0.0},
            "right_hip": {"x": 0.6, "y": 0.5, "z": 0.0},
            "left_shoulder": {"x": 0.4, "y": 0.3, "z": 0.1},
            "right_shoulder": {"x": 0.6, "y": 0.3, "z": 0.1},
        }
        assert calculate_trunk_lean(landmarks) > 0

    def test_missing_landmarks(self):
        assert calculate_trunk_lean({"left_hip": {"x": 0.4, "y": 0.5, "z": 0}}) == 0.0


class TestUpperArmAngle:
    """Tests for calculate_upper_arm_angle."""

    def test_arm_hanging_down(self):
        landmarks = {
            "right_shoulder": {"x": 0.6, "y": 0.3, "z": 0},
            "right_elbow": {"x": 0.6, "y": 0.5, "z": 0},
            "right_hip": {"x": 0.6, "y": 0.6, "z": 0},
        }
        assert calculate_upper_arm_angle(landmarks, side="right") < 20

    def test_arm_horizontal(self):
        landmarks = {
            "right_shoulder": {"x": 0.6, "y": 0.3, "z": 0},
            "right_elbow": {"x": 0.8, "y": 0.3, "z": 0},
            "right_hip": {"x": 0.6, "y": 0.6, "z": 0},
        }
        assert 80 < calculate_upper_arm_angle(landmarks, side="right") < 100

    def test_arm_raised_up(self):
        landmarks = {
            "right_shoulder": {"x": 0.6, "y": 0.3, "z": 0},
            "right_elbow": {"x": 0.6, "y": 0.1, "z": 0},
            "right_hip": {"x": 0.6, "y": 0.6, "z": 0},
        }
        assert calculate_upper_arm_angle(landmarks, side="right") > 160

    def test_left_arm(self):
        landmarks = {
            "left_shoulder": {"x": 0.4, "y": 0.3, "z": 0},
            "left_elbow": {"x": 0.4, "y": 0.5, "z": 0},
            "left_hip": {"x": 0.4, "y": 0.6, "z": 0},
        }
        assert calculate_upper_arm_angle(landmarks, side="left") < 20

    def test_missing_landmarks(self):
        assert calculate_upper_arm_angle({}, side="right") == 0.0

    def test_valid_range(self):
        landmarks = {
            "right_shoulder": {"x": 0.6, "y": 0.3, "z": 0},
            "right_elbow": {"x": 0.8, "y": 0.3, "z": 0},
            "right_hip": {"x": 0.6, "y": 0.6, "z": 0},
        }
        angle = calculate_upper_arm_angle(landmarks, side="right")
        assert 0 <= angle <= 180


class TestSampleLandmarks:
    """Tests for create_sample_landmarks helper."""

    def test_all_landmarks_present(self):
        landmarks = create_sample_landmarks()
        required = [
            "left_hip", "right_hip",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
        ]
        for name in required:
            assert name in landmarks

    def test_landmark_structure(self):
        landmarks = create_sample_landmarks()
        for name, lm in landmarks.items():
            assert "x" in lm and "y" in lm and "z" in lm
            assert "visibility" in lm

    def test_works_with_all_functions(self):
        landmarks = create_sample_landmarks()
        assert isinstance(calculate_hip_rotation(landmarks), float)
        assert isinstance(calculate_shoulder_rotation(landmarks), float)
        assert isinstance(calculate_knee_bend(landmarks), float)
        assert isinstance(calculate_trunk_lean(landmarks), float)
        assert isinstance(calculate_upper_arm_angle(landmarks), float)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dict(self):
        assert calculate_hip_rotation({}) == 0.0
        assert calculate_shoulder_rotation({}) == 0.0
        assert calculate_knee_bend({}) == 180.0
        assert calculate_trunk_lean({}) == 0.0
        assert calculate_upper_arm_angle({}) == 0.0

    def test_none_values(self):
        landmarks = {"left_hip": None, "right_hip": None}
        assert calculate_hip_rotation(landmarks) == 0.0

    def test_malformed_data(self):
        landmarks = {"left_hip": {"x": "invalid"}}
        assert calculate_hip_rotation(landmarks) == 0.0

    def test_extreme_z_values(self):
        landmarks = {
            "left_hip": {"x": 0.4, "y": 0.5, "z": 10.0},
            "right_hip": {"x": 0.6, "y": 0.5, "z": -10.0},
        }
        angle = calculate_hip_rotation(landmarks)
        assert isinstance(angle, float)
        assert -180 <= angle <= 180