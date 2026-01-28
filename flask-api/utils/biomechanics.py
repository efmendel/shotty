"""
Biomechanical utility functions for kinematic chain analysis.

Provides hip/shoulder rotation, knee bend, trunk lean, and upper arm angle
calculations used in tennis swing phase detection and analysis.
"""

import math


def calculate_hip_rotation(landmarks: dict) -> float:
    """
    Calculate hip rotation angle relative to camera plane.

    Args:
        landmarks: Dictionary with 'left_hip' and 'right_hip' entries,
            each containing 'x', 'y', 'z' coordinates.

    Returns:
        Hip rotation angle in degrees (-180 to 180).
        0 = hips parallel to camera.
        Positive = right hip forward.
        Returns 0.0 if required landmarks are missing.
    """
    try:
        left_hip = landmarks.get("left_hip")
        right_hip = landmarks.get("right_hip")

        if not left_hip or not right_hip:
            return 0.0

        z_diff = right_hip["z"] - left_hip["z"]
        x_diff = right_hip["x"] - left_hip["x"]

        angle_rad = math.atan2(z_diff, x_diff)
        return math.degrees(angle_rad)

    except (KeyError, TypeError, ValueError):
        return 0.0


def calculate_shoulder_rotation(landmarks: dict) -> float:
    """
    Calculate shoulder rotation angle relative to camera plane.

    Args:
        landmarks: Dictionary with 'left_shoulder' and 'right_shoulder' entries,
            each containing 'x', 'y', 'z' coordinates.

    Returns:
        Shoulder rotation angle in degrees (-180 to 180).
        0 = shoulders parallel to camera.
        Positive = right shoulder forward.
        Returns 0.0 if required landmarks are missing.
    """
    try:
        left_shoulder = landmarks.get("left_shoulder")
        right_shoulder = landmarks.get("right_shoulder")

        if not left_shoulder or not right_shoulder:
            return 0.0

        z_diff = right_shoulder["z"] - left_shoulder["z"]
        x_diff = right_shoulder["x"] - left_shoulder["x"]

        angle_rad = math.atan2(z_diff, x_diff)
        return math.degrees(angle_rad)

    except (KeyError, TypeError, ValueError):
        return 0.0


def calculate_knee_bend(landmarks: dict, side: str = "right") -> float:
    """
    Calculate knee bend angle for the specified leg.

    The knee bend is the angle formed by hip-knee-ankle. A straight leg
    is ~180 degrees, a bent knee is less.

    Args:
        landmarks: Dictionary containing hip, knee, and ankle landmarks
            with 'x', 'y', 'z' coordinates.
        side: Which leg to measure - 'left' or 'right'.

    Returns:
        Knee bend angle in degrees (0-180).
        180 = fully straight leg.
        90 = leg bent at right angle.
        Returns 180.0 if required landmarks are missing.
    """
    try:
        hip = landmarks.get(f"{side}_hip")
        knee = landmarks.get(f"{side}_knee")
        ankle = landmarks.get(f"{side}_ankle")

        if not hip or not knee or not ankle:
            return 180.0

        # Vector from knee to hip
        hip_vector = (
            hip["x"] - knee["x"],
            hip["y"] - knee["y"],
            hip["z"] - knee["z"],
        )

        # Vector from knee to ankle
        ankle_vector = (
            ankle["x"] - knee["x"],
            ankle["y"] - knee["y"],
            ankle["z"] - knee["z"],
        )

        dot_product = sum(h * a for h, a in zip(hip_vector, ankle_vector))

        hip_mag = math.sqrt(sum(h * h for h in hip_vector))
        ankle_mag = math.sqrt(sum(a * a for a in ankle_vector))

        if hip_mag == 0 or ankle_mag == 0:
            return 180.0

        cos_angle = max(-1.0, min(1.0, dot_product / (hip_mag * ankle_mag)))
        return math.degrees(math.acos(cos_angle))

    except (KeyError, TypeError, ValueError):
        return 180.0


def calculate_trunk_lean(landmarks: dict) -> float:
    """
    Calculate trunk lean angle relative to vertical.

    Measured by the angle between hip midpoint and shoulder midpoint
    relative to vertical.

    Args:
        landmarks: Dictionary with 'left_hip', 'right_hip',
            'left_shoulder', 'right_shoulder' entries.

    Returns:
        Trunk lean angle in degrees (-90 to 90).
        0 = perfectly upright.
        Positive = leaning backward.
        Negative = leaning forward.
        Returns 0.0 if required landmarks are missing.
    """
    try:
        left_hip = landmarks.get("left_hip")
        right_hip = landmarks.get("right_hip")
        left_shoulder = landmarks.get("left_shoulder")
        right_shoulder = landmarks.get("right_shoulder")

        if not all([left_hip, right_hip, left_shoulder, right_shoulder]):
            return 0.0

        # Calculate midpoints
        hip_mid_y = (left_hip["y"] + right_hip["y"]) / 2
        hip_mid_z = (left_hip["z"] + right_hip["z"]) / 2

        shoulder_mid_y = (left_shoulder["y"] + right_shoulder["y"]) / 2
        shoulder_mid_z = (left_shoulder["z"] + right_shoulder["z"]) / 2

        # Trunk vector (hips to shoulders)
        dy = shoulder_mid_y - hip_mid_y
        dz = shoulder_mid_z - hip_mid_z

        # Lean angle in z-y plane
        lean_angle_rad = math.atan2(dz, -dy)
        return math.degrees(lean_angle_rad)

    except (KeyError, TypeError, ValueError):
        return 0.0


def calculate_upper_arm_angle(landmarks: dict, side: str = "right") -> float:
    """
    Calculate upper arm angle relative to torso.

    Measures the angle between the shoulder-elbow line and the vertical
    trunk line. Useful for measuring arm elevation.

    Args:
        landmarks: Dictionary containing shoulder, elbow, and hip landmarks.
        side: Which arm to measure - 'left' or 'right'.

    Returns:
        Upper arm angle in degrees (0-180).
        0 = arm pointing straight down.
        90 = arm horizontal.
        180 = arm pointing straight up.
        Returns 0.0 if required landmarks are missing.
    """
    try:
        shoulder = landmarks.get(f"{side}_shoulder")
        elbow = landmarks.get(f"{side}_elbow")
        hip = landmarks.get(f"{side}_hip")

        if not shoulder or not elbow or not hip:
            return 0.0

        # Upper arm vector (shoulder to elbow)
        arm_vector = (
            elbow["x"] - shoulder["x"],
            elbow["y"] - shoulder["y"],
            elbow["z"] - shoulder["z"],
        )

        # Vertical reference vector (shoulder toward hip)
        vertical_vector = (
            0,
            hip["y"] - shoulder["y"],
            0,
        )

        dot_product = sum(a * v for a, v in zip(arm_vector, vertical_vector))

        arm_mag = math.sqrt(sum(a * a for a in arm_vector))
        vertical_mag = math.sqrt(sum(v * v for v in vertical_vector))

        if arm_mag == 0 or vertical_mag == 0:
            return 0.0

        cos_angle = max(-1.0, min(1.0, dot_product / (arm_mag * vertical_mag)))
        return math.degrees(math.acos(cos_angle))

    except (KeyError, TypeError, ValueError):
        return 0.0


def create_sample_landmarks() -> dict:
    """
    Create sample landmarks for testing purposes.

    Returns:
        Dictionary with all 12 required landmarks in a neutral standing position.
    """
    return {
        "left_hip": {"x": 0.4, "y": 0.5, "z": 0.0, "visibility": 0.9},
        "right_hip": {"x": 0.6, "y": 0.5, "z": 0.0, "visibility": 0.9},
        "left_shoulder": {"x": 0.4, "y": 0.3, "z": 0.0, "visibility": 0.9},
        "right_shoulder": {"x": 0.6, "y": 0.3, "z": 0.0, "visibility": 0.9},
        "left_elbow": {"x": 0.35, "y": 0.4, "z": 0.0, "visibility": 0.9},
        "right_elbow": {"x": 0.65, "y": 0.4, "z": 0.0, "visibility": 0.9},
        "left_wrist": {"x": 0.35, "y": 0.5, "z": 0.0, "visibility": 0.9},
        "right_wrist": {"x": 0.65, "y": 0.5, "z": 0.0, "visibility": 0.9},
        "left_knee": {"x": 0.4, "y": 0.7, "z": 0.0, "visibility": 0.9},
        "right_knee": {"x": 0.6, "y": 0.7, "z": 0.0, "visibility": 0.9},
        "left_ankle": {"x": 0.4, "y": 0.9, "z": 0.0, "visibility": 0.9},
        "right_ankle": {"x": 0.6, "y": 0.9, "z": 0.0, "visibility": 0.9},
    }