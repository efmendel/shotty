"""
Geometry utility functions for swing analysis.

Provides angle calculation, velocity measurement, and body position
helpers used throughout the swing analysis pipeline.
"""

import math

import numpy as np


def calculate_angle(point_a: dict, point_b: dict, point_c: dict) -> float:
    """
    Calculate the angle at point B formed by points A, B, C.

    Args:
        point_a: First point with 'x' and 'y' keys.
        point_b: Vertex point with 'x' and 'y' keys.
        point_c: Third point with 'x' and 'y' keys.

    Returns:
        Angle in degrees (0-180).
    """
    a = np.array([point_a["x"], point_a["y"]])
    b = np.array([point_b["x"], point_b["y"]])
    c = np.array([point_c["x"], point_c["y"]])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)

    return float(np.degrees(angle))


def calculate_velocity(current_pos: dict, previous_pos: dict, time_delta: float) -> float:
    """
    Calculate velocity between two positions.

    Args:
        current_pos: Current position with 'x' and 'y' keys.
        previous_pos: Previous position with 'x' and 'y' keys.
        time_delta: Time between positions in seconds.

    Returns:
        Velocity in units per second. Returns 0 if time_delta is 0.
    """
    if time_delta == 0:
        return 0

    dx = current_pos["x"] - previous_pos["x"]
    dy = current_pos["y"] - previous_pos["y"]

    distance = math.sqrt(dx**2 + dy**2)
    return distance / time_delta


def get_body_center_x(left_shoulder: dict, right_shoulder: dict) -> float:
    """
    Get the x-coordinate of body center (midpoint between shoulders).

    Args:
        left_shoulder: Left shoulder position with 'x' key.
        right_shoulder: Right shoulder position with 'x' key.

    Returns:
        X-coordinate of body center.
    """
    return (left_shoulder["x"] + right_shoulder["x"]) / 2


def is_wrist_behind_body(wrist: dict, left_shoulder: dict, right_shoulder: dict) -> bool:
    """
    Check if wrist is behind the body center line.

    For a right-handed forehand from right side view, the wrist is
    "behind" if its x-coordinate is less than body center.

    Args:
        wrist: Wrist position with 'x' key.
        left_shoulder: Left shoulder position with 'x' key.
        right_shoulder: Right shoulder position with 'x' key.

    Returns:
        True if wrist is behind body center.
    """
    body_center_x = get_body_center_x(left_shoulder, right_shoulder)
    return wrist["x"] < body_center_x


def calculate_shoulder_rotation(left_shoulder: dict, right_shoulder: dict) -> float:
    """
    Calculate shoulder rotation angle relative to horizontal.

    Args:
        left_shoulder: Left shoulder position with 'x' and 'y' keys.
        right_shoulder: Right shoulder position with 'x' and 'y' keys.

    Returns:
        Rotation angle in degrees.
    """
    dx = right_shoulder["x"] - left_shoulder["x"]
    dy = right_shoulder["y"] - left_shoulder["y"]

    angle = math.atan2(dy, dx)
    return math.degrees(angle)