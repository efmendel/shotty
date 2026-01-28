"""
Video quality validation for swing analysis.

Assesses resolution, frame rate, brightness, and sharpness to determine
if a video is suitable for pose detection and swing analysis.
"""

from typing import Any, Dict, List

import cv2
import numpy as np


def check_video_quality(video_path: str) -> Dict[str, Any]:
    """
    Analyze video quality and return a comprehensive quality report.

    Checks resolution (>= 720p), frame rate (>= 24fps), brightness (>= 100/255),
    and sharpness via Laplacian variance (>= 100).

    Args:
        video_path: Path to the video file.

    Returns:
        Dictionary containing resolution, fps, brightness, sharpness,
        warnings list, and is_acceptable boolean.

    Raises:
        ValueError: If video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    warnings: List[str] = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width < 1280 or height < 720:
        warnings.append(f"Resolution {width}x{height} is below recommended 720p (1280x720)")

    if fps < 24:
        warnings.append(f"Frame rate {fps:.1f}fps is below recommended 24fps")

    sample_interval = max(1, frame_count // 30)
    sample_count = min(30, frame_count // sample_interval)

    brightness_values = []
    sharpness_values = []

    frame_idx = 0
    samples_collected = 0

    while samples_collected < sample_count:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        brightness_values.append(avg_brightness)

        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        sharpness_values.append(variance)

        frame_idx += sample_interval
        samples_collected += 1

    cap.release()

    avg_brightness = float(np.mean(brightness_values)) if brightness_values else 0.0
    avg_sharpness = float(np.mean(sharpness_values)) if sharpness_values else 0.0

    if avg_brightness < 100:
        warnings.append(
            f"Video is too dark (brightness: {avg_brightness:.1f}/255). Recommended: >= 100"
        )

    if avg_sharpness < 100:
        warnings.append(
            f"Video has motion blur or low sharpness (sharpness: {avg_sharpness:.1f}). "
            f"Recommended: >= 100"
        )

    is_acceptable = len(warnings) == 0

    return {
        "resolution": (width, height),
        "fps": fps,
        "brightness": avg_brightness,
        "sharpness": avg_sharpness,
        "warnings": warnings,
        "is_acceptable": is_acceptable,
    }