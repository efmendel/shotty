"""
Video annotation and rendering for swing analysis.

Provides functions to create annotated videos with swing phase overlays,
skeleton drawings, and metric displays.
"""

import logging
from typing import Any, Dict, Optional, Tuple

import cv2

from services.models import SwingAnalysisResults
from services.swing_analyzer import SwingAnalyzer
from services.video_processor import PRESET_DIFFICULT_VIDEO, VideoProcessor

logger = logging.getLogger(__name__)

# Upper body pose connections for drawing skeleton
# Each tuple is (start_landmark, end_landmark)
POSE_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]


def _draw_skeleton(frame, landmarks, width, height):
    """
    Draw pose skeleton on frame using extracted landmarks.

    Args:
        frame: OpenCV image (BGR).
        landmarks: Dictionary of landmark coordinates from VideoProcessor.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """
    if not landmarks:
        return

    # Draw connections (lines between joints)
    for start_name, end_name in POSE_CONNECTIONS:
        if start_name in landmarks and end_name in landmarks:
            start = landmarks[start_name]
            end = landmarks[end_name]

            # Convert normalized coords to pixel coords
            start_pt = (int(start["x"] * width), int(start["y"] * height))
            end_pt = (int(end["x"] * width), int(end["y"] * height))

            # Draw line (blue)
            cv2.line(frame, start_pt, end_pt, (255, 0, 0), 2)

    # Draw landmarks (joints as circles)
    for name, lm in landmarks.items():
        pt = (int(lm["x"] * width), int(lm["y"] * height))
        # Green for high visibility, yellow for medium, red for low
        if lm["visibility"] > 0.7:
            color = (0, 255, 0)
        elif lm["visibility"] > 0.5:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)
        cv2.circle(frame, pt, 5, color, -1)


def visualize_swing_phases(
    video_path: str,
    analysis_results: Optional[SwingAnalysisResults] = None,
    output_path: str = "output_annotated.mp4",
    use_adaptive: bool = False,
    velocity_threshold: float = 0.5,
    adaptive_percent: float = 0.15,
    contact_angle_min: int = 150,
    kinematic_chain_mode: bool = False,
    contact_detection_method: str = "velocity_peak",
) -> str:
    """
    Create annotated video with swing phases overlaid.

    Can accept pre-computed analysis results or perform analysis on the fly.

    Args:
        video_path: Path to input video.
        analysis_results: Pre-computed SwingAnalysisResults (optional).
        output_path: Path for output video.
        use_adaptive: Use adaptive velocity threshold.
        velocity_threshold: Fixed velocity threshold.
        adaptive_percent: Fraction of max velocity for threshold.
        contact_angle_min: Minimum elbow angle at contact.
        kinematic_chain_mode: Use multi-joint biomechanical analysis.
        contact_detection_method: Detection method for contact.

    Returns:
        Path to the output annotated video file.
    """
    if analysis_results is None:
        logger.info("Processing video for analysis...")
        processor = VideoProcessor(pose_config=PRESET_DIFFICULT_VIDEO)
        video_data = processor.process_video(video_path)

        logger.info("Analyzing swing phases...")
        analyzer = SwingAnalyzer(
            velocity_threshold=velocity_threshold,
            contact_angle_min=contact_angle_min,
            use_adaptive_velocity=use_adaptive,
            adaptive_velocity_percent=adaptive_percent,
            kinematic_chain_mode=kinematic_chain_mode,
            contact_detection_method=contact_detection_method,
        )
        analysis_results = analyzer.analyze_swing(video_data)
        phases = analysis_results.to_dict()["phases"]
    else:
        logger.info("Using pre-computed analysis results...")
        phases = analysis_results.to_dict()["phases"]
        processor = VideoProcessor(pose_config=PRESET_DIFFICULT_VIDEO)
        video_data = processor.process_video(video_path)

    logger.info("Creating annotated video...")

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_phases = assign_phases_to_frames(phases, video_data["frame_count"])

    # Build lookup for frame landmarks from video_data
    frame_landmarks = {}
    for frame_data in video_data["frames"]:
        frame_landmarks[frame_data["frame_number"]] = frame_data.get("landmarks")

    frame_number = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_number += 1

        # Draw skeleton using pre-extracted landmarks
        landmarks = frame_landmarks.get(frame_number)
        if landmarks:
            _draw_skeleton(frame, landmarks, width, height)

        phase_info = frame_phases.get(frame_number, ("Analyzing...", 0.0, "Unknown"))
        current_phase, phase_confidence, phase_reason = phase_info
        timestamp = frame_number / fps

        # Semi-transparent background for phase text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (700, 150), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        phase_color = get_phase_color(current_phase, phase_confidence)
        cv2.putText(
            frame, f"Phase: {current_phase}",
            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, phase_color, 3,
        )

        if phase_confidence > 0.0:
            cv2.putText(
                frame, f"Confidence: {phase_confidence:.2f}",
                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 2,
            )

        cv2.putText(
            frame, f"Time: {timestamp:.2f}s | Frame: {frame_number}",
            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        # Tracking quality overlay
        if "tracking_quality" in video_data:
            tq = video_data["tracking_quality"]
            detection_rate = tq.get("detection_rate", 0.0)

            overlay_quality = frame.copy()
            cv2.rectangle(overlay_quality, (width - 350, 10), (width - 10, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay_quality, 0.6, frame, 0.4, 0, frame)

            tracking_color = (
                (0, 255, 0) if detection_rate > 0.7
                else (0, 255, 255) if detection_rate > 0.5
                else (0, 0, 255)
            )
            cv2.putText(
                frame, f"Tracking: {detection_rate * 100:.1f}%",
                (width - 340, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, tracking_color, 2,
            )

            overall_confidence = analysis_results.get_overall_confidence()
            phases_detected = analysis_results.get_phases_detected_count()

            analysis_color = (
                (0, 255, 0) if overall_confidence > 0.7
                else (0, 255, 255) if overall_confidence > 0.5
                else (0, 0, 255)
            )
            cv2.putText(
                frame, f"Analysis: {overall_confidence * 100:.1f}%",
                (width - 340, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, analysis_color, 2,
            )
            cv2.putText(
                frame, f"Phases: {phases_detected}/5",
                (width - 340, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

        # Engine metrics during backswing
        if "BACKSWING" in current_phase or "UNIT TURN" in current_phase:
            _draw_engine_metrics(frame, analysis_results, width, height)

        # Tempo metrics during finish
        if "FINISH" in current_phase:
            _draw_tempo_metrics(frame, analysis_results, width, height)

        # Key frame marker
        key_frames = [
            p.get("frame")
            for p in phases.values()
            if p and isinstance(p, dict) and p.get("detected", False)
        ]
        if frame_number in key_frames:
            cv2.circle(frame, (width - 50, 160), 20, (0, 255, 255), -1)
            cv2.putText(
                frame, "KEY",
                (width - 80, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )

        out.write(frame)

        if frame_number % 30 == 0:
            logger.info("Annotating frame %d/%d...", frame_number, video_data["frame_count"])

    cap.release()
    out.release()

    logger.info("Annotated video saved to: %s", output_path)
    return output_path


def _draw_engine_metrics(frame, analysis_results, width, height):
    """Draw engine metrics overlay on frame during backswing phases."""
    engine_data = analysis_results.to_dict()["engine"]
    if not engine_data.get("hip_shoulder_separation"):
        return

    hip_shoulder_sep = engine_data["hip_shoulder_separation"].get("max_value", 0)

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, height - 120), (400, height - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame, f"Hip-Shoulder Sep: {hip_shoulder_sep:.1f}",
        (20, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2,
    )

    if engine_data.get("max_shoulder_rotation"):
        shoulder_rot = engine_data["max_shoulder_rotation"].get("value", 0)
        cv2.putText(
            frame, f"Shoulder Rotation: {shoulder_rot:.1f}",
            (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2,
        )

    if engine_data.get("max_hip_rotation"):
        hip_rot = engine_data["max_hip_rotation"].get("value", 0)
        cv2.putText(
            frame, f"Hip Rotation: {hip_rot:.1f}",
            (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 0), 2,
        )


def _draw_tempo_metrics(frame, analysis_results, width, height):
    """Draw tempo metrics overlay on frame during finish phase."""
    tempo_data = analysis_results.to_dict()["tempo"]
    if tempo_data.get("backswing_duration") is None:
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (10, height - 120), (450, height - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    backswing_dur = tempo_data["backswing_duration"]
    forward_dur = tempo_data.get("forward_swing_duration", 0)
    rhythm = tempo_data.get("swing_rhythm_ratio", 0)

    cv2.putText(
        frame, f"Backswing: {backswing_dur:.2f}s",
        (20, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
    )
    cv2.putText(
        frame, f"Forward Swing: {forward_dur:.2f}s",
        (20, height - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
    )
    cv2.putText(
        frame, f"Rhythm Ratio: {rhythm:.2f}",
        (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2,
    )


def assign_phases_to_frames(
    phases: Dict[str, Any], total_frames: int
) -> Dict[int, Tuple[str, float, str]]:
    """
    Assign phase labels with confidence to each frame number.

    Args:
        phases: Phase dict from SwingAnalysisResults.
        total_frames: Total number of frames in video.

    Returns:
        Dictionary mapping frame_number to (phase_label, confidence, reason).
    """
    frame_phases = {}

    def get_phase_info(phase_name):
        phase_data = phases.get(phase_name, {})
        if not phase_data or not isinstance(phase_data, dict):
            return None, 0.0, "Phase data not available"
        if not phase_data.get("detected", False):
            reason = phase_data.get("reason", "Not detected")
            return None, 0.0, reason
        frame = phase_data.get("frame", 0)
        confidence = phase_data.get("confidence", 0.0)
        return frame, confidence, "Detected"

    unit_turn_frame, unit_turn_conf, _ = get_phase_info("unit_turn")
    backswing_frame, backswing_conf, _ = get_phase_info("backswing")
    forward_swing_frame, forward_swing_conf, _ = get_phase_info("forward_swing")
    contact_frame, contact_conf, contact_reason = get_phase_info("contact")
    follow_through_frame, follow_through_conf, _ = get_phase_info("follow_through")

    unit_turn = unit_turn_frame if unit_turn_frame else 0
    backswing = backswing_frame if backswing_frame else 0
    forward_swing = forward_swing_frame if forward_swing_frame else 0
    contact = contact_frame if contact_frame else 0
    follow_through = follow_through_frame if follow_through_frame else 0

    for frame_num in range(1, total_frames + 1):
        if unit_turn == 0 and backswing == 0:
            frame_phases[frame_num] = ("Analyzing...", 0.0, "No phases detected")
        elif unit_turn > 0 and frame_num < unit_turn:
            frame_phases[frame_num] = ("Ready Position", 1.0, "Before swing starts")
        elif unit_turn > 0 and backswing > 0 and frame_num >= unit_turn and frame_num < backswing:
            frame_phases[frame_num] = ("UNIT TURN", unit_turn_conf, "Preparing")
        elif backswing > 0 and forward_swing > 0 and frame_num >= backswing and frame_num < forward_swing:
            frame_phases[frame_num] = ("BACKSWING", backswing_conf, "Loading")
        elif forward_swing > 0 and contact > 0 and frame_num >= forward_swing and frame_num < contact:
            frame_phases[frame_num] = ("FORWARD SWING", forward_swing_conf, "Accelerating")
        elif contact > 0 and frame_num == contact:
            frame_phases[frame_num] = ("*** CONTACT ***", contact_conf, contact_reason)
        elif contact > 0 and follow_through > 0 and frame_num > contact and frame_num < follow_through:
            frame_phases[frame_num] = ("FOLLOW THROUGH", contact_conf, "Decelerating")
        elif follow_through > 0 and frame_num >= follow_through:
            frame_phases[frame_num] = ("FINISH", follow_through_conf, "Recovery")
        else:
            if contact > 0 and frame_num > contact:
                frame_phases[frame_num] = ("FOLLOW THROUGH", contact_conf, "After contact")
            elif forward_swing > 0 and frame_num >= forward_swing:
                frame_phases[frame_num] = ("FORWARD SWING", forward_swing_conf, "Accelerating")
            elif backswing > 0 and frame_num >= backswing:
                frame_phases[frame_num] = ("BACKSWING", backswing_conf, "Loading")
            elif unit_turn > 0 and frame_num >= unit_turn:
                frame_phases[frame_num] = ("UNIT TURN", unit_turn_conf, "Preparing")
            else:
                frame_phases[frame_num] = ("Analyzing...", 0.0, "Unknown phase")

    return frame_phases


def get_phase_color(phase_name: str, confidence: float = 1.0) -> Tuple[int, int, int]:
    """
    Return BGR color based on phase and confidence level.

    Args:
        phase_name: Name of the current phase.
        confidence: Detection confidence (0.0-1.0).

    Returns:
        BGR color tuple.
    """
    if phase_name == "Analyzing..." or confidence == 0.0:
        return (128, 128, 128)

    if "BACKSWING" in phase_name or "UNIT TURN" in phase_name or "Ready Position" in phase_name:
        if confidence > 0.8:
            return (255, 100, 0)
        elif confidence >= 0.5:
            return (200, 80, 0)
        else:
            return (150, 60, 0)

    elif "FORWARD SWING" in phase_name:
        if confidence > 0.8:
            return (0, 255, 0)
        elif confidence >= 0.5:
            return (0, 200, 0)
        else:
            return (0, 150, 0)

    elif "CONTACT" in phase_name:
        if confidence > 0.8:
            return (0, 0, 255)
        elif confidence >= 0.5:
            return (0, 0, 200)
        else:
            return (0, 0, 150)

    elif "FOLLOW THROUGH" in phase_name or "FINISH" in phase_name:
        if confidence > 0.8:
            return (0, 200, 255)
        elif confidence >= 0.5:
            return (0, 160, 200)
        else:
            return (0, 120, 150)

    if confidence > 0.8:
        return (0, 255, 0)
    elif confidence >= 0.5:
        return (0, 255, 255)
    else:
        return (0, 0, 255)