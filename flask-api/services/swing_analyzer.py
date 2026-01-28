"""
Biomechanical swing analysis algorithms.

Detects swing phases (unit turn, backswing, forward swing, contact,
follow through) from pose landmark data using velocity, angle, and
kinematic chain analysis.
"""

import logging

from utils.geometry import (
    calculate_angle,
    calculate_velocity,
    get_body_center_x,
    is_wrist_behind_body,
)
from utils.biomechanics import (
    calculate_hip_rotation,
    calculate_shoulder_rotation,
    calculate_knee_bend,
    calculate_trunk_lean,
)
from services.models import SwingAnalysisResults

logger = logging.getLogger(__name__)


class SwingAnalyzerConfig:
    """
    Configuration for swing phase detection.

    Attributes:
        velocity_threshold: Minimum wrist velocity for swing detection.
        contact_angle_min: Minimum elbow extension angle at contact (degrees).
        use_adaptive_velocity: Use percentage of max velocity instead of fixed threshold.
        adaptive_velocity_percent: Percentage of max velocity for threshold.
        contact_frame_offset: Frames to adjust forward after peak velocity.
        follow_through_offset: Wrist position threshold past body center (0-1).
        forward_swing_search_window: Maximum frames to search for contact.
        min_valid_frames: Minimum frames with pose detected for analysis.
        wrist_behind_body_threshold: X-position threshold relative to body center.
        kinematic_chain_mode: Enable multi-joint kinematic chain analysis.
        contact_detection_method: 'velocity_peak', 'kinematic_chain', or 'hybrid'.
    """

    def __init__(
        self,
        velocity_threshold=0.5,
        contact_angle_min=150,
        use_adaptive_velocity=False,
        adaptive_velocity_percent=0.15,
        contact_frame_offset=3,
        follow_through_offset=0.15,
        forward_swing_search_window=40,
        min_valid_frames=10,
        wrist_behind_body_threshold=0.0,
        kinematic_chain_mode=False,
        contact_detection_method="velocity_peak",
    ):
        """
        Initialize swing analyzer configuration.

        Args:
            velocity_threshold: Fixed minimum wrist velocity (default: 0.5).
            contact_angle_min: Minimum elbow angle at contact in degrees (default: 150).
            use_adaptive_velocity: Use adaptive threshold (default: False).
            adaptive_velocity_percent: Fraction of max velocity (default: 0.15).
            contact_frame_offset: Frame offset after peak velocity (default: 3).
            follow_through_offset: Wrist offset past body center (default: 0.15).
            forward_swing_search_window: Max frames to search (default: 40).
            min_valid_frames: Minimum valid frames required (default: 10).
            wrist_behind_body_threshold: Reserved for future use (default: 0.0).
            kinematic_chain_mode: Enable kinematic chain mode (default: False).
            contact_detection_method: Detection method (default: 'velocity_peak').

        Raises:
            ValueError: If any parameter is out of valid range.
        """
        if velocity_threshold < 0:
            raise ValueError("velocity_threshold must be non-negative")
        if not 0 <= contact_angle_min <= 180:
            raise ValueError("contact_angle_min must be between 0 and 180 degrees")
        if not 0.0 < adaptive_velocity_percent < 1.0:
            raise ValueError("adaptive_velocity_percent must be between 0.0 and 1.0")
        if contact_frame_offset < 0:
            raise ValueError("contact_frame_offset must be non-negative")
        if not 0.0 <= follow_through_offset <= 1.0:
            raise ValueError("follow_through_offset must be between 0.0 and 1.0")
        if forward_swing_search_window < 1:
            raise ValueError("forward_swing_search_window must be at least 1")
        if min_valid_frames < 1:
            raise ValueError("min_valid_frames must be at least 1")
        if not isinstance(kinematic_chain_mode, bool):
            raise ValueError("kinematic_chain_mode must be a boolean")
        if contact_detection_method not in ["velocity_peak", "kinematic_chain", "hybrid"]:
            raise ValueError(
                "contact_detection_method must be 'velocity_peak', 'kinematic_chain', or 'hybrid'"
            )

        self.velocity_threshold = velocity_threshold
        self.contact_angle_min = contact_angle_min
        self.use_adaptive_velocity = use_adaptive_velocity
        self.adaptive_velocity_percent = adaptive_velocity_percent
        self.contact_frame_offset = contact_frame_offset
        self.follow_through_offset = follow_through_offset
        self.forward_swing_search_window = forward_swing_search_window
        self.min_valid_frames = min_valid_frames
        self.wrist_behind_body_threshold = wrist_behind_body_threshold
        self.kinematic_chain_mode = kinematic_chain_mode
        self.contact_detection_method = contact_detection_method

    def __repr__(self):
        """String representation of configuration."""
        return (
            f"SwingAnalyzerConfig("
            f"velocity_threshold={self.velocity_threshold}, "
            f"contact_angle_min={self.contact_angle_min}, "
            f"use_adaptive_velocity={self.use_adaptive_velocity}, "
            f"adaptive_velocity_percent={self.adaptive_velocity_percent}, "
            f"contact_frame_offset={self.contact_frame_offset}, "
            f"follow_through_offset={self.follow_through_offset}, "
            f"forward_swing_search_window={self.forward_swing_search_window}, "
            f"min_valid_frames={self.min_valid_frames}, "
            f"kinematic_chain_mode={self.kinematic_chain_mode}, "
            f"contact_detection_method={self.contact_detection_method})"
        )


# Preset configurations
PRESET_STANDARD = SwingAnalyzerConfig()

PRESET_SENSITIVE = SwingAnalyzerConfig(
    velocity_threshold=0.3,
    contact_angle_min=120,
    use_adaptive_velocity=True,
    adaptive_velocity_percent=0.10,
)

PRESET_STRICT = SwingAnalyzerConfig(
    velocity_threshold=0.7,
    contact_angle_min=160,
    contact_frame_offset=2,
)


class SwingAnalyzer:
    """
    Analyzes swing phases from processed video pose data.

    Supports traditional (wrist-position) and kinematic chain
    (multi-joint biomechanical) detection modes.
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize swing analyzer.

        Args:
            config: SwingAnalyzerConfig instance. If None, creates from kwargs
                or uses PRESET_STANDARD.
            **kwargs: Config parameters for backward compatibility.
        """
        if config is None:
            if kwargs:
                config = SwingAnalyzerConfig(**kwargs)
            else:
                config = PRESET_STANDARD

        self.config = config

        self.velocity_threshold = config.velocity_threshold
        self.contact_angle_min = config.contact_angle_min
        self.use_adaptive_velocity = config.use_adaptive_velocity
        self.adaptive_velocity_percent = config.adaptive_velocity_percent
        self.contact_frame_offset = config.contact_frame_offset
        self.follow_through_offset = config.follow_through_offset
        self.forward_swing_search_window = config.forward_swing_search_window
        self.min_valid_frames = config.min_valid_frames
        self.wrist_behind_body_threshold = config.wrist_behind_body_threshold
        self.kinematic_chain_mode = config.kinematic_chain_mode
        self.contact_detection_method = config.contact_detection_method

        logger.info(
            "Swing Analyzer Config: velocity_threshold=%s, contact_angle_min=%s, "
            "kinematic_chain_mode=%s, contact_detection_method=%s",
            self.velocity_threshold,
            self.contact_angle_min,
            self.kinematic_chain_mode,
            self.contact_detection_method,
        )

    def analyze_swing(self, video_data):
        """
        Analyze swing phases from processed video data.

        Args:
            video_data: Dictionary with fps, frames, and optional tracking_quality.

        Returns:
            SwingAnalysisResults with phases, engine, tempo, and kinetic chain data.
        """
        frames = video_data["frames"]
        fps = video_data["fps"]

        results = SwingAnalysisResults()

        if "tracking_quality" in video_data:
            results.set_tracking_quality(video_data["tracking_quality"])

        valid_frames = [f for f in frames if f["pose_detected"]]

        if len(valid_frames) < self.min_valid_frames:
            logger.warning(
                "Not enough frames with pose detected (%d, need %d)",
                len(valid_frames),
                self.min_valid_frames,
            )
            return results

        logger.info("Analyzing %d valid frames...", len(valid_frames))

        frame_metrics = self._calculate_frame_metrics(valid_frames, fps)

        velocity_threshold = self.velocity_threshold
        if self.use_adaptive_velocity:
            velocities = [m["wrist_velocity"] for m in frame_metrics]
            max_velocity = max(velocities)
            velocity_threshold = max_velocity * self.adaptive_velocity_percent
            logger.info(
                "Adaptive velocity: max=%.4f, threshold=%.4f (%.0f%%)",
                max_velocity,
                velocity_threshold,
                self.adaptive_velocity_percent * 100,
            )

        phases_dict = self._detect_phases(
            frame_metrics, valid_frames, velocity_threshold, self.kinematic_chain_mode
        )

        self._populate_results_from_phases(results, phases_dict, frame_metrics, fps)

        return results

    def _calculate_frame_metrics(self, frames, fps):
        """
        Calculate angles and velocities for each frame.

        Args:
            frames: List of valid frame dicts with landmarks.
            fps: Video frames per second.

        Returns:
            List of per-frame metric dictionaries.
        """
        metrics = []

        for i, frame in enumerate(frames):
            landmarks = frame["landmarks"]

            elbow_angle = calculate_angle(
                landmarks["right_shoulder"],
                landmarks["right_elbow"],
                landmarks["right_wrist"],
            )

            wrist_velocity = 0
            if i > 0:
                prev_wrist = frames[i - 1]["landmarks"]["right_wrist"]
                curr_wrist = landmarks["right_wrist"]
                time_delta = 1 / fps
                wrist_velocity = calculate_velocity(curr_wrist, prev_wrist, time_delta)

            wrist_behind = is_wrist_behind_body(
                landmarks["right_wrist"],
                landmarks["left_shoulder"],
                landmarks["right_shoulder"],
            )

            body_center = get_body_center_x(
                landmarks["left_shoulder"], landmarks["right_shoulder"]
            )

            hip_rotation = calculate_hip_rotation(landmarks)
            shoulder_rotation = calculate_shoulder_rotation(landmarks)
            knee_bend = calculate_knee_bend(landmarks, side="right")
            trunk_lean = calculate_trunk_lean(landmarks)

            hip_velocity = 0
            shoulder_velocity = 0
            elbow_velocity = 0
            if i > 0:
                prev_landmarks = frames[i - 1]["landmarks"]
                prev_hip_rotation = calculate_hip_rotation(prev_landmarks)
                prev_shoulder_rotation = calculate_shoulder_rotation(prev_landmarks)

                time_delta = 1 / fps
                hip_velocity = abs(hip_rotation - prev_hip_rotation) / time_delta
                shoulder_velocity = abs(shoulder_rotation - prev_shoulder_rotation) / time_delta
                elbow_velocity = calculate_velocity(
                    landmarks["right_elbow"], prev_landmarks["right_elbow"], time_delta
                )

            metrics.append(
                {
                    "frame_number": frame["frame_number"],
                    "timestamp": frame["timestamp"],
                    "elbow_angle": elbow_angle,
                    "wrist_velocity": wrist_velocity,
                    "wrist_x": landmarks["right_wrist"]["x"],
                    "wrist_behind_body": wrist_behind,
                    "body_center_x": body_center,
                    "hip_rotation": hip_rotation,
                    "shoulder_rotation": shoulder_rotation,
                    "knee_bend": knee_bend,
                    "trunk_lean": trunk_lean,
                    "hip_velocity": hip_velocity,
                    "shoulder_velocity": shoulder_velocity,
                    "elbow_velocity": elbow_velocity,
                }
            )

        return metrics

    def _detect_contact_kinematic_chain(self, metrics, forward_idx, velocity_threshold):
        """
        Detect contact using kinematic chain sequencing.

        Looks for shoulder deceleration, elbow at peak, wrist at peak,
        proper velocity ordering, and arm extension.

        Args:
            metrics: List of frame metrics.
            forward_idx: Index of forward swing start.
            velocity_threshold: Velocity threshold for detection.

        Returns:
            Tuple of (contact_metric or None, confidence_score).
        """
        search_window_end = min(forward_idx + self.forward_swing_search_window, len(metrics))
        contact_candidates = []

        for i in range(forward_idx + 1, search_window_end):
            m = metrics[i]

            elbow_high_velocity = m["elbow_velocity"] > velocity_threshold * 0.6
            wrist_peak_velocity = m["wrist_velocity"] > velocity_threshold
            wrist_fastest = m["wrist_velocity"] > m["elbow_velocity"]
            arm_extended = m["elbow_angle"] > self.contact_angle_min
            wrist_in_front = not m["wrist_behind_body"]

            if (
                elbow_high_velocity
                and wrist_peak_velocity
                and wrist_fastest
                and arm_extended
                and wrist_in_front
            ):
                velocity_gradient = m["wrist_velocity"] - m["elbow_velocity"]
                sequencing_quality = min(1.0, velocity_gradient / 0.5)

                elbow_wrist_ratio = (
                    m["elbow_velocity"] / m["wrist_velocity"] if m["wrist_velocity"] > 0 else 0
                )
                shoulder_elbow_ratio = (
                    m["shoulder_velocity"] / m["elbow_velocity"] if m["elbow_velocity"] > 0 else 0
                )

                ratio_score = (
                    (1.0 - abs(shoulder_elbow_ratio - 0.6))
                    + (1.0 - abs(elbow_wrist_ratio - 0.7))
                ) / 2.0
                ratio_score = max(0.0, min(1.0, ratio_score))

                velocity_score = min(1.0, m["wrist_velocity"] / (velocity_threshold * 2))
                angle_score = min(1.0, (m["elbow_angle"] - self.contact_angle_min) / 30.0)

                m["sequencing_quality"] = sequencing_quality
                m["ratio_score"] = ratio_score
                m["velocity_score"] = velocity_score
                m["angle_score"] = angle_score
                m["shoulder_velocity_at_contact"] = m["shoulder_velocity"]
                m["elbow_velocity_at_contact"] = m["elbow_velocity"]

                contact_candidates.append(m)

        if not contact_candidates:
            return None, 0.0

        best_contact = max(contact_candidates, key=lambda x: x["wrist_velocity"])

        confidence = (
            best_contact["sequencing_quality"]
            + best_contact["ratio_score"]
            + best_contact["velocity_score"]
            + best_contact["angle_score"]
        ) / 4.0

        return best_contact, confidence

    def _detect_phases(self, metrics, frames, velocity_threshold, kinematic_chain_mode=False):
        """
        Detect swing phases from frame metrics.

        Args:
            metrics: List of per-frame metrics.
            frames: List of valid frames.
            velocity_threshold: Velocity threshold.
            kinematic_chain_mode: Use kinematic chain detection.

        Returns:
            Dictionary of phase results with detected, confidence, reason, etc.
        """
        phases = {
            "backswing_start": {"detected": False, "confidence": 0.0, "reason": "Not yet analyzed"},
            "max_backswing": {"detected": False, "confidence": 0.0, "reason": "Not yet analyzed"},
            "forward_swing_start": {"detected": False, "confidence": 0.0, "reason": "Not yet analyzed"},
            "contact": {"detected": False, "confidence": 0.0, "reason": "Not yet analyzed"},
            "follow_through": {"detected": False, "confidence": 0.0, "reason": "Not yet analyzed"},
        }

        # --- Backswing start ---
        backswing_found = False

        if kinematic_chain_mode:
            for i, m in enumerate(metrics):
                hip_rotating = abs(m["hip_rotation"]) > 10
                shoulder_rotating = abs(m["shoulder_rotation"]) > 15

                if hip_rotating and shoulder_rotating and m["wrist_behind_body"]:
                    wrist_offset = m["body_center_x"] - m["wrist_x"]
                    rotation_score = min(
                        1.0, (abs(m["hip_rotation"]) + abs(m["shoulder_rotation"])) / 50
                    )
                    wrist_score = min(1.0, wrist_offset / 0.1)
                    confidence = (rotation_score + wrist_score) / 2.0

                    phases["backswing_start"] = {
                        "detected": True,
                        "confidence": confidence,
                        "reason": "Successfully detected (kinematic chain)",
                        "frame": m["frame_number"],
                        "timestamp": m["timestamp"],
                        "wrist_offset": wrist_offset,
                        "hip_rotation": m["hip_rotation"],
                        "shoulder_rotation": m["shoulder_rotation"],
                        "rotation_score": rotation_score,
                    }
                    backswing_found = True
                    break

            if not backswing_found:
                phases["backswing_start"]["reason"] = "insufficient_body_rotation"
        else:
            for i, m in enumerate(metrics):
                if m["wrist_behind_body"]:
                    wrist_offset = m["body_center_x"] - m["wrist_x"]
                    confidence = min(1.0, wrist_offset / 0.1)

                    phases["backswing_start"] = {
                        "detected": True,
                        "confidence": confidence,
                        "reason": "Successfully detected",
                        "frame": m["frame_number"],
                        "timestamp": m["timestamp"],
                        "wrist_offset": wrist_offset,
                    }
                    backswing_found = True
                    break

            if not backswing_found:
                phases["backswing_start"]["reason"] = "wrist_never_behind_body"

        # --- Max backswing ---
        if phases["backswing_start"]["detected"]:
            backswing_frames = [m for m in metrics if m["wrist_behind_body"]]
            if backswing_frames:
                max_back = min(backswing_frames, key=lambda x: x["wrist_x"])
                backswing_depth = max_back["body_center_x"] - max_back["wrist_x"]
                confidence = min(1.0, backswing_depth / 0.15)

                phases["max_backswing"] = {
                    "detected": True,
                    "confidence": confidence,
                    "reason": "Successfully detected",
                    "frame": max_back["frame_number"],
                    "timestamp": max_back["timestamp"],
                    "wrist_x": max_back["wrist_x"],
                    "backswing_depth": backswing_depth,
                }
            else:
                phases["max_backswing"]["reason"] = "no_backswing_frames_found"
        else:
            phases["max_backswing"]["reason"] = "backswing_start_not_detected"

        # --- Forward swing start ---
        if phases["max_backswing"]["detected"]:
            max_back_idx = next(
                i for i, m in enumerate(metrics)
                if m["frame_number"] == phases["max_backswing"]["frame"]
            )

            forward_found = False

            if kinematic_chain_mode:
                for i in range(max_back_idx, len(metrics)):
                    m = metrics[i]
                    if m["hip_velocity"] > 30 and m["wrist_velocity"] > velocity_threshold * 0.5:
                        hip_vel_score = min(1.0, m["hip_velocity"] / 60)
                        wrist_vel_score = min(1.0, m["wrist_velocity"] / velocity_threshold)
                        confidence = (hip_vel_score + wrist_vel_score) / 2.0

                        phases["forward_swing_start"] = {
                            "detected": True,
                            "confidence": confidence,
                            "reason": "Successfully detected (kinematic chain)",
                            "frame": m["frame_number"],
                            "timestamp": m["timestamp"],
                            "velocity": m["wrist_velocity"],
                            "hip_velocity": m["hip_velocity"],
                            "shoulder_velocity": m["shoulder_velocity"],
                            "hip_vel_score": hip_vel_score,
                        }
                        forward_found = True
                        break

                if not forward_found:
                    phases["forward_swing_start"]["reason"] = "no_hip_velocity_reversal"
            else:
                for i in range(max_back_idx, len(metrics)):
                    if metrics[i]["wrist_velocity"] > velocity_threshold:
                        velocity_ratio = metrics[i]["wrist_velocity"] / velocity_threshold
                        confidence = min(1.0, (velocity_ratio - 1.0) / 2.0 + 0.5)

                        phases["forward_swing_start"] = {
                            "detected": True,
                            "confidence": confidence,
                            "reason": "Successfully detected",
                            "frame": metrics[i]["frame_number"],
                            "timestamp": metrics[i]["timestamp"],
                            "velocity": metrics[i]["wrist_velocity"],
                            "velocity_ratio": velocity_ratio,
                        }
                        forward_found = True
                        break

                if not forward_found:
                    phases["forward_swing_start"]["reason"] = "insufficient_velocity"
        else:
            phases["forward_swing_start"]["reason"] = "max_backswing_not_detected"

        # --- Contact ---
        if phases["forward_swing_start"]["detected"]:
            forward_idx = next(
                i for i, m in enumerate(metrics)
                if m["frame_number"] == phases["forward_swing_start"]["frame"]
            )

            detection_method = self.contact_detection_method

            if detection_method == "kinematic_chain":
                self._detect_contact_kc_method(phases, metrics, forward_idx, velocity_threshold)
            elif detection_method == "hybrid":
                self._detect_contact_hybrid_method(phases, metrics, forward_idx, velocity_threshold)
            else:
                self._detect_contact_vp_method(phases, metrics, forward_idx, velocity_threshold)
        else:
            phases["contact"]["reason"] = "forward_swing_start_not_detected"
            phases["contact"]["method"] = self.contact_detection_method

        # --- Follow through ---
        if phases["contact"]["detected"]:
            contact_idx = next(
                i for i, m in enumerate(metrics)
                if m["frame_number"] == phases["contact"]["frame"]
            )

            follow_found = False
            for i in range(contact_idx, len(metrics)):
                m = metrics[i]
                if m["wrist_x"] > m["body_center_x"] + self.follow_through_offset:
                    follow_distance = m["wrist_x"] - m["body_center_x"]
                    confidence = min(1.0, follow_distance / 0.3)

                    phases["follow_through"] = {
                        "detected": True,
                        "confidence": confidence,
                        "reason": "Successfully detected",
                        "frame": m["frame_number"],
                        "timestamp": m["timestamp"],
                        "wrist_x": m["wrist_x"],
                        "follow_distance": follow_distance,
                    }
                    follow_found = True
                    break

            if not follow_found:
                phases["follow_through"]["reason"] = "wrist_never_crossed_body_center"
        else:
            phases["follow_through"]["reason"] = "contact_not_detected"

        # Overall quality
        detected_count = sum(1 for p in phases.values() if p.get("detected"))
        total_phases = len(phases)
        avg_confidence = sum(p.get("confidence", 0) for p in phases.values()) / total_phases

        phases["_analysis_quality"] = {
            "overall_score": avg_confidence,
            "phases_detected": detected_count,
            "total_phases": total_phases,
            "detection_rate": detected_count / total_phases,
        }

        return phases

    def _detect_contact_vp_method(self, phases, metrics, forward_idx, velocity_threshold):
        """Detect contact using velocity peak method."""
        search_window_end = min(forward_idx + self.forward_swing_search_window, len(metrics))
        contact_candidates = []

        for i in range(forward_idx, search_window_end):
            m = metrics[i]
            if (
                m["elbow_angle"] > self.contact_angle_min
                and m["wrist_velocity"] > velocity_threshold
                and not m["wrist_behind_body"]
            ):
                contact_candidates.append(m)

        if contact_candidates:
            contact = max(contact_candidates, key=lambda x: x["wrist_velocity"])
            contact_idx = next(
                i for i, m in enumerate(metrics)
                if m["frame_number"] == contact["frame_number"]
            )
            adjusted_idx = min(len(metrics) - 1, contact_idx + self.contact_frame_offset)
            adjusted_contact = metrics[adjusted_idx]

            velocity_score = min(1.0, adjusted_contact["wrist_velocity"] / (velocity_threshold * 2))
            angle_score = min(1.0, (adjusted_contact["elbow_angle"] - self.contact_angle_min) / 30.0)
            confidence = (velocity_score + angle_score) / 2.0

            phases["contact"] = {
                "detected": True,
                "confidence": confidence,
                "reason": "Successfully detected",
                "method": "velocity_peak",
                "frame": adjusted_contact["frame_number"],
                "timestamp": adjusted_contact["timestamp"],
                "velocity": adjusted_contact["wrist_velocity"],
                "elbow_angle": adjusted_contact["elbow_angle"],
                "velocity_score": velocity_score,
                "angle_score": angle_score,
            }
        else:
            no_velocity = all(
                m["wrist_velocity"] <= velocity_threshold
                for m in metrics[forward_idx:search_window_end]
            )
            no_extension = all(
                m["elbow_angle"] <= self.contact_angle_min
                for m in metrics[forward_idx:search_window_end]
            )

            if no_velocity and no_extension:
                phases["contact"]["reason"] = "insufficient_velocity_and_arm_not_extended"
            elif no_velocity:
                phases["contact"]["reason"] = "insufficient_velocity"
            elif no_extension:
                phases["contact"]["reason"] = "arm_not_extended"
            else:
                phases["contact"]["reason"] = "wrist_position_unclear"
            phases["contact"]["method"] = "velocity_peak"

    def _detect_contact_kc_method(self, phases, metrics, forward_idx, velocity_threshold):
        """Detect contact using kinematic chain method."""
        contact_result, confidence = self._detect_contact_kinematic_chain(
            metrics, forward_idx, velocity_threshold
        )

        if contact_result:
            contact_idx = next(
                i for i, m in enumerate(metrics)
                if m["frame_number"] == contact_result["frame_number"]
            )
            adjusted_idx = min(len(metrics) - 1, contact_idx + self.contact_frame_offset)
            adjusted_contact = metrics[adjusted_idx]

            phases["contact"] = {
                "detected": True,
                "confidence": confidence,
                "reason": "Successfully detected",
                "method": "kinematic_chain",
                "frame": adjusted_contact["frame_number"],
                "timestamp": adjusted_contact["timestamp"],
                "velocity": adjusted_contact["wrist_velocity"],
                "elbow_angle": adjusted_contact["elbow_angle"],
                "shoulder_velocity": contact_result.get("shoulder_velocity_at_contact", 0),
                "elbow_velocity": contact_result.get("elbow_velocity_at_contact", 0),
                "sequencing_quality": contact_result.get("sequencing_quality", 0),
                "ratio_score": contact_result.get("ratio_score", 0),
                "velocity_score": contact_result.get("velocity_score", 0),
                "angle_score": contact_result.get("angle_score", 0),
            }
        else:
            phases["contact"]["reason"] = "no_kinematic_chain_signature_found"
            phases["contact"]["method"] = "kinematic_chain"

    def _detect_contact_hybrid_method(self, phases, metrics, forward_idx, velocity_threshold):
        """Detect contact using hybrid method (best of both)."""
        kc_result, kc_confidence = self._detect_contact_kinematic_chain(
            metrics, forward_idx, velocity_threshold
        )

        search_window_end = min(forward_idx + self.forward_swing_search_window, len(metrics))
        vp_candidates = []
        for i in range(forward_idx, search_window_end):
            m = metrics[i]
            if (
                m["elbow_angle"] > self.contact_angle_min
                and m["wrist_velocity"] > velocity_threshold
                and not m["wrist_behind_body"]
            ):
                vp_candidates.append(m)

        vp_result = None
        vp_confidence = 0.0
        if vp_candidates:
            vp_contact = max(vp_candidates, key=lambda x: x["wrist_velocity"])
            velocity_score = min(1.0, vp_contact["wrist_velocity"] / (velocity_threshold * 2))
            angle_score = min(1.0, (vp_contact["elbow_angle"] - self.contact_angle_min) / 30.0)
            vp_confidence = (velocity_score + angle_score) / 2.0
            vp_result = vp_contact

        if kc_confidence >= vp_confidence and kc_result:
            contact_idx = next(
                i for i, m in enumerate(metrics)
                if m["frame_number"] == kc_result["frame_number"]
            )
            adjusted_idx = min(len(metrics) - 1, contact_idx + self.contact_frame_offset)
            adjusted_contact = metrics[adjusted_idx]

            phases["contact"] = {
                "detected": True,
                "confidence": kc_confidence,
                "reason": "Successfully detected",
                "method": "hybrid (used kinematic_chain)",
                "frame": adjusted_contact["frame_number"],
                "timestamp": adjusted_contact["timestamp"],
                "velocity": adjusted_contact["wrist_velocity"],
                "elbow_angle": adjusted_contact["elbow_angle"],
                "shoulder_velocity": kc_result.get("shoulder_velocity_at_contact", 0),
                "elbow_velocity": kc_result.get("elbow_velocity_at_contact", 0),
                "sequencing_quality": kc_result.get("sequencing_quality", 0),
            }
        elif vp_result:
            contact_idx = next(
                i for i, m in enumerate(metrics)
                if m["frame_number"] == vp_result["frame_number"]
            )
            adjusted_idx = min(len(metrics) - 1, contact_idx + self.contact_frame_offset)
            adjusted_contact = metrics[adjusted_idx]

            velocity_score = min(1.0, adjusted_contact["wrist_velocity"] / (velocity_threshold * 2))
            angle_score = min(1.0, (adjusted_contact["elbow_angle"] - self.contact_angle_min) / 30.0)

            phases["contact"] = {
                "detected": True,
                "confidence": (velocity_score + angle_score) / 2.0,
                "reason": "Successfully detected",
                "method": "hybrid (used velocity_peak)",
                "frame": adjusted_contact["frame_number"],
                "timestamp": adjusted_contact["timestamp"],
                "velocity": adjusted_contact["wrist_velocity"],
                "elbow_angle": adjusted_contact["elbow_angle"],
                "velocity_score": velocity_score,
                "angle_score": angle_score,
            }
        else:
            phases["contact"]["reason"] = "no_contact_detected_by_any_method"
            phases["contact"]["method"] = "hybrid"

    def _populate_results_from_phases(self, results, phases_dict, frame_metrics, fps):
        """
        Populate SwingAnalysisResults from phase detection dict.

        Args:
            results: SwingAnalysisResults to populate.
            phases_dict: Phase detection results dict.
            frame_metrics: List of per-frame metrics.
            fps: Video frames per second.
        """
        phase_mapping = {
            "backswing_start": "unit_turn",
            "max_backswing": "backswing",
            "forward_swing_start": "forward_swing",
            "contact": "contact",
            "follow_through": "follow_through",
        }

        for old_name, new_name in phase_mapping.items():
            if old_name in phases_dict:
                phase_data = phases_dict[old_name]

                detected = phase_data.get("detected", False)
                frame = phase_data.get("frame")
                timestamp = phase_data.get("timestamp")
                confidence = phase_data.get("confidence", 0.0)

                phase_metrics = {}

                if new_name == "unit_turn" and detected:
                    phase_metrics["shoulder_rotation"] = phase_data.get("shoulder_rotation", 0.0)

                elif new_name == "backswing" and detected:
                    phase_metrics["shoulder_rotation"] = phase_data.get("shoulder_rotation", 0.0)
                    if frame:
                        metric = next(
                            (m for m in frame_metrics if m["frame_number"] == frame), None
                        )
                        if metric:
                            phase_metrics["max_wrist_depth"] = 1.0 - metric.get("wrist_x", 0.5)

                elif new_name == "forward_swing" and detected:
                    phase_metrics["hip_velocity"] = phase_data.get("hip_velocity", 0.0)

                elif new_name == "contact" and detected:
                    phase_metrics["wrist_velocity"] = phase_data.get("velocity", 0.0)
                    phase_metrics["elbow_angle"] = phase_data.get("elbow_angle", 0.0)
                    phase_metrics["method"] = phase_data.get("method", "unknown")
                    if "shoulder_velocity" in phase_data:
                        phase_metrics["shoulder_velocity"] = phase_data["shoulder_velocity"]
                    if "elbow_velocity" in phase_data:
                        phase_metrics["elbow_velocity"] = phase_data["elbow_velocity"]
                    if "sequencing_quality" in phase_data:
                        phase_metrics["sequencing_quality"] = phase_data["sequencing_quality"]

                results.add_phase(
                    new_name,
                    detected=detected,
                    frame=frame,
                    timestamp=timestamp,
                    confidence=confidence,
                    **phase_metrics,
                )

        self._calculate_engine_metrics(results, frame_metrics)
        self._calculate_tempo_metrics(results, phases_dict)
        self._calculate_kinetic_chain_metrics(results, frame_metrics, phases_dict)

    def _calculate_engine_metrics(self, results, frame_metrics):
        """
        Calculate engine metrics (hip-shoulder separation, rotations).

        Args:
            results: SwingAnalysisResults to populate.
            frame_metrics: List of per-frame metrics.
        """
        if not frame_metrics:
            return

        max_separation = 0.0
        max_sep_frame = None
        max_sep_timestamp = None

        max_shoulder_rot = 0.0
        max_shoulder_frame = None
        max_shoulder_timestamp = None

        max_hip_rot = 0.0
        max_hip_frame = None
        max_hip_timestamp = None

        for metric in frame_metrics:
            hip_rot = metric.get("hip_rotation", 0.0)
            shoulder_rot = metric.get("shoulder_rotation", 0.0)
            separation = abs(shoulder_rot - hip_rot)

            if separation > max_separation:
                max_separation = separation
                max_sep_frame = metric["frame_number"]
                max_sep_timestamp = metric["timestamp"]

            if shoulder_rot < max_shoulder_rot:
                max_shoulder_rot = shoulder_rot
                max_shoulder_frame = metric["frame_number"]
                max_shoulder_timestamp = metric["timestamp"]

            if hip_rot < max_hip_rot:
                max_hip_rot = hip_rot
                max_hip_frame = metric["frame_number"]
                max_hip_timestamp = metric["timestamp"]

        results.add_engine_metrics(
            hip_shoulder_sep={
                "max_value": max_separation,
                "frame": max_sep_frame,
                "timestamp": max_sep_timestamp,
            },
            max_shoulder_rot={
                "value": max_shoulder_rot,
                "frame": max_shoulder_frame,
                "timestamp": max_shoulder_timestamp,
            },
            max_hip_rot={
                "value": max_hip_rot,
                "frame": max_hip_frame,
                "timestamp": max_hip_timestamp,
            },
        )

    def _calculate_tempo_metrics(self, results, phases_dict):
        """
        Calculate tempo metrics (durations, rhythm ratio).

        Args:
            results: SwingAnalysisResults to populate.
            phases_dict: Phase detection results dict.
        """
        unit_turn_ts = None
        forward_swing_ts = None
        contact_ts = None

        if "backswing_start" in phases_dict and phases_dict["backswing_start"].get("detected"):
            unit_turn_ts = phases_dict["backswing_start"].get("timestamp")

        if "forward_swing_start" in phases_dict and phases_dict["forward_swing_start"].get("detected"):
            forward_swing_ts = phases_dict["forward_swing_start"].get("timestamp")

        if "contact" in phases_dict and phases_dict["contact"].get("detected"):
            contact_ts = phases_dict["contact"].get("timestamp")

        backswing_duration = None
        forward_swing_duration = None
        swing_rhythm_ratio = None

        if unit_turn_ts is not None and forward_swing_ts is not None:
            backswing_duration = forward_swing_ts - unit_turn_ts

        if forward_swing_ts is not None and contact_ts is not None:
            forward_swing_duration = contact_ts - forward_swing_ts

        if backswing_duration and forward_swing_duration and forward_swing_duration > 0:
            swing_rhythm_ratio = backswing_duration / forward_swing_duration

        results.add_tempo_metrics(
            backswing_duration=backswing_duration,
            forward_swing_duration=forward_swing_duration,
            swing_rhythm_ratio=swing_rhythm_ratio,
        )

    def _calculate_kinetic_chain_metrics(self, results, frame_metrics, phases_dict):
        """
        Calculate kinetic chain metrics (velocity sequencing, lag times).

        Args:
            results: SwingAnalysisResults to populate.
            frame_metrics: List of per-frame metrics.
            phases_dict: Phase detection results dict.
        """
        if not frame_metrics:
            return

        max_hip_vel = 0.0
        max_hip_frame = None
        max_hip_ts = None

        max_shoulder_vel = 0.0
        max_shoulder_frame = None
        max_shoulder_ts = None

        max_elbow_vel = 0.0
        max_elbow_frame = None
        max_elbow_ts = None

        max_wrist_vel = 0.0
        max_wrist_frame = None
        max_wrist_ts = None

        for metric in frame_metrics:
            hip_vel = metric.get("hip_velocity", 0.0)
            shoulder_vel = metric.get("shoulder_velocity", 0.0)
            elbow_vel = metric.get("elbow_velocity", 0.0)
            wrist_vel = metric.get("wrist_velocity", 0.0)

            if hip_vel > max_hip_vel:
                max_hip_vel = hip_vel
                max_hip_frame = metric["frame_number"]
                max_hip_ts = metric["timestamp"]

            if shoulder_vel > max_shoulder_vel:
                max_shoulder_vel = shoulder_vel
                max_shoulder_frame = metric["frame_number"]
                max_shoulder_ts = metric["timestamp"]

            if elbow_vel > max_elbow_vel:
                max_elbow_vel = elbow_vel
                max_elbow_frame = metric["frame_number"]
                max_elbow_ts = metric["timestamp"]

            if wrist_vel > max_wrist_vel:
                max_wrist_vel = wrist_vel
                max_wrist_frame = metric["frame_number"]
                max_wrist_ts = metric["timestamp"]

        sequence = {
            "hip": {"frame": max_hip_frame, "timestamp": max_hip_ts, "velocity": max_hip_vel},
            "shoulder": {
                "frame": max_shoulder_frame,
                "timestamp": max_shoulder_ts,
                "velocity": max_shoulder_vel,
            },
            "elbow": {
                "frame": max_elbow_frame,
                "timestamp": max_elbow_ts,
                "velocity": max_elbow_vel,
            },
            "wrist": {
                "frame": max_wrist_frame,
                "timestamp": max_wrist_ts,
                "velocity": max_wrist_vel,
            },
        }

        chain_lag = {}
        if max_hip_ts is not None and max_shoulder_ts is not None:
            chain_lag["hip_to_shoulder"] = max_shoulder_ts - max_hip_ts
        if max_shoulder_ts is not None and max_elbow_ts is not None:
            chain_lag["shoulder_to_elbow"] = max_elbow_ts - max_shoulder_ts
        if max_elbow_ts is not None and max_wrist_ts is not None:
            chain_lag["elbow_to_wrist"] = max_wrist_ts - max_elbow_ts

        confidence = 0.0
        if all([max_hip_ts, max_shoulder_ts, max_elbow_ts, max_wrist_ts]):
            correct_sequence = max_hip_ts <= max_shoulder_ts <= max_elbow_ts <= max_wrist_ts
            if correct_sequence:
                confidence = 1.0
            else:
                correct_pairs = 0
                total_pairs = 3
                if max_hip_ts <= max_shoulder_ts:
                    correct_pairs += 1
                if max_shoulder_ts <= max_elbow_ts:
                    correct_pairs += 1
                if max_elbow_ts <= max_wrist_ts:
                    correct_pairs += 1
                confidence = correct_pairs / total_pairs

        results.add_kinetic_chain_metrics(
            sequence=sequence,
            chain_lag=chain_lag,
            confidence=confidence,
        )