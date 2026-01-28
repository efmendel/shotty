"""
Swing analysis data models.

Provides the SwingAnalysisResults class for storing and managing
comprehensive tennis swing analysis data including phase detection,
engine metrics, tempo analysis, and kinetic chain information.
"""

import json
from typing import Any, Dict, Optional


class SwingAnalysisResults:
    """
    Stores comprehensive swing analysis data.

    Attributes:
        phases: Detected swing phases (unit_turn, backswing, forward_swing, contact, follow_through).
        engine: Engine metrics (hip_shoulder_separation, max_shoulder_rotation, max_hip_rotation).
        tempo: Tempo metrics (backswing_duration, forward_swing_duration, swing_rhythm_ratio).
        kinetic_chain: Kinetic chain metrics (peak_velocity_sequence, chain_lag, confidence).
        video_quality: Video quality assessment.
        tracking_quality: Pose tracking quality.
    """

    def __init__(self):
        """Initialize empty swing analysis results structure."""
        self.phases = {
            "unit_turn": None,
            "backswing": None,
            "forward_swing": None,
            "contact": None,
            "follow_through": None,
        }

        self.engine = {
            "hip_shoulder_separation": None,
            "max_shoulder_rotation": None,
            "max_hip_rotation": None,
        }

        self.tempo = {
            "backswing_duration": None,
            "forward_swing_duration": None,
            "swing_rhythm_ratio": None,
        }

        self.kinetic_chain = {
            "peak_velocity_sequence": None,
            "chain_lag": None,
            "confidence": None,
        }

        self.video_quality = None
        self.tracking_quality = None

    def add_phase(
        self,
        phase_name: str,
        detected: bool,
        frame: Optional[int] = None,
        timestamp: Optional[float] = None,
        confidence: float = 0.0,
        **metrics,
    ) -> None:
        """
        Add or update a swing phase with detection results and metrics.

        Args:
            phase_name: Name of phase ('unit_turn', 'backswing', 'forward_swing',
                'contact', 'follow_through').
            detected: Whether the phase was detected.
            frame: Frame number where phase occurs.
            timestamp: Timestamp in seconds where phase occurs.
            confidence: Detection confidence score (0.0-1.0).
            **metrics: Phase-specific metrics (e.g., wrist_velocity, elbow_angle).

        Raises:
            ValueError: If phase_name is invalid or data types are incorrect.
        """
        if phase_name not in self.phases:
            raise ValueError(
                f"Invalid phase_name: {phase_name}. "
                f"Must be one of: {', '.join(self.phases.keys())}"
            )

        if not isinstance(detected, bool):
            raise ValueError(f"detected must be a boolean, got {type(detected)}")

        if detected:
            if frame is not None and not isinstance(frame, int):
                raise ValueError(f"frame must be an integer, got {type(frame)}")
            if timestamp is not None and not isinstance(timestamp, (int, float)):
                raise ValueError(f"timestamp must be a number, got {type(timestamp)}")

        if not isinstance(confidence, (int, float)):
            raise ValueError(f"confidence must be a number, got {type(confidence)}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence}")

        phase_data = {
            "detected": detected,
            "frame": frame,
            "timestamp": timestamp,
            "confidence": confidence,
        }

        for key, value in metrics.items():
            phase_data[key] = value

        self.phases[phase_name] = phase_data

    def add_engine_metrics(
        self,
        hip_shoulder_sep: Optional[Dict[str, Any]] = None,
        max_shoulder_rot: Optional[Dict[str, Any]] = None,
        max_hip_rot: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add engine metrics (hip-shoulder separation, rotations).

        Args:
            hip_shoulder_sep: Dict with max_value, frame, timestamp.
            max_shoulder_rot: Dict with value, frame, timestamp.
            max_hip_rot: Dict with value, frame, timestamp.

        Raises:
            ValueError: If data types are incorrect.
        """
        if hip_shoulder_sep is not None:
            if not isinstance(hip_shoulder_sep, dict):
                raise ValueError(f"hip_shoulder_sep must be a dict, got {type(hip_shoulder_sep)}")
            self.engine["hip_shoulder_separation"] = hip_shoulder_sep

        if max_shoulder_rot is not None:
            if not isinstance(max_shoulder_rot, dict):
                raise ValueError(f"max_shoulder_rot must be a dict, got {type(max_shoulder_rot)}")
            self.engine["max_shoulder_rotation"] = max_shoulder_rot

        if max_hip_rot is not None:
            if not isinstance(max_hip_rot, dict):
                raise ValueError(f"max_hip_rot must be a dict, got {type(max_hip_rot)}")
            self.engine["max_hip_rotation"] = max_hip_rot

    def add_tempo_metrics(
        self,
        backswing_duration: Optional[float] = None,
        forward_swing_duration: Optional[float] = None,
        swing_rhythm_ratio: Optional[float] = None,
    ) -> None:
        """
        Add tempo metrics (swing timing and rhythm).

        Args:
            backswing_duration: Duration of backswing in seconds.
            forward_swing_duration: Duration of forward swing in seconds.
            swing_rhythm_ratio: Ratio of backswing to forward swing duration.

        Raises:
            ValueError: If data types are incorrect.
        """
        if backswing_duration is not None:
            if not isinstance(backswing_duration, (int, float)):
                raise ValueError(f"backswing_duration must be a number, got {type(backswing_duration)}")
            if backswing_duration < 0:
                raise ValueError(f"backswing_duration must be non-negative, got {backswing_duration}")
            self.tempo["backswing_duration"] = backswing_duration

        if forward_swing_duration is not None:
            if not isinstance(forward_swing_duration, (int, float)):
                raise ValueError(f"forward_swing_duration must be a number, got {type(forward_swing_duration)}")
            if forward_swing_duration < 0:
                raise ValueError(f"forward_swing_duration must be non-negative, got {forward_swing_duration}")
            self.tempo["forward_swing_duration"] = forward_swing_duration

        if swing_rhythm_ratio is not None:
            if not isinstance(swing_rhythm_ratio, (int, float)):
                raise ValueError(f"swing_rhythm_ratio must be a number, got {type(swing_rhythm_ratio)}")
            if swing_rhythm_ratio < 0:
                raise ValueError(f"swing_rhythm_ratio must be non-negative, got {swing_rhythm_ratio}")
            self.tempo["swing_rhythm_ratio"] = swing_rhythm_ratio

    def add_kinetic_chain_metrics(
        self,
        sequence: Optional[Dict[str, Any]] = None,
        chain_lag: Optional[Dict[str, Any]] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """
        Add kinetic chain metrics (velocity sequencing and lag times).

        Args:
            sequence: Dict describing peak velocity sequence per body segment.
            chain_lag: Dict with lag times between segments.
            confidence: Overall confidence in kinetic chain detection (0.0-1.0).

        Raises:
            ValueError: If data types are incorrect.
        """
        if sequence is not None:
            if not isinstance(sequence, dict):
                raise ValueError(f"sequence must be a dict, got {type(sequence)}")
            self.kinetic_chain["peak_velocity_sequence"] = sequence

        if chain_lag is not None:
            if not isinstance(chain_lag, dict):
                raise ValueError(f"chain_lag must be a dict, got {type(chain_lag)}")
            self.kinetic_chain["chain_lag"] = chain_lag

        if confidence is not None:
            if not isinstance(confidence, (int, float)):
                raise ValueError(f"confidence must be a number, got {type(confidence)}")
            if not 0.0 <= confidence <= 1.0:
                raise ValueError(f"confidence must be between 0.0 and 1.0, got {confidence}")
            self.kinetic_chain["confidence"] = confidence

    def set_video_quality(self, quality_metrics: Dict[str, Any]) -> None:
        """
        Set video quality metrics.

        Args:
            quality_metrics: Dict containing video quality assessment.

        Raises:
            ValueError: If quality_metrics is not a dict.
        """
        if not isinstance(quality_metrics, dict):
            raise ValueError(f"quality_metrics must be a dict, got {type(quality_metrics)}")
        self.video_quality = quality_metrics

    def set_tracking_quality(self, tracking_metrics: Dict[str, Any]) -> None:
        """
        Set pose tracking quality metrics.

        Args:
            tracking_metrics: Dict containing pose tracking quality.

        Raises:
            ValueError: If tracking_metrics is not a dict.
        """
        if not isinstance(tracking_metrics, dict):
            raise ValueError(f"tracking_metrics must be a dict, got {type(tracking_metrics)}")
        self.tracking_quality = tracking_metrics

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert results to nested dictionary.

        Returns:
            Complete analysis results as nested dictionary.
        """
        return {
            "phases": self.phases,
            "engine": self.engine,
            "tempo": self.tempo,
            "kinetic_chain": self.kinetic_chain,
            "video_quality": self.video_quality,
            "tracking_quality": self.tracking_quality,
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Convert results to JSON string.

        Args:
            indent: Number of spaces for JSON indentation.

        Returns:
            JSON-formatted string of analysis results.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def get_phases_detected_count(self) -> int:
        """
        Get count of detected phases.

        Returns:
            Number of phases that were successfully detected.
        """
        return sum(
            1 for phase_data in self.phases.values()
            if phase_data and phase_data.get("detected", False)
        )

    def get_overall_confidence(self) -> float:
        """
        Calculate overall detection confidence across all phases.

        Returns:
            Average confidence score (0.0-1.0), or 0.0 if no phases detected.
        """
        confidences = [
            phase_data.get("confidence", 0.0)
            for phase_data in self.phases.values()
            if phase_data and phase_data.get("detected", False)
        ]

        if not confidences:
            return 0.0

        return sum(confidences) / len(confidences)

    def __repr__(self) -> str:
        """String representation of results."""
        phases_detected = self.get_phases_detected_count()
        overall_confidence = self.get_overall_confidence()

        return (
            f"SwingAnalysisResults("
            f"phases_detected={phases_detected}/5, "
            f"overall_confidence={overall_confidence:.2f})"
        )