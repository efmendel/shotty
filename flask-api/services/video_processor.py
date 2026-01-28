"""
MediaPipe pose detection pipeline for video processing.

Processes video files frame-by-frame using MediaPipe Pose to extract
body landmark coordinates for swing analysis.
"""

import logging

import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)


class PoseConfig:
    """
    Configuration for MediaPipe Pose detection.

    Attributes:
        model_complexity: Complexity of pose model (0, 1, or 2).
        static_image_mode: Whether to treat each frame independently.
        min_detection_confidence: Minimum confidence for initial detection (0.0-1.0).
        min_tracking_confidence: Minimum confidence for tracking (0.0-1.0).
        smooth_landmarks: Whether to smooth landmarks across frames.
    """

    def __init__(
        self,
        model_complexity=1,
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        smooth_landmarks=True,
    ):
        """
        Initialize pose configuration.

        Args:
            model_complexity: 0 = fastest, 1 = balanced, 2 = most accurate.
            static_image_mode: False for video, True for static images.
            min_detection_confidence: Detection threshold (0.0-1.0).
            min_tracking_confidence: Tracking threshold (0.0-1.0).
            smooth_landmarks: Smooth landmarks across frames.

        Raises:
            ValueError: If parameters are out of valid range.
        """
        if model_complexity not in [0, 1, 2]:
            raise ValueError("model_complexity must be 0, 1, or 2")
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError("min_detection_confidence must be between 0.0 and 1.0")
        if not 0.0 <= min_tracking_confidence <= 1.0:
            raise ValueError("min_tracking_confidence must be between 0.0 and 1.0")

        self.model_complexity = model_complexity
        self.static_image_mode = static_image_mode
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.smooth_landmarks = smooth_landmarks

    def to_dict(self):
        """
        Return configuration as dictionary for MediaPipe Pose initialization.

        Returns:
            Dictionary of configuration parameters.
        """
        return {
            "model_complexity": self.model_complexity,
            "static_image_mode": self.static_image_mode,
            "min_detection_confidence": self.min_detection_confidence,
            "min_tracking_confidence": self.min_tracking_confidence,
            "smooth_landmarks": self.smooth_landmarks,
        }

    def __repr__(self):
        """String representation of configuration."""
        return (
            f"PoseConfig(model_complexity={self.model_complexity}, "
            f"static_image_mode={self.static_image_mode}, "
            f"min_detection_confidence={self.min_detection_confidence}, "
            f"min_tracking_confidence={self.min_tracking_confidence}, "
            f"smooth_landmarks={self.smooth_landmarks})"
        )


# Preset configurations
PRESET_HIGH_QUALITY = PoseConfig(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    smooth_landmarks=True,
)

PRESET_FAST = PoseConfig(
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    smooth_landmarks=False,
)

PRESET_DIFFICULT_VIDEO = PoseConfig(
    model_complexity=2,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3,
    smooth_landmarks=True,
)

PRESET_SLOW_MOTION = PoseConfig(
    model_complexity=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.7,
    smooth_landmarks=True,
)


class VideoProcessor:
    """
    Processes video files to extract pose landmarks using MediaPipe.

    Attributes:
        pose_config: PoseConfig instance controlling detection parameters.
        mp_pose: MediaPipe pose solution reference.
        pose: Initialized MediaPipe Pose instance.
    """

    # MediaPipe landmark indices for tennis swing analysis
    LANDMARK_INDICES = {
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_elbow": 13,
        "right_elbow": 14,
        "left_wrist": 15,
        "right_wrist": 16,
        "left_hip": 23,
        "right_hip": 24,
    }

    def __init__(self, pose_config=None):
        """
        Initialize VideoProcessor with optional pose configuration.

        Args:
            pose_config: PoseConfig instance. If None, uses default balanced config.
        """
        self.mp_pose = mp.solutions.pose

        if pose_config is None:
            pose_config = PoseConfig()

        self.pose_config = pose_config
        self.pose = self.mp_pose.Pose(**pose_config.to_dict())

    def process_video(self, video_path):
        """
        Process entire video and extract pose landmarks for each frame.

        Args:
            video_path: Path to video file.

        Returns:
            Dictionary containing fps, frame_count, width, height,
            frames (list of per-frame data), and tracking_quality.

        Raises:
            ValueError: If video file cannot be opened.
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info("Processing video: %d frames at %s FPS", frame_count, fps)

        frames_data = []
        frame_number = 0

        while cap.isOpened():
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1
            timestamp = frame_number / fps

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            frame_data = {
                "frame_number": frame_number,
                "timestamp": timestamp,
                "landmarks": None,
                "pose_detected": False,
            }

            if results.pose_landmarks:
                frame_data["pose_detected"] = True
                frame_data["landmarks"] = self._extract_landmarks(results.pose_landmarks)

            frames_data.append(frame_data)

            if frame_number % 30 == 0:
                logger.info("Processed %d/%d frames...", frame_number, frame_count)

        cap.release()
        self.pose.close()

        logger.info("Processing complete! %d frames processed.", frame_number)

        video_data = {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "frames": frames_data,
        }

        tracking_quality = self.assess_tracking_quality(video_data)
        video_data["tracking_quality"] = tracking_quality

        logger.info(
            "Tracking quality: detection_rate=%.1f%%, avg_confidence=%.3f",
            tracking_quality["detection_rate"] * 100,
            tracking_quality["average_confidence"],
        )

        if tracking_quality["detection_rate"] < 0.7:
            logger.warning("Detection rate below 70%% - video may not be suitable for analysis")

        return video_data

    def assess_tracking_quality(self, video_data):
        """
        Assess the quality of pose tracking across all frames.

        Args:
            video_data: Dictionary containing processed frame data.

        Returns:
            Dictionary with detection_rate, high_confidence_rate,
            and average_confidence metrics.
        """
        frames = video_data["frames"]
        total_frames = len(frames)

        if total_frames == 0:
            return {
                "detection_rate": 0.0,
                "high_confidence_rate": 0.0,
                "average_confidence": 0.0,
            }

        detected_frames = [f for f in frames if f["pose_detected"]]
        detection_rate = len(detected_frames) / total_frames

        if not detected_frames:
            return {
                "detection_rate": 0.0,
                "high_confidence_rate": 0.0,
                "average_confidence": 0.0,
            }

        frame_confidences = []
        high_confidence_frames = 0

        for frame in detected_frames:
            landmarks = frame["landmarks"]
            if landmarks:
                visibilities = [lm["visibility"] for lm in landmarks.values()]
                avg_visibility = sum(visibilities) / len(visibilities)
                frame_confidences.append(avg_visibility)

                if avg_visibility > 0.7:
                    high_confidence_frames += 1

        average_confidence = (
            sum(frame_confidences) / len(frame_confidences) if frame_confidences else 0.0
        )
        high_confidence_rate = high_confidence_frames / total_frames

        return {
            "detection_rate": detection_rate,
            "high_confidence_rate": high_confidence_rate,
            "average_confidence": average_confidence,
        }

    def _extract_landmarks(self, pose_landmarks):
        """
        Extract key landmarks for swing analysis from MediaPipe results.

        Args:
            pose_landmarks: MediaPipe pose landmarks object.

        Returns:
            Dictionary mapping landmark names to x, y, z, visibility dicts.
        """
        landmarks = {}

        for name, idx in self.LANDMARK_INDICES.items():
            landmark = pose_landmarks.landmark[idx]
            landmarks[name] = {
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility,
            }

        return landmarks