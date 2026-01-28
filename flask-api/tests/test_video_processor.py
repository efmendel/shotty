"""Tests for services/video_processor.py - MediaPipe pose detection pipeline."""

import pytest
from unittest.mock import patch, MagicMock
from services.video_processor import (
    PoseConfig,
    VideoProcessor,
    PRESET_HIGH_QUALITY,
    PRESET_FAST,
    PRESET_DIFFICULT_VIDEO,
    PRESET_SLOW_MOTION,
)


class TestPoseConfig:
    """Tests for PoseConfig configuration class."""

    def test_default_config(self):
        config = PoseConfig()
        assert config.model_complexity == 1
        assert config.static_image_mode is False
        assert config.min_detection_confidence == 0.5
        assert config.min_tracking_confidence == 0.5
        assert config.smooth_landmarks is True

    def test_custom_config(self):
        config = PoseConfig(
            model_complexity=2,
            static_image_mode=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.8,
            smooth_landmarks=False,
        )
        assert config.model_complexity == 2
        assert config.static_image_mode is True
        assert config.min_detection_confidence == 0.7
        assert config.min_tracking_confidence == 0.8
        assert config.smooth_landmarks is False

    def test_invalid_model_complexity(self):
        with pytest.raises(ValueError, match="model_complexity must be 0, 1, or 2"):
            PoseConfig(model_complexity=3)

    def test_invalid_detection_confidence_low(self):
        with pytest.raises(ValueError, match="min_detection_confidence"):
            PoseConfig(min_detection_confidence=-0.1)

    def test_invalid_detection_confidence_high(self):
        with pytest.raises(ValueError, match="min_detection_confidence"):
            PoseConfig(min_detection_confidence=1.1)

    def test_invalid_tracking_confidence(self):
        with pytest.raises(ValueError, match="min_tracking_confidence"):
            PoseConfig(min_tracking_confidence=1.5)

    def test_to_dict(self):
        config = PoseConfig()
        d = config.to_dict()
        assert d["model_complexity"] == 1
        assert d["static_image_mode"] is False
        assert d["min_detection_confidence"] == 0.5
        assert d["min_tracking_confidence"] == 0.5
        assert d["smooth_landmarks"] is True

    def test_repr(self):
        config = PoseConfig()
        r = repr(config)
        assert "PoseConfig" in r
        assert "model_complexity=1" in r


class TestPresets:
    """Tests for preset configurations."""

    def test_high_quality_preset(self):
        assert PRESET_HIGH_QUALITY.model_complexity == 2
        assert PRESET_HIGH_QUALITY.smooth_landmarks is True

    def test_fast_preset(self):
        assert PRESET_FAST.model_complexity == 0
        assert PRESET_FAST.smooth_landmarks is False

    def test_difficult_video_preset(self):
        assert PRESET_DIFFICULT_VIDEO.model_complexity == 2
        assert PRESET_DIFFICULT_VIDEO.min_detection_confidence == 0.3

    def test_slow_motion_preset(self):
        assert PRESET_SLOW_MOTION.model_complexity == 2
        assert PRESET_SLOW_MOTION.min_tracking_confidence == 0.7


class TestVideoProcessorInit:
    """Tests for VideoProcessor initialization."""

    @patch("services.video_processor.mp")
    def test_default_config(self, mock_mp):
        mock_mp.solutions.pose.Pose.return_value = MagicMock()
        processor = VideoProcessor()
        assert processor.pose_config.model_complexity == 1

    @patch("services.video_processor.mp")
    def test_custom_config(self, mock_mp):
        mock_mp.solutions.pose.Pose.return_value = MagicMock()
        config = PoseConfig(model_complexity=0)
        processor = VideoProcessor(pose_config=config)
        assert processor.pose_config.model_complexity == 0


class TestAssessTrackingQuality:
    """Tests for assess_tracking_quality method."""

    @patch("services.video_processor.mp")
    def test_empty_frames(self, mock_mp):
        mock_mp.solutions.pose.Pose.return_value = MagicMock()
        processor = VideoProcessor()
        result = processor.assess_tracking_quality({"frames": []})
        assert result["detection_rate"] == 0.0
        assert result["high_confidence_rate"] == 0.0
        assert result["average_confidence"] == 0.0

    @patch("services.video_processor.mp")
    def test_all_detected(self, mock_mp):
        mock_mp.solutions.pose.Pose.return_value = MagicMock()
        processor = VideoProcessor()
        frames = [
            {
                "pose_detected": True,
                "landmarks": {
                    "left_shoulder": {"x": 0.4, "y": 0.3, "z": 0, "visibility": 0.9},
                    "right_shoulder": {"x": 0.6, "y": 0.3, "z": 0, "visibility": 0.9},
                },
            },
            {
                "pose_detected": True,
                "landmarks": {
                    "left_shoulder": {"x": 0.4, "y": 0.3, "z": 0, "visibility": 0.8},
                    "right_shoulder": {"x": 0.6, "y": 0.3, "z": 0, "visibility": 0.8},
                },
            },
        ]
        result = processor.assess_tracking_quality({"frames": frames})
        assert result["detection_rate"] == 1.0
        assert result["high_confidence_rate"] == 1.0
        assert abs(result["average_confidence"] - 0.85) < 0.01

    @patch("services.video_processor.mp")
    def test_partial_detection(self, mock_mp):
        mock_mp.solutions.pose.Pose.return_value = MagicMock()
        processor = VideoProcessor()
        frames = [
            {
                "pose_detected": True,
                "landmarks": {
                    "left_shoulder": {"x": 0.4, "y": 0.3, "z": 0, "visibility": 0.9},
                },
            },
            {"pose_detected": False, "landmarks": None},
        ]
        result = processor.assess_tracking_quality({"frames": frames})
        assert result["detection_rate"] == 0.5

    @patch("services.video_processor.mp")
    def test_no_detection(self, mock_mp):
        mock_mp.solutions.pose.Pose.return_value = MagicMock()
        processor = VideoProcessor()
        frames = [
            {"pose_detected": False, "landmarks": None},
            {"pose_detected": False, "landmarks": None},
        ]
        result = processor.assess_tracking_quality({"frames": frames})
        assert result["detection_rate"] == 0.0
        assert result["average_confidence"] == 0.0

    @patch("services.video_processor.mp")
    def test_low_confidence(self, mock_mp):
        mock_mp.solutions.pose.Pose.return_value = MagicMock()
        processor = VideoProcessor()
        frames = [
            {
                "pose_detected": True,
                "landmarks": {
                    "left_shoulder": {"x": 0.4, "y": 0.3, "z": 0, "visibility": 0.3},
                },
            },
        ]
        result = processor.assess_tracking_quality({"frames": frames})
        assert result["detection_rate"] == 1.0
        assert result["high_confidence_rate"] == 0.0


class TestExtractLandmarks:
    """Tests for _extract_landmarks method."""

    @patch("services.video_processor.mp")
    def test_extracts_correct_landmarks(self, mock_mp):
        mock_mp.solutions.pose.Pose.return_value = MagicMock()
        processor = VideoProcessor()

        # Create mock pose landmarks
        mock_landmarks = MagicMock()
        landmark_data = {}
        for idx in range(33):
            lm = MagicMock()
            lm.x = idx * 0.01
            lm.y = idx * 0.02
            lm.z = idx * 0.001
            lm.visibility = 0.9
            landmark_data[idx] = lm
        mock_landmarks.landmark = landmark_data

        result = processor._extract_landmarks(mock_landmarks)

        expected_keys = {
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
        }
        assert set(result.keys()) == expected_keys

    @patch("services.video_processor.mp")
    def test_landmark_structure(self, mock_mp):
        mock_mp.solutions.pose.Pose.return_value = MagicMock()
        processor = VideoProcessor()

        mock_landmarks = MagicMock()
        for idx in range(33):
            lm = MagicMock()
            lm.x = 0.5
            lm.y = 0.5
            lm.z = 0.0
            lm.visibility = 0.9
            mock_landmarks.landmark.__getitem__ = lambda self, i, lm=lm: lm

        result = processor._extract_landmarks(mock_landmarks)
        for name, data in result.items():
            assert "x" in data
            assert "y" in data
            assert "z" in data
            assert "visibility" in data
