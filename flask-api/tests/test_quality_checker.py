"""Tests for services/quality_checker.py - video quality validation."""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from services.quality_checker import check_video_quality


class TestCheckVideoQuality:
    """Tests for check_video_quality function."""

    def _mock_video_capture(self, width=1920, height=1080, fps=30.0, frame_count=90,
                            brightness=150.0, sharpness=200.0):
        """Create a mock cv2.VideoCapture with configurable properties."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True

        props = {
            3: float(width),    # CAP_PROP_FRAME_WIDTH
            4: float(height),   # CAP_PROP_FRAME_HEIGHT
            5: fps,             # CAP_PROP_FPS
            7: float(frame_count),  # CAP_PROP_FRAME_COUNT
        }
        mock_cap.get.side_effect = lambda prop: props.get(prop, 0.0)

        # Create a frame with controlled brightness
        frame = np.full((height, width, 3), int(brightness), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)

        return mock_cap

    def _setup_cv2_constants(self, mock_cv2):
        """Set cv2 constants on the mock so cap.get() works correctly."""
        mock_cv2.CAP_PROP_FRAME_WIDTH = 3
        mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
        mock_cv2.CAP_PROP_FPS = 5
        mock_cv2.CAP_PROP_FRAME_COUNT = 7
        mock_cv2.CAP_PROP_POS_FRAMES = 1
        mock_cv2.COLOR_BGR2GRAY = 6
        mock_cv2.CV_64F = 6

    @patch("services.quality_checker.cv2")
    def test_good_quality_video(self, mock_cv2):
        mock_cap = self._mock_video_capture()
        mock_cv2.VideoCapture.return_value = mock_cap
        self._setup_cv2_constants(mock_cv2)
        mock_cv2.cvtColor.return_value = np.full((1080, 1920), 150, dtype=np.uint8)
        mock_cv2.Laplacian.return_value = np.random.normal(0, 15, (1080, 1920))

        report = check_video_quality("test.mp4")

        assert report["resolution"] == (1920, 1080)
        assert report["fps"] == 30.0
        assert report["is_acceptable"] is True
        assert len(report["warnings"]) == 0

    @patch("services.quality_checker.cv2")
    def test_low_resolution(self, mock_cv2):
        mock_cap = self._mock_video_capture(width=640, height=480)
        mock_cv2.VideoCapture.return_value = mock_cap
        self._setup_cv2_constants(mock_cv2)
        mock_cv2.cvtColor.return_value = np.full((480, 640), 150, dtype=np.uint8)
        mock_cv2.Laplacian.return_value = np.random.normal(0, 15, (480, 640))

        report = check_video_quality("test.mp4")

        assert report["resolution"] == (640, 480)
        assert report["is_acceptable"] is False
        assert any("Resolution" in w for w in report["warnings"])

    @patch("services.quality_checker.cv2")
    def test_low_fps(self, mock_cv2):
        mock_cap = self._mock_video_capture(fps=15.0)
        mock_cv2.VideoCapture.return_value = mock_cap
        self._setup_cv2_constants(mock_cv2)
        mock_cv2.cvtColor.return_value = np.full((1080, 1920), 150, dtype=np.uint8)
        mock_cv2.Laplacian.return_value = np.random.normal(0, 15, (1080, 1920))

        report = check_video_quality("test.mp4")

        assert report["is_acceptable"] is False
        assert any("Frame rate" in w for w in report["warnings"])

    @patch("services.quality_checker.cv2")
    def test_dark_video(self, mock_cv2):
        mock_cap = self._mock_video_capture(brightness=50)
        mock_cv2.VideoCapture.return_value = mock_cap
        self._setup_cv2_constants(mock_cv2)
        mock_cv2.cvtColor.return_value = np.full((1080, 1920), 50, dtype=np.uint8)
        mock_cv2.Laplacian.return_value = np.random.normal(0, 15, (1080, 1920))

        report = check_video_quality("test.mp4")

        assert report["brightness"] < 100
        assert any("dark" in w for w in report["warnings"])

    @patch("services.quality_checker.cv2")
    def test_blurry_video(self, mock_cv2):
        mock_cap = self._mock_video_capture()
        mock_cv2.VideoCapture.return_value = mock_cap
        self._setup_cv2_constants(mock_cv2)
        mock_cv2.cvtColor.return_value = np.full((1080, 1920), 150, dtype=np.uint8)
        # Very low variance = blurry
        mock_cv2.Laplacian.return_value = np.random.normal(0, 1, (1080, 1920))

        report = check_video_quality("test.mp4")

        assert report["sharpness"] < 100
        assert any("blur" in w or "sharpness" in w for w in report["warnings"])

    @patch("services.quality_checker.cv2")
    def test_cannot_open_video(self, mock_cv2):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        with pytest.raises(ValueError, match="Cannot open video file"):
            check_video_quality("nonexistent.mp4")

    @patch("services.quality_checker.cv2")
    def test_report_keys(self, mock_cv2):
        mock_cap = self._mock_video_capture()
        mock_cv2.VideoCapture.return_value = mock_cap
        self._setup_cv2_constants(mock_cv2)
        mock_cv2.cvtColor.return_value = np.full((1080, 1920), 150, dtype=np.uint8)
        mock_cv2.Laplacian.return_value = np.random.normal(0, 15, (1080, 1920))

        report = check_video_quality("test.mp4")

        assert "resolution" in report
        assert "fps" in report
        assert "brightness" in report
        assert "sharpness" in report
        assert "warnings" in report
        assert "is_acceptable" in report
