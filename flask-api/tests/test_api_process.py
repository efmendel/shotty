"""
Tests for video processing endpoint.

Tests the main /api/process endpoint with mocked services.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from app import create_app


@pytest.fixture
def client():
    """Create test client for Flask app."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def mock_env_secret(monkeypatch):
    """Set FLASK_SECRET_KEY environment variable."""
    monkeypatch.setenv("FLASK_SECRET_KEY", "test-secret")


@pytest.fixture
def valid_headers():
    """Return valid request headers with auth."""
    return {
        "Content-Type": "application/json",
        "x-secret": "test-secret",
    }


@pytest.fixture
def mock_analysis_results():
    """Create mock SwingAnalysisResults object."""
    mock_results = MagicMock()
    mock_results.to_dict.return_value = {
        "phases": {
            "backswing": {"detected": True, "frame": 10, "confidence": 0.9},
            "contact": {"detected": True, "frame": 30, "confidence": 0.85},
        },
        "engine": {},
        "tempo": {},
        "kinetic_chain": {},
    }
    return mock_results


class TestProcessEndpointValidation:
    """Tests for request validation."""

    def test_non_json_content_type_returns_415(self, client, mock_env_secret, valid_headers):
        """Request without JSON content type returns 415."""
        response = client.post(
            "/api/process",
            headers={"x-secret": "test-secret"},
            data="not json",
        )
        # Flask returns 415 Unsupported Media Type for non-JSON requests
        assert response.status_code == 415

    def test_empty_json_body_returns_400(self, client, mock_env_secret, valid_headers):
        """Request with empty JSON body returns 400."""
        response = client.post(
            "/api/process",
            headers=valid_headers,
            json={},
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "failed"
        # Empty dict {} is falsy, so API returns "Request body must be JSON"
        assert "JSON" in data["error"]

    def test_missing_video_path_returns_400(self, client, mock_env_secret, valid_headers):
        """Request without video_path returns 400."""
        response = client.post(
            "/api/process",
            headers=valid_headers,
            json={"other_field": "value"},
        )
        assert response.status_code == 400
        data = response.get_json()
        assert data["status"] == "failed"
        assert "video_path" in data["error"]


class TestProcessEndpointAuth:
    """Tests for authentication."""

    def test_missing_auth_header_returns_401(self, client, mock_env_secret):
        """Request without x-secret header returns 401."""
        response = client.post(
            "/api/process",
            headers={"Content-Type": "application/json"},
            json={"video_path": "video/test.mp4"},
        )
        assert response.status_code == 401
        data = response.get_json()
        assert data["status"] == "failed"
        assert "Unauthorized" in data["error"]

    def test_wrong_auth_header_returns_401(self, client, mock_env_secret):
        """Request with wrong x-secret header returns 401."""
        response = client.post(
            "/api/process",
            headers={
                "Content-Type": "application/json",
                "x-secret": "wrong-secret",
            },
            json={"video_path": "video/test.mp4"},
        )
        assert response.status_code == 401
        data = response.get_json()
        assert data["status"] == "failed"
        assert "Unauthorized" in data["error"]

    def test_no_secret_configured_allows_request(self, client, monkeypatch):
        """When FLASK_SECRET_KEY is not set, requests are allowed."""
        monkeypatch.delenv("FLASK_SECRET_KEY", raising=False)

        with patch("api.process.download_video") as mock_download:
            mock_download.side_effect = Exception("Test stop")

            response = client.post(
                "/api/process",
                headers={"Content-Type": "application/json"},
                json={"video_path": "video/test.mp4"},
            )
            # Should get past auth and fail on download (500)
            assert response.status_code == 500


class TestProcessEndpointSuccess:
    """Tests for successful processing."""

    @patch("api.process.get_signed_url")
    @patch("api.process.upload_video")
    @patch("api.process.visualize_swing_phases")
    @patch("api.process.SwingAnalyzer")
    @patch("api.process.VideoProcessor")
    @patch("api.process.check_video_quality")
    @patch("api.process.download_video")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_successful_processing_returns_results(
        self,
        mock_remove,
        mock_exists,
        mock_download,
        mock_quality,
        mock_processor_cls,
        mock_analyzer_cls,
        mock_visualize,
        mock_upload,
        mock_signed_url,
        client,
        mock_env_secret,
        valid_headers,
        mock_analysis_results,
    ):
        """Successful processing returns analysis results."""
        # Setup mocks
        mock_download.return_value = "/tmp/input.mp4"
        mock_quality.return_value = {
            "is_acceptable": True,
            "resolution": {"width": 1920, "height": 1080},
        }

        mock_processor = MagicMock()
        mock_processor.process_video.return_value = {
            "frame_count": 100,
            "fps": 30,
            "landmarks": [],
        }
        mock_processor_cls.return_value = mock_processor

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_swing.return_value = mock_analysis_results
        mock_analyzer_cls.return_value = mock_analyzer

        mock_visualize.return_value = "/tmp/output.mp4"
        mock_signed_url.return_value = "https://example.com/signed/url"
        mock_exists.return_value = True

        response = client.post(
            "/api/process",
            headers=valid_headers,
            json={"video_path": "video/test.mp4"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "completed"
        assert data["video_path"] == "video/test.mp4"
        assert "annotated_path" in data
        assert "annotated_url" in data
        assert "quality" in data
        assert "analysis" in data

    @patch("api.process.get_signed_url")
    @patch("api.process.upload_video")
    @patch("api.process.visualize_swing_phases")
    @patch("api.process.SwingAnalyzer")
    @patch("api.process.VideoProcessor")
    @patch("api.process.check_video_quality")
    @patch("api.process.download_video")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_annotated_path_uses_results_folder(
        self,
        mock_remove,
        mock_exists,
        mock_download,
        mock_quality,
        mock_processor_cls,
        mock_analyzer_cls,
        mock_visualize,
        mock_upload,
        mock_signed_url,
        client,
        mock_env_secret,
        valid_headers,
        mock_analysis_results,
    ):
        """Annotated video is uploaded to results folder."""
        mock_download.return_value = "/tmp/input.mp4"
        mock_quality.return_value = {"is_acceptable": True}

        mock_processor = MagicMock()
        mock_processor.process_video.return_value = {"frame_count": 100}
        mock_processor_cls.return_value = mock_processor

        mock_analyzer = MagicMock()
        mock_analyzer.analyze_swing.return_value = mock_analysis_results
        mock_analyzer_cls.return_value = mock_analyzer

        mock_signed_url.return_value = "https://example.com/signed/url"
        mock_exists.return_value = True

        response = client.post(
            "/api/process",
            headers=valid_headers,
            json={"video_path": "video/myfile.mp4"},
        )

        assert response.status_code == 200
        data = response.get_json()
        assert data["annotated_path"].startswith("results/")
        assert "myfile_annotated.mp4" in data["annotated_path"]


class TestProcessEndpointErrors:
    """Tests for error handling."""

    @patch("api.process.download_video")
    def test_download_error_returns_500(
        self,
        mock_download,
        client,
        mock_env_secret,
        valid_headers,
    ):
        """Download failure returns error response."""
        mock_download.side_effect = Exception("Storage unavailable")

        response = client.post(
            "/api/process",
            headers=valid_headers,
            json={"video_path": "video/test.mp4"},
        )

        assert response.status_code == 500
        data = response.get_json()
        assert data["status"] == "failed"
        assert "Storage unavailable" in data["error"]
        assert data["video_path"] == "video/test.mp4"

    @patch("api.process.check_video_quality")
    @patch("api.process.download_video")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_processing_error_returns_500(
        self,
        mock_remove,
        mock_exists,
        mock_download,
        mock_quality,
        client,
        mock_env_secret,
        valid_headers,
    ):
        """Processing failure returns error response."""
        mock_download.return_value = "/tmp/input.mp4"
        mock_quality.side_effect = Exception("Quality check failed")
        mock_exists.return_value = True

        response = client.post(
            "/api/process",
            headers=valid_headers,
            json={"video_path": "video/test.mp4"},
        )

        assert response.status_code == 500
        data = response.get_json()
        assert data["status"] == "failed"
        assert "Quality check failed" in data["error"]

    @patch("api.process.download_video")
    @patch("os.path.exists")
    @patch("os.remove")
    def test_cleanup_happens_on_error(
        self,
        mock_remove,
        mock_exists,
        mock_download,
        client,
        mock_env_secret,
        valid_headers,
    ):
        """Temp files are cleaned up even on error."""
        mock_download.return_value = "/tmp/input.mp4"
        mock_download.side_effect = ["/tmp/input.mp4", Exception("Later error")]
        mock_exists.return_value = True

        # Trigger an error after download
        with patch("api.process.check_video_quality") as mock_quality:
            mock_quality.side_effect = Exception("Quality error")

            client.post(
                "/api/process",
                headers=valid_headers,
                json={"video_path": "video/test.mp4"},
            )

            # Verify cleanup was attempted
            mock_exists.assert_called()
            mock_remove.assert_called()
