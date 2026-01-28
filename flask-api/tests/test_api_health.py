"""
Tests for health check endpoints.

Tests the root and health check routes.
"""

import pytest

from app import create_app


@pytest.fixture
def client():
    """Create test client for Flask app."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestRootEndpoint:
    """Tests for GET / endpoint."""

    def test_root_returns_200(self, client):
        """Root endpoint returns 200 status."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_json(self, client):
        """Root endpoint returns JSON response."""
        response = client.get("/")
        assert response.content_type == "application/json"

    def test_root_contains_message(self, client):
        """Root endpoint contains service message."""
        response = client.get("/")
        data = response.get_json()
        assert "message" in data
        assert "Shotty" in data["message"]

    def test_root_contains_status(self, client):
        """Root endpoint contains running status."""
        response = client.get("/")
        data = response.get_json()
        assert data["status"] == "running"


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint returns 200 status."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_json(self, client):
        """Health endpoint returns JSON response."""
        response = client.get("/health")
        assert response.content_type == "application/json"

    def test_health_status_is_healthy(self, client):
        """Health endpoint reports healthy status."""
        response = client.get("/health")
        data = response.get_json()
        assert data["status"] == "healthy"

    def test_health_contains_service_name(self, client):
        """Health endpoint contains service name."""
        response = client.get("/health")
        data = response.get_json()
        assert "service" in data
        assert "Shotty" in data["service"]

    def test_health_contains_supabase_configured(self, client):
        """Health endpoint reports supabase configuration status."""
        response = client.get("/health")
        data = response.get_json()
        assert "supabase_configured" in data
        assert isinstance(data["supabase_configured"], bool)
