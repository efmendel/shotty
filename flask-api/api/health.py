"""
Health check endpoints.

Provides root status and health check routes for monitoring.
"""

import os

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/", methods=["GET"])
def root():
    """
    Root status endpoint.

    Returns:
        JSON with service name and status.
    """
    return jsonify({"message": "Shotty Flask API", "status": "running"})


@health_bp.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.

    Returns:
        JSON with health status and configuration info.
    """
    supabase_configured = bool(
        os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    )

    return jsonify({
        "status": "healthy",
        "service": "Shotty Swing Analysis API",
        "supabase_configured": supabase_configured,
    })