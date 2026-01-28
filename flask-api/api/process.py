"""
Video processing endpoint.

Handles the full swing analysis pipeline: download from Supabase,
run pose detection + swing analysis + visualization, upload results.
"""

import logging
import os
import tempfile

from flask import Blueprint, jsonify, request

from services.quality_checker import check_video_quality
from services.storage import download_video, get_signed_url, upload_video
from services.swing_analyzer import SwingAnalyzer
from services.video_processor import PRESET_DIFFICULT_VIDEO, VideoProcessor
from services.visualizer import visualize_swing_phases

logger = logging.getLogger(__name__)

process_bp = Blueprint("process", __name__)


@process_bp.route("/api/process", methods=["POST"])
def process_video():
    """
    Run the full swing analysis pipeline on an uploaded video.

    Expects JSON body: { "video_path": "video/123_file.mp4" }
    Requires header: x-secret matching FLASK_SECRET_KEY env var.

    Returns:
        JSON with analysis results and annotated video path, or error.
    """
    # --- Auth ---
    secret = os.environ.get("FLASK_SECRET_KEY")
    if secret and request.headers.get("x-secret") != secret:
        return jsonify({"status": "failed", "error": "Unauthorized"}), 401

    # --- Validate request ---
    data = request.get_json()
    if not data:
        return jsonify({"status": "failed", "error": "Request body must be JSON"}), 400

    video_path = data.get("video_path")
    if not video_path:
        return jsonify({"status": "failed", "error": "video_path is required"}), 400

    local_input = None
    local_output = None

    try:
        # --- 1. Download from Supabase ---
        logger.info("Downloading video: %s", video_path)
        local_input = download_video(video_path)

        # --- 2. Quality check ---
        quality_report = check_video_quality(local_input)
        logger.info("Quality check: acceptable=%s", quality_report["is_acceptable"])

        # --- 3. Pose extraction ---
        logger.info("Running pose extraction...")
        processor = VideoProcessor(pose_config=PRESET_DIFFICULT_VIDEO)
        video_data = processor.process_video(local_input)

        # --- 4. Swing analysis ---
        logger.info("Analyzing swing phases...")
        analyzer = SwingAnalyzer(
            use_adaptive_velocity=True,
            adaptive_velocity_percent=0.15,
            contact_angle_min=120,
            kinematic_chain_mode=True,
            contact_detection_method="hybrid",
        )
        analysis_results = analyzer.analyze_swing(video_data)

        # --- 5. Create annotated video ---
        _, ext = os.path.splitext(video_path)
        fd, local_output = tempfile.mkstemp(suffix=ext or ".mp4")
        os.close(fd)

        logger.info("Creating annotated video...")
        visualize_swing_phases(
            video_path=local_input,
            analysis_results=analysis_results,
            output_path=local_output,
        )

        # --- 6. Upload annotated video to Supabase ---
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        annotated_path = f"results/{base_name}_annotated.mp4"

        logger.info("Uploading annotated video: %s", annotated_path)
        upload_video(local_output, annotated_path)

        annotated_url = get_signed_url(annotated_path)

        # --- 7. Return results ---
        return jsonify({
            "status": "completed",
            "video_path": video_path,
            "annotated_path": annotated_path,
            "annotated_url": annotated_url,
            "quality": quality_report,
            "analysis": analysis_results.to_dict(),
        })

    except Exception as e:
        logger.exception("Processing failed for %s", video_path)
        return jsonify({
            "status": "failed",
            "video_path": video_path,
            "error": str(e),
        }), 500

    finally:
        # Cleanup temp files
        if local_input and os.path.exists(local_input):
            os.remove(local_input)
        if local_output and os.path.exists(local_output):
            os.remove(local_output)