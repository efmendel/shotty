import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

load_dotenv(".env.local")

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Shotty Flask API", "status": "running"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/notify", methods=["POST"])
def notify():
    from services.storage import get_signed_url

    data = request.get_json()
    video_path = data.get("video_path") if data else None

    if not video_path:
        return jsonify({"error": "video_path is required"}), 400

    # Test Supabase connection by generating a signed URL
    try:
        signed_url = get_signed_url(video_path)
        return jsonify({
            "status": "received",
            "message": f"Flask received and verified: {video_path}",
            "video_path": video_path,
            "signed_url": signed_url
        })
    except Exception as e:
        return jsonify({
            "status": "received",
            "message": f"Flask received: {video_path} (Supabase error: {str(e)})",
            "video_path": video_path,
            "error": str(e)
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
