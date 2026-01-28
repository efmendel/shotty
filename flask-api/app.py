import os

from dotenv import load_dotenv
from flask import Flask
from flask_cors import CORS

load_dotenv(".env.local")


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    CORS(app)

    from api.health import health_bp
    from api.process import process_bp

    app.register_blueprint(health_bp)
    app.register_blueprint(process_bp)

    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)