"""Flask API for SemiLab."""

import os

from flask import Flask, jsonify, request
from flask_cors import CORS

from physics.junction import simulate_junction
from physics.materials import MATERIALS, get_material


def create_app() -> Flask:
    app = Flask(__name__)
    configured_origins = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173").split(",")
    origins = configured_origins + [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://semi-lab-front-end.vercel.app",
    ]
    CORS(app, origins=[origin.strip() for origin in origins if origin.strip()])

    @app.get("/api/health")
    def health():
        return jsonify({"status": "ok", "service": "SemiLab physics API"})

    @app.get("/api/materials")
    def materials():
        return jsonify({"materials": [material.public_dict() for material in MATERIALS.values()]})

    @app.post("/api/simulate")
    def simulate():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be a JSON object."}), 400
        try:
            result = simulate_junction(
                get_material(str(payload.get("material", ""))),
                float(payload["Na"]), float(payload["Nd"]),
                float(payload["temperature"]), float(payload["appliedVoltage"]),
            )
        except KeyError as exc:
            return jsonify({"error": f"Missing required parameter: {exc.args[0]}."}), 400
        except (TypeError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(result)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG") == "1")
