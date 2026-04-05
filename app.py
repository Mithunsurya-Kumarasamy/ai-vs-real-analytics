"""
Flask API — AI vs Real Image Analytics
Serves pre-computed results + live prediction endpoint
"""

import os
import json
import pickle
import base64
import tempfile
import numpy as np
from pathlib import Path
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_DIR  = Path("models")
OUTPUT_DIR = Path("output")

# ── Load artefacts ────────────────────────────────────────────────────────────
def load_model(name):
    p = MODEL_DIR / f"{name}.pkl"
    if p.exists():
        with open(p, "rb") as f:
            return pickle.load(f)
    return None

scaler = load_model("scaler")
pca    = load_model("pca")
lr     = load_model("lr")
dt     = load_model("dt")
rf     = load_model("rf")

def load_results():
    p = OUTPUT_DIR / "results.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": {
            "scaler": scaler is not None,
            "pca": pca is not None,
            "lr": lr is not None,
            "dt": dt is not None,
            "rf": rf is not None
        }
    })


@app.route("/api/results")
def get_results():
    """Return full pipeline results."""
    return jsonify(load_results())


@app.route("/api/eda")
def get_eda():
    results = load_results()
    return jsonify(results.get("eda", {}))


@app.route("/api/models/comparison")
def get_comparison():
    results = load_results()
    return jsonify(results.get("model_comparison", []))


@app.route("/api/models/<model_name>")
def get_model_results(model_name):
    results = load_results()
    key_map = {
        "logistic-regression": "logistic_regression",
        "decision-tree": "decision_tree",
        "random-forest": "random_forest"
    }
    key = key_map.get(model_name)
    if not key:
        return jsonify({"error": f"Unknown model: {model_name}"}), 404
    return jsonify(results.get(key, {}))


@app.route("/api/calibration")
def get_calibration():
    results = load_results()
    return jsonify(results.get("calibration", {}))


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Accepts: { "image_b64": "<base64 encoded image>" }
    Returns: predictions from all three models
    """
    if not all([scaler, lr, dt, rf]):
        return jsonify({"error": "Models not loaded — run ml_pipeline.py first"}), 503

    data = request.get_json()
    if not data or "image_b64" not in data:
        return jsonify({"error": "Missing image_b64 field"}), 400

    try:
        img_bytes = base64.b64decode(data["image_b64"])
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(img_bytes)
            tmp_path = tmp.name

        from feature_extraction import extract_all_features
        feats = extract_all_features(tmp_path).reshape(1, -1)
        os.unlink(tmp_path)

        feats = np.nan_to_num(feats, nan=0, posinf=0, neginf=0)
        feats_s = scaler.transform(feats)
        feats_p = pca.transform(feats_s)

        def pred_result(model, X, name):
            y_pred = int(model.predict(X)[0])
            y_prob = float(model.predict_proba(X)[0][1])
            return {
                "model": name,
                "prediction": "AI" if y_pred == 1 else "Real",
                "confidence": round(y_prob if y_pred == 1 else 1 - y_prob, 4),
                "ai_probability": round(y_prob, 4)
            }

        return jsonify({
            "predictions": [
                pred_result(lr, feats_p, "Logistic Regression"),
                pred_result(dt, feats_s, "Decision Tree"),
                pred_result(rf, feats_s, "Random Forest"),
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Flask API running on http://localhost:5000")
    app.run(debug=True, port=5000)
