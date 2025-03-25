import os
import joblib
import pandas as pd
import logging
import yaml
from flask import Flask, request, jsonify
from logging_config import configure_logging


# Configure logging
log_dir = os.environ.get("LOG_DIR", "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "predict_api.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ml_app.api") 

app = Flask(__name__)

# Load configuration
config_path = "configs/predict_config.yaml"
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

model_v1_path = config.get("model_v1_path")
model_v2_path = config.get("model_v2_path")

if not model_v1_path or not os.path.exists(model_v1_path):
    raise FileNotFoundError(f"Model v1 file not found: {model_v1_path}")

if not model_v2_path or not os.path.exists(model_v2_path):
    raise FileNotFoundError(f"Model v2 file not found: {model_v2_path}")

# Load models
model_v1 = joblib.load(model_v1_path)
model_v2 = joblib.load(model_v2_path)

@app.route('/food_drive_home', methods=['GET'])
def home():
    """Home endpoint providing API usage information"""
    info = {
        "description": "This API serves machine learning models for predicting donation bags collected.",
        "endpoints": {
            "/v1/predict": "Predict using model version 1",
            "/v2/predict": "Predict using model version 2",
            "/health_status": "Check the health status of the API"
        },
        "request_format": {
            "time_spent": "float",
            "doors_in_route": "int",
            "assessed_value": "float"
        }
    }
    return jsonify(info)

@app.route('/health_status', methods=['GET'])
def health_status():
    """Health endpoint to confirm API is running"""
    return jsonify({"status": "API is running and operational"})

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    """Predict endpoint using model version 1"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    try:
        df = pd.DataFrame([data])
        prediction = model_v1.predict(df).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    """Predict endpoint using model version 2"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
    try:
        df = pd.DataFrame([data])
        prediction = model_v2.predict(df).tolist()
        return jsonify({"prediction": prediction})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)