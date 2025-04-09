import os
import joblib
import pandas as pd
import logging
import yaml
import time
import psutil
import threading
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from logging.handlers import RotatingFileHandler

# Load configuration
CONFIG_PATH = "configs/predict_config.yaml"
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

model_v1_path = config.get("model_v1_path")
model_v2_path = config.get("model_v2_path")
feature_columns = config.get("feature_columns")
if not model_v1_path or not os.path.exists(model_v1_path):
    raise FileNotFoundError(f"Model v1 file not found: {model_v1_path}")
if not model_v2_path or not os.path.exists(model_v2_path):
    raise FileNotFoundError(f"Model v2 file not found: {model_v2_path}")
if not feature_columns:
    raise ValueError("Feature columns not found in config")

# Load models
model_v1 = joblib.load(model_v1_path)
model_v2 = joblib.load(model_v2_path)

# Initialize Flask app
app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Setup logging
log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True)
logger = logging.getLogger('ml_app.predict')
logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler(f'{log_directory}/predict.log', maxBytes=10485760, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Prometheus custom metrics
REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests to the API")
prediction_requests = Counter('model_prediction_requests_total', 'Prediction request count', ['model_version', 'status'])
prediction_time = Histogram('model_prediction_duration_seconds', 'Prediction time duration', ['model_version'])
memory_usage = Gauge('app_memory_usage_bytes', 'App memory usage')
cpu_usage = Gauge('app_cpu_usage_percent', 'App CPU usage')
disk_usage = Gauge('app_disk_usage_bytes', 'App disk usage')
validation_accuracy = Gauge('model_validation_accuracy', 'Validation accuracy of the model')


# System monitoring
def monitor_resources():
    """Update system resource metrics every 15 seconds"""
    while True:
        process = psutil.Process(os.getpid())
        memory_usage.set(process.memory_info().rss)
        cpu_usage.set(process.cpu_percent())
        disk_usage.set(psutil.disk_usage('/').used)
        time.sleep(15)

# Input validation
def validate_input(data):
    missing = [feature for feature in feature_columns if feature not in data]
    return (len(missing) == 0, f"Missing features: {missing}" if missing else None)

# API Endpoints
@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Welcome to the Food Drive Prediction API."})

@app.route("/food_drive_home", methods=["GET"])
def home():
    return jsonify({
        "description": "Predict donation bags collected.",
        "endpoints": {
            "/v1/predict": "Model v1",
            "/v2/predict": "Model v2",
            "/health_status": "API health check"
        },
        "input_format": {
            "time_spent": "float",
            "doors_in_route": "int",
            "assessed_value": "float"
        }
    })

@app.route("/health_status", methods=["GET"])
def health_status():
    return jsonify({"status": "API operational"})

@app.route("/v1/predict", methods=["POST"])
def predict_v1():
    return make_prediction(model_v1, "v1")

@app.route("/v2/predict", methods=["POST"])
def predict_v2():
    return make_prediction(model_v2, "v2")

def make_prediction(model, version):
    REQUEST_COUNT.inc()
    start_time = time.time()

    data = request.get_json()
    if not data:
        prediction_requests.labels(version, "error").inc()
        return jsonify({"error": "No data provided"}), 400

    is_valid, error_msg = validate_input(data)
    if not is_valid:
        prediction_requests.labels(version, "error").inc()
        return jsonify({"error": error_msg}), 400

    try:
        df = pd.DataFrame([data], columns=feature_columns)
        prediction = model.predict(df).tolist()
        prediction_requests.labels(version, "success").inc()
        prediction_time.labels(version).observe(time.time() - start_time)
        return jsonify({"prediction": prediction})
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        prediction_requests.labels(version, "error").inc()
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/update_metrics', methods=['POST'])
def update_metrics():
    """Endpoint to update validation accuracy"""
    data = request.json
    if not data or 'validation_accuracy' not in data:
        return jsonify({"error": "Missing 'validation_accuracy' in request data"}), 400

    val_accuracy = data.get('validation_accuracy', 0.0)
    logger.info(f"Validation accuracy updated to {val_accuracy}")
    return jsonify({"message": f"Validation accuracy updated to {val_accuracy}"}), 200

if __name__ == "__main__":
    logger.info("Starting API server...")

    # Start Prometheus metrics server
    threading.Thread(target=start_http_server, args=(8020,), daemon=True).start()
    
    # Start resource monitoring in a separate thread
    threading.Thread(target=monitor_resources, daemon=True).start()

    # Start Flask app on a different port
    app.run(host="0.0.0.0", port=5001, debug=True)