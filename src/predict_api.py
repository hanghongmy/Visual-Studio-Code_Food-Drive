import os
import joblib
import pandas as pd
import logging
import yaml
from flask import Flask, request, jsonify, request
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import start_http_server
import matplotlib.pyplot as plt
import time
import psutil
import threading
from logging.handlers import RotatingFileHandler
'''
Flask API designed to serve machine learning models for making predictions
# Endpoints: 
    - /v1/predict: Predict using model version 1 - 
    - /v2/predict: Predict using model version 2
    - /food_drive_home: Home endpoint providing API usage information
    - /health_status: Check the health status of the API
# Monitoring:
    - Tracks metrics prediction requests, processing time, memory usage, and CPU usage using Prometheus
# Flask
    - Flask is used to create the API server
# Prometheus
    - Prometheus is used for monitoring and exposing metrics
'''
# Configure logging
log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
start_http_server(9000)

REQUEST_COUNT = Counter("http_requests_total", "Total HTTP Requests to the API")



# Add a rotating file handler for the predict module
file_handler = RotatingFileHandler(
    f'{log_directory}/predict.log',
    maxBytes=10485760,  # 10MB
    backupCount=5  # Keep up to 5 backup files
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger('ml_app.predict')
logger.addHandler(file_handler)

# Initialize Flask app
app = Flask(__name__)
metrics = PrometheusMetrics(app)

# Custom metrics
prediction_requests = Counter(
    'model_prediction_requests_total',
    'Total number of prediction requests',
    ['model_version', 'status']
)
prediction_time = Histogram(
    'model_prediction_duration_seconds',
    'Time spent processing prediction',
    ['model_version']
)
memory_usage = Gauge('app_memory_usage_bytes', 'Memory usage of the application')
cpu_usage = Gauge('app_cpu_usage_percent', 'CPU usage percentage of the application')
disk_usage = Gauge('app_disk_usage_bytes', 'Disk usage of the application')
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

# Extract feature columns from config
feature_columns = config.get("feature_columns")
if not feature_columns:
    raise ValueError("Feature columns not found in config")


@app.route('/', methods=['GET'])
def root():
    """Root endpoint to redirect to /food_drive_home"""
    return jsonify({
        "message": "Welcome to the Food Drive Prediction API. Try making a prediction for the experiment."
    })

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

def validate_input(data):
    """Validate input data against required feature"""
    missing_features = [feature for feature in feature_columns if feature not in data]
    if missing_features:
        return False, f"Missing features: {missing_features}"
    return True, None




@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    """Predict endpoint using model version 1"""
    REQUEST_COUNT.inc()
    start_time = time.time()
    model_version = "v1"
    
    data = request.get_json()
    if not data:
        logger.warning("No data provided")
        prediction_requests.labels(model_version=model_version, status="error").inc()
        return jsonify({"error": "No data provided"}), 400
    
    # Validate input
    is_valid, error_message = validate_input(data)
    if not is_valid:
        logger.warning(f"Invalid input: {error_message}")
        prediction_requests.labels(model_version=model_version, status="error").inc()
        return jsonify({"error": error_message}), 400
    
    try:
        # Convert input to dataframe
        df = pd.DataFrame([data], columns=feature_columns)
        prediction = model_v1.predict(df).tolist()
        
        # Record successful prediction
        prediction_requests.labels(model_version=model_version, status="success").inc()
        duration = time.time() - start_time
        prediction_time.labels(model_version=model_version).observe(duration)
        
        logger.info(f"Prediction successful: {prediction}")
        return jsonify({"prediction": prediction})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        prediction_requests.labels(model_version=model_version, status="error").inc()
        return jsonify({"error": "Prediction failed"}), 500

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    """Predict endpoint using model version 2"""
    REQUEST_COUNT.inc()
    start_time = time.time()
    model_version = "v2"
    
    data = request.get_json()
    if not data:
        logger.warning("No data provided")
        prediction_requests.labels(model_version=model_version, status="error").inc()
        return jsonify({"error": "No data provided"}), 400
    
    # Validate input
    is_valid, error_message = validate_input(data)
    if not is_valid:
        logger.warning(f"Invalid input: {error_message}")
        prediction_requests.labels(model_version=model_version, status="error").inc()
        return jsonify({"error": error_message}), 400
    try:
        #Convert input to dataframe
        df = pd.DataFrame([data], columns=feature_columns)
        prediction = model_v2.predict(df).tolist()
        
        # Record successful prediction
        prediction_requests.labels(model_version=model_version, status="success").inc()
        duration = time.time() - start_time
        prediction_time.labels(model_version=model_version).observe(duration)
        
        logger.info(f"Prediction successful: {prediction}")
        return jsonify({"prediction": prediction})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        prediction_requests.labels(model_version=model_version, status="error").inc()
        return jsonify({"error": "Prediction failed"}), 500

def monitor_resources():
    """Update system resource metrics every 15 seconds"""
    while True:
        process = psutil.Process(os.getpid())
        memory_usage.set(process.memory_info().rss)  # in bytes
        cpu_usage.set(process.cpu_percent())
        disk_usage.set(psutil.disk_usage('/').used) 
        time.sleep(15)      

@app.route('/metrics', methods=['GET'])
def metrics():
    return "Metrics endpoint"


if __name__ == "__main__":
    logger.info("Starting API server...")

    # Start Prometheus metrics server
    threading.Thread(target=start_http_server, args=(8010,), daemon=True).start()
    
    # Start resource monitoring in a separate thread
    threading.Thread(target=monitor_resources, daemon=True).start()

    # Start Flask app
    app.run(host="0.0.0.0", port=5001, debug=True)