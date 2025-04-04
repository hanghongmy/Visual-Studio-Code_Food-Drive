import os
import joblib
import pandas as pd
import logging
import yaml
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import Counter, Histogram, Gauge
import time
import psutil
import threading
import shap
import base64
from io import BytesIO
import traceback
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for matplotlib
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/predict_api.log", mode="a")
    ]
)
logger = logging.getLogger("predict_api")

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

# Load configuration
config_path = "configs/predict_config.yaml"
if not os.path.exists(config_path):
    logger.error(f'Config file not found: {config_path}')
    raise FileNotFoundError(f"Config file not found: {config_path}")
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

model_v1_path = config.get("model_v1_path")
model_v2_path = config.get("model_v2_path")

if not model_v1_path or not os.path.exists(model_v1_path):
    logger.error(f'Model v1 file not found: {model_v1_path}')
    raise FileNotFoundError(f"Model v1 file not found: {model_v1_path}")

if not model_v2_path or not os.path.exists(model_v2_path):
    logger.error(f'Model v2 file not found: {model_v2_path}')
    raise FileNotFoundError(f"Model v2 file not found: {model_v2_path}")

# Load models
model_v1 = joblib.load(model_v1_path)
model_v2 = joblib.load(model_v2_path)

# Extract feature columns from config
feature_columns = config.get("feature_columns")
if not feature_columns:
    logger.error("Feature columns not found in config")
    raise ValueError("Feature columns not found in config")

@app.route('/', methods=['GET'])
def root():
    """Root endpoint to redirect to /food_drive_home"""
    return jsonify({
        "message": "Welcome to the Food Drive Prediction API. Visit /food_drive_home for more information."
    })

@app.route('/health_status', methods=['GET'])
def health_status():
    """Health endpoint to confirm API is running"""
    return jsonify({"status": "API is running and operational"})

def validate_input(data):
    """Validate input data against required features"""
    missing_features = [feature for feature in feature_columns if feature not in data]
    if missing_features:
        return False, f"Missing features: {missing_features}"
    return True, None

def predict_v1_logic(data):
    """Prediction logic for model version 1"""
    start_time = time.time()
    model_version = "v1"

    # Validate input
    is_valid, error_message = validate_input(data)
    if not is_valid:
        logger.warning(f"Invalid input: {error_message}")
        prediction_requests.labels(model_version=model_version, status="error").inc()
        return {"error": error_message}, 400

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data], columns=feature_columns)
        prediction = model_v1.predict(df).tolist()

        # Record successful prediction
        prediction_requests.labels(model_version=model_version, status="success").inc()
        duration = time.time() - start_time
        prediction_time.labels(model_version=model_version).observe(duration)

        logger.info(f"Prediction successful: {prediction}")
        return {"prediction": prediction}, 200
    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        prediction_requests.labels(model_version=model_version, status="error").inc()
        return {"error": "Prediction failed"}, 500

def predict_v2_logic(data):
    """Prediction logic for model version 2"""
    start_time = time.time()
    model_version = "v2"

    # Validate input
    is_valid, error_message = validate_input(data)
    if not is_valid:
        logger.warning(f"Invalid input: {error_message}")
        prediction_requests.labels(model_version=model_version, status="error").inc()
        return {"error": error_message}, 400

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data], columns=feature_columns)
        prediction = model_v2.predict(df).tolist()

        # Record successful prediction
        prediction_requests.labels(model_version=model_version, status="success").inc()
        duration = time.time() - start_time
        prediction_time.labels(model_version=model_version).observe(duration)

        logger.info(f"Prediction successful: {prediction}")
        return {"prediction": prediction}, 200
    except Exception as e:
        logger.error(f"Prediction error: {traceback.format_exc()}")
        prediction_requests.labels(model_version=model_version, status="error").inc()
        return {"error": "Prediction failed"}, 500

@app.route('/v1/predict', methods=['POST'])
def predict_v1():
    """Predict endpoint using model version 1"""
    data = request.get_json()
    if not data:
        logger.warning("No data provided")
        return jsonify({"error": "No data provided"}), 400

    response, status_code = predict_v1_logic(data)
    return jsonify(response), status_code

@app.route('/v2/predict', methods=['POST'])
def predict_v2():
    """Predict endpoint using model version 2"""
    data = request.get_json()
    if not data:
        logger.warning("No data provided")
        return jsonify({"error": "No data provided"}), 400

    response, status_code = predict_v2_logic(data)
    return jsonify(response), status_code

@app.route('/v1/explain_shap', methods=['POST'])
def explain_shap_v1():
    """Explain predictions using SHAP for model version 1"""
    data = request.get_json()
    if not data:
        logger.warning("No data provided")
        return jsonify({"error": "No data provided"}), 400

    # Validate input
    is_valid, error_message = validate_input(data)
    if not is_valid:
        logger.warning(f"Invalid input: {error_message}")
        return jsonify({"error": error_message}), 400

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data], columns=feature_columns)
        logger.debug("Dataframe input to SHAP: {df}")

        # Generate SHAP explanation
        explainer = shap.Explainer(model_v1, df)
        logger.debug("Explainer created")
        shap_values = explainer(df)
        logger.debug("SHAP values calculated")

        # Handle scalar or array expected_value
        if isinstance(explainer.expected_value, (int, float)):
            expected_value = explainer.expected_value
        else:
            expected_value = explainer.expected_value[0]

        # Generate SHAP force plot
        os.makedirs("logs", exist_ok=True)
        force_plot = shap.force_plot(
            expected_value, shap_values.values[0], df.iloc[0], matplotlib=True
        )

        # Save the force plot as a PNG image
        image_path = "logs/shap_explanation_v1.png"
        plt.savefig(image_path, bbox_inches="tight")
        plt.close()

        logger.info(f"SHAP explanation saved to {image_path}")
        return jsonify({"message": f"SHAP explanation saved to {image_path}"})
    except Exception as e:
        logger.error(f"SHAP explanation error: {traceback.format_exc()}")
        return jsonify({"error": "Failed to generate SHAP explanation"}), 500


@app.route('/v2/explain_shap', methods=['POST'])
def explain_shap_v2():
    """Explain predictions using SHAP for model version 2"""
    data = request.get_json()
    if not data:
        logger.warning("No data provided")
        return jsonify({"error": "No data provided"}), 400

    # Validate input
    is_valid, error_message = validate_input(data)
    if not is_valid:
        logger.warning(f"Invalid input: {error_message}")
        return jsonify({"error": error_message}), 400

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data], columns=feature_columns)
        logger.debug(f"DataFrame input to SHAP: {df}")

        # Generate SHAP explanation
        explainer = shap.Explainer(model_v2, df)
        logger.debug("Explanier created")
        shap_values = explainer(df)
        logger.debug("SHAP values calculated")

        # Handle scalar or array expected_value
        if isinstance(explainer.expected_value, (int, float)):
            expected_value = explainer.expected_value
        else:
            expected_value = explainer.expected_value[0]

        # Generate SHAP force plot
        os.makedirs("logs", exist_ok=True)
        force_plot = shap.force_plot(
            expected_value, shap_values.values[0], df.iloc[0], matplotlib=True
        )

        # Save the force plot as a PNG image
        image_path = "logs/shap_explanation_v2.png"
        
        plt.savefig(image_path, bbox_inches="tight")
        plt.close()

        logger.info(f"SHAP explanation saved to {image_path}")
        return jsonify({"message": f"SHAP explanation saved to {image_path}"})
    except Exception as e:
        logger.error(f"SHAP explanation error: {traceback.format_exc()}")
        return jsonify({"error": "Failed to generate SHAP explanation"}), 500

def monitor_resources():
    """Update system resource metrics every 15 seconds"""
    while True:
        process = psutil.Process(os.getpid())
        memory_usage.set(process.memory_info().rss)  # in bytes
        cpu_usage.set(process.cpu_percent())
        time.sleep(15)

if __name__ == "__main__":
    logger.info("Starting API server...")
    threading.Thread(target=monitor_resources, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=True)