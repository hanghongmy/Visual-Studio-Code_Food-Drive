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
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import traceback

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

# Load training data for LIME
train_data_path = config.get("train_data_path")
if not train_data_path or not os.path.exists(train_data_path):
    logger.error(f"Training data file not found: {train_data_path}")
    raise FileNotFoundError(f"Training data file not found: {train_data_path}")

train_data = pd.read_csv(train_data_path)
X_train = train_data[feature_columns]

# Initialize LIME explainers
lime_explainer_v1 = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_columns,
    class_names=["Predicted Donation Bags Collected"],
    mode="regression"
)

lime_explainer_v2 = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=feature_columns,
    class_names=["Predicted Donation Bags Collected"],
    mode="regression"
)

@app.route('/v1/explain_lime', methods=['POST'])
def explain_lime_v1():
    """Explain predictions using LIME for model version 1"""
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
        instance_to_explain = df.iloc[0].values

        # Generate LIME explanation
        explanation = lime_explainer_v1.explain_instance(
            data_row=instance_to_explain,
            predict_fn=model_v1.predict
        )

        # Save the explanation as an image
        image_path = "logs/lime_explanation_v1.png"
        explanation.save_to_file(image_path)
        explanation.as_pyplot_figure()
        plt.savefig(image_path)
        plt.close()

        logger.info(f"LIME explanation saved to {image_path}")
        return jsonify({"message": f"LIME explanation saved to {image_path}"})
    except Exception as e:
        logger.error(f"LIME explanation error: {traceback.format_exc()}")
        return jsonify({"error": "Failed to generate LIME explanation"}), 500

@app.route('/v2/explain_lime', methods=['POST'])
def explain_lime_v2():
    """Explain predictions using LIME for model version 2"""
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
        instance_to_explain = df.iloc[0].values

        # Generate LIME explanation
        explanation = lime_explainer_v2.explain_instance(
            data_row=instance_to_explain,
            predict_fn=model_v2.predict
        )

        # Save the explanation as an image
        image_path = "logs/lime_explanation_v2.png"
        explanation.save_to_file(image_path)
        explanation.as_pyplot_figure()
        plt.savefig(image_path)
        plt.close()

        logger.info(f"LIME explanation saved to {image_path}")
        return jsonify({"message": f"LIME explanation saved to {image_path}"})
    except Exception as e:
        logger.error(f"LIME explanation error: {traceback.format_exc()}")
        return jsonify({"error": "Failed to generate LIME explanation"}), 500

def validate_input(data):
    """Validate input data against required features"""
    missing_features = [feature for feature in feature_columns if feature not in data]
    if missing_features:
        return False, f"Missing features: {missing_features}"
    return True, None

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