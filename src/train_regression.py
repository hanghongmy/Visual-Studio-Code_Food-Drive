import os
import time
import logging
from logging.handlers import RotatingFileHandler
from utils.monitoring import RegressionMonitor
from threading import Thread
from flask import Flask, jsonify
"""
Regression training script with monitoring and logging. This used for train a regression model.
    - Monitoring: uses the RegressionMonitor to track training progress, including epochs,
    batches, training loss.
    - Logging: configures logging to record training progress and metrics.
    - Simulated training process: simulates a training process with epochs and batches,
    recording metrics at each step.
    - Metrics: track and logs regression metrics such as MSE, RMSE, MAE, and R2
    
"""
# Initialize Flask app
app = Flask(__name__)

# Configure logging
log_directory = 'logs'
os.makedirs(log_directory, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add a rotating file handler for the regression module
file_handler = RotatingFileHandler(
    f'{log_directory}/regression.log',
    maxBytes=10485760,  # 10MB
    backupCount=5  # Keep up to 5 backup files
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger = logging.getLogger('ml_app.regression')
logger.addHandler(file_handler)

logger.info("Starting regression training process...")

# Initialize the RegressionMonitor
monitor = RegressionMonitor(port=8002)

# Define a route to expose metrics
@app.route('/metrics', methods=['GET'])
def get_metrics():
    monitor.flask_requests.inc()  # Increment the Flask HTTP requests counter
    return jsonify(monitor.get_metrics())

# Function to run Flask app in a separate thread
def run_flask_app():
    app.run(port=5000, debug=False)

# Start Flask app in a separate thread
flask_thread = Thread(target=run_flask_app)
flask_thread.daemon = True
flask_thread.start()

logger.info("Flask metrics server started on port 5000.")

# Simulate training process
for epoch in range(10):
    monitor.record_epoch()
    logger.info(f"Epoch {epoch + 1} started.")

    for batch in range(100):
        monitor.record_batch()
        loss = 0.01 * (100 - batch)  # Simulated loss
        monitor.record_loss(loss)
        logger.info(f"Epoch {epoch + 1}, Batch {batch + 1}: Training loss = {loss:.4f}")
        time.sleep(0.1)  # Simulate training time

    # Simulate validation metrics
    validation_loss = 0.01 * (10 - epoch)
    validation_accuracy = 0.8 + 0.02 * epoch
    monitor.record_validation_metrics(loss=validation_loss, accuracy=validation_accuracy)
    logger.info(f"Epoch {epoch + 1}: Validation loss = {validation_loss:.4f}, Validation accuracy = {validation_accuracy:.4f}")

    # Simulate regression metrics
    mse = validation_loss
    rmse = mse ** 0.5
    mae = mse * 0.8
    r_squared = 0.9 + 0.01 * epoch
    monitor.record_metrics(mse=mse, rmse=rmse, mae=mae, r_squared=r_squared)
    logger.info(f"Epoch {epoch + 1}: MSE = {mse:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}, R² = {r_squared:.4f}")
    # Update metrics one last time
    monitor.record_metrics(mse=mse, rmse=rmse, mae=mae, r_squared=r_squared)

logger.info("Regression training process completed.")

if __name__ == "__main__":
    logger.info("Training completed. Keeping the application running to expose metrics.")
    
    app.run(host="0.0.0.0", port=8004, debug=True)
    while True:
        time.sleep(1)