import os
import time
import logging
from logging.handlers import RotatingFileHandler
from utils.monitoring import RegressionMonitor

# Initialize the RegressionMonitor
monitor = RegressionMonitor(port=8004)

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

logger.info("Regression training process completed.")