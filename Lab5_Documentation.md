# Running Docker-compose
Docker-compose up --build

# Service Access Information
- Service        |Flask   |    Prometheus Port  |   Host Port Mapping
+ ml-app         |5000    |   8010 (internal)   |     5000 -> 5000
+ train-metrics  |-       |   8003 (internal)   |      -----------
+ mlflow         |5000    |   ---------------   |     5002 -> 5000
+ prometheus     |-       |   9090 (internal)   |    localhost:9090
+ grafana        |-       |   ---------------   | http://localhost:3000/

# Running ML Application - Logs and metrics are sent to both MLflow and Prometheus
Training the Model (train.py)

python src/train.py

# Running the Prediction API (predict_api.py) - This starts a Flask server on port 5001. 
# This exposes /v1/predict, /v2/predict, /health_status, and /metrics
python src/predict.py

# Testing with check_api.py
python check_api.py --data '{"time_spent": 30, "doors_in_route": 50, "assessed_value": 200000}' --port 5001
# Output expected:
- INFO - Sending request to http://localhost:5001/v1/predict with data: {'time_spent': 30, 'doors_in_route': 50, 'assessed_value': 200000}
- INFO - Status code: 200
- INFO - Response:
{
  "prediction": [
    22.068840579710145
  ]
}

# Monitoring Report
# Monitoring Strategy
The ML pipeline integrates Prometheus for real-time metrics collection and Grafana for visualization. Each containerized service exposes its metrics on a unique port, which is scraped by Prometheus. This enables end-to-end visibility into API performance, model accuracy, and system resource usage.

# Key Metrics Tracked
- Prediction API:
    + http_requests_total: 
        * Total number of API calls
    + model_prediction_requests_total: 
        * Success/error rate per model version
    + model_prediction_duration_seconds
        * Response time of predictions
- Training Pipeline:
    + training_loss: 
        * Loss after each training epoch
    + training_epoch_total
        * Total epochs completed
    + validation_accuracy
        * Accuracy after each validation step
- System Resources
    + app_cpu_usage_percent
        * CPU usage of container
    + app_memory_usage_bytes
        * RAM usage of container
    + app_disk_usage_bytes
        * Disk consumption

# Configured Alerts (Prometheus): Alerts notify about performance drops, system overuse, and failures in the training process.

- HighMemoryUsage:
    + app_memory_usage_bytes > 500000000: > 500MB for 1 min
- HighCPUUsage:
    + app_cpu_usage_percent > 80: > 80% for 1 min
- HighErrorRate:
    + rate(flask_http_requests_total{status="500"}[5m]) > 0.05: > 5% error rate
- HighTrainingLoss:
    + training_loss > 0.5: > 0.5 for 5 min
- LowValidationAccuracy:
    + validation_accuracy < 0.8: < 80% for 5 min

