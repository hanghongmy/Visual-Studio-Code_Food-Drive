# Overview
This project aims to analyze and predict donation bag collections based on historical Food Drive data from 2023 and 2024. Using various regression models, we identify trends, optimize predictions, and improve data-driven decision-making for future donation drives.

# Key Features:
✅ Data Preprocessing with Pandas

🔍 Feature Engineering for better model performance

🤖 Multiple Regression Models: Linear Regression, Decision Tree, Random Forest

🛠️ Hyperparameter Tuning using GridSearchCV

📊 Model Evaluation with R², MSE, and MAE

🏆 Best Model Selection and saving

🌐 Flask API for real-time predictions

📈 MLflow for experiment tracking

📊 Prometheus & Grafana for model and system monitoring

🐳 Docker & Docker Compose for containerized deployment

# Project Structure:


VISUAL-STUDIO-CODE_FOOD-DRIVE/

Visual-Studio-Code_Food-Drive/
│
├── data/
│   ├── raw/                  # Unprocessed input CSV files
│   ├── external/             # Assessed value & geo data
│   └── processed/            # Cleaned datasets
│
├── models/                   # Saved trained models (.pkl)
├── reports/                  # Feature importance, heatmaps
├── src/                      # Core ML scripts
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict_api.py
│   ├── predict.py
│   ├── logging_config.py
│   └── utils/monitoring.py   # Monitoring utilities
│
├── configs/                  # YAML config files for training & prediction
├── logs/                     # App & training logs
│
├── requirements.txt          # Python dependencies
├── Dockerfile.mlapp          # Flask app Dockerfile
├── Dockerfile.mlflow         # MLflow Dockerfile
├── Dockerfile.train          # Model training Dockerfile
├── docker-compose.yml        # Multi-container orchestration
├── API_DOCUMENTATION.md      # API usage examples
├── Makefile
└── README.md                 # Project overview



Dataset:
- Food_Drive_2023.csv: Historical donation data used for training
- Food_Drive_2024.csvL New donation data for testing and evaluation
- External: Property_Assessment_Data__Current_Calendar_Year__20240925.csv: Assessed value & geographical information

Setup Instructions
1. Clone the Repository
git clone https://github.com/hanghongmy/Visual-Studio-Code_Food-Drive.git

2. Build and Run the Containers
docker-compose up --build
# Check running Containers:
docker ps

3. Access the Servers
- ML Application API: http://127.0.0.1:5000
- MLflow UI: http://127.0.0.1:5001

API Endpoints

Health Check:
- Endpoint: GET /health_status
- Method: GET
- Description: Check if the API is running
- Example curl http://127.0.0.1:5001/health_status

Prediction (v1):
- Endpoint: /v1/predict
- Method: POST
- Description: Predict donation bags using model version 1 as Best_model - Linear Regression
- Example curl http://127.0.0.1.5001/v1/predict

Prediction (v2):
- Endpoint: /v2/predict
- Method: POST
- Description: Predict donation bags using model version 2 as Random_Forest
- Example curl http://127.0.0.1.5001/v2/predict

MLflow Integration: Automatically logs: parameters, metrics, and model artifacts for each run
Experiment Tracking:
- The MLflow server is accessible at http://localhost:5002
- All experiments and runs are logged the experiment name CMPT2500

Docker Hub links:
- ML Application Image: https://hub.docker.com/repository/docker/hanghongmy/ml-application
- MLflow Images: https://hub.docker.com/repository/docker/hanghongmy/mlflow

📬 Contact
If you encounter issues or have suggestions, feel free to open an issue or reach out via GitHub.