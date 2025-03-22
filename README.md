Overview
This project aims to analyze and predict donation bag collections based on historical Food Drive data from 2023 and 2024. Using various regression models, we identify trends, optimize predictions, and improve data-driven decision-making for future donation drives.

Key Features:

- Data Cleaning & Preprocessing using Pandas
- Feature Engineering for enhanced predictions
- Multiple Regression Models (Linear, Decision Tree, Random Forest, XGBoost)
- Hyperparameter Tuning for optimal performance
- Model Evaluation & Selection based on R² Score
- Final Prediction Using the Best Model

Project Structure:


Food Drive-2500/

│── ml_project/

│   ├── data/                  # Data Directory

│   │   ├── raw/               # Raw CSV Files (Unprocessed)

│   │   ├── external/          # External Datasets (Property Assessment)

│   │   ├── processed/         # Cleaned & Processed Datasets

│   ├── models/                # Saved Machine Learning Models

│   ├── reports                # Saved Feature Importance & Heatmap

│   ├── src/                   # Source Code for ML Pipeline

│   │   ├── preprocess.py      # Data Preprocessing & Cleaning

│   │   ├── train.py           # Model Training & Saving

│   │   ├── evaluate.py        # Model Evaluation & Best Model Selection

|   |   ├── feature_analysis.py# 

│   │   ├── predict.py         # Making Predictions using Best Model

│── README.md                  # Project Documentation

│── requirements.txt           # Dependencies & Libraries

Dataset:
- Food_Drive_2023.csv: Historical donation data used for training
- Food_Drive_2024.csvL New donation data for testing and evaluation
- External: Property_Assessment_Data__Current_Calendar_Year__20240925.csv: Assessed value & geographical information

# Food Drive ML Application & MLflow Tracking

This repository contains:
- An ML application for predicting donation bag collections.
- MLflow for experiment tracking.
Both are containerized with Docker and orchestrated using Docker Compose.

## How to Run the Project

### Prerequisites:
- Docker
- Docker Compose

### 1. Clone the Repository

1. git clone https://github.com/hanghongmy/Visual-Studio-Code_Food-Drive/tree/FOOD_DRIVE_2024
cd Visual-Studio-Code_Food-Drive-OLD

2. Build and Run the Containers using docker-compose up --build

# Check running Containers:
docker ps

3. Access the Servers
ML Application API: http://127.0.0.1:5000
MLflow UI: http://127.0.0.1:5001

API Endpoints

Health Check:

- Endpoint: /health_status
- Method: GET
- Description: Check if the API is running
- Example curl http://127.0.0.1:5000/health_status
- Prediction (v1):

Endpoint: /v1/predict
- Method: POST
- Description: Predict donation bags using model - - version 1 as Best_model - Linear Regression
- Example curl http://127.0.0.1.5000/v1/predict
- Prediction (v2):

Endpoint: /v2/predict
- Method: POST
- Description: Predict donation bags using model version 2 as Random_Forest
- Example curl http://127.0.0.1.5000/v2/predict
- MLflow Integration Experiment Tracking:

The MLflow server is accessible at http://127.0.0.1.5001

All experiments and runs are logged the experiment name cmpt2500-452908

Docker Hub links:

- ML Apllication Image: https://hub.docker.com/repository/docker/kiranpreet0850/ml-application/general

- ML Flow Image: https://hub.docker.com/repository/docker/kiranpreet0850/mlflow-tracking/general

