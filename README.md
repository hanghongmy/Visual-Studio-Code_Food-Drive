# Overview
This project aims to analyze and predict donation bag collections based on historical Food Drive data from 2023 and 2024. Using various regression models, we identify trends, optimize predictions, and improve data-driven decision-making for future donation drives.

# Key Features:
- Data Cleaning & Preprocessing using Pandas
- Feature Engineering for enhanced predictions
- Multiple Regression Models (Linear Regression, Decision Tree, Random Forest)
- Hyperparameter Tuning for optimal performance
- Model Evaluation & Selection based on R² Score
- Final Prediction Using the Best Model
- API for prediction using Flask

# Project Structure:


VISUAL-STUDIO-CODE_FOOD-DRIVE/

│── ml_project/

│   ├── data/                   # Data Directory

│   │   ├── raw/                # Raw CSV Files (Unprocessed)

│   │   ├── external/           # External Datasets (Property Assessment)

│   │   ├── processed/          # Cleaned & Processed Datasets

│   ├── models/                 # Saved Machine Learning Models

│   ├── reports                 # Saved Feature Importance & Heatmap

│   ├── src/                    # Source Code for ML Pipeline

│   │   ├── preprocess.py       # Data Preprocessing & Cleaning

│   │   ├── train.py            # Model Training & Saving

│   │   ├── evaluate.py         # Model Evaluation & Best Model Selection

|   |   |── predict_api.py      # Flask API for Predictions

│   │   ├── predict.py          # Making Predictions using Best Model

│   │   ├── logging_config.py   # Logging config

│── API_DOCUMENTATION.md        # Overview and process how to run the Flask prediction

│── README.md                   # Project Documentation

│── Makefile

│── requirements.txt            # Dependencies & Libraries

│── Dockerfile.mlapp            # Dockerfile for ML Application

│── Dockerfile.mlflow           # Dockerfile for MLFlow Server

│── Dockerfile.train            # Dockerfile for MLFlow Server

│── Docker-compose.yml          # Docker Compose file for Multi-Container Setup


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
- Endpoint: /health_status
- Method: GET
- Description: Check if the API is running
- Example curl http://127.0.0.1:5000/health_status

Prediction (v1):
- Endpoint: /v1/predict
- Method: POST
- Description: Predict donation bags using model version 1 as Best_model - Linear Regression
- Example curl http://127.0.0.1.5000/v1/predict

Prediction (v2):
- Endpoint: /v2/predict
- Method: POST
- Description: Predict donation bags using model version 2 as Random_Forest
- Example curl http://127.0.0.1.5000/v2/predict

MLflow Integration
Experiment Tracking:
- The MLflow server is accessible at http://127.0.0.1.5001
- All experiments and runs are logged the experiment name CMPT2500

Docker Hub links:
- ML Application Image: https://hub.docker.com/repository/docker/hanghongmy/ml-application/tags/latest/sha256-4a871b4c6b87a9c393669345bb791280a999b5d20a5778c716c72b04f23f1035
- MLflow Images: https://hub.docker.com/repository/docker/hanghongmy/ml-application/image-management