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

|   |   ├── feature_importance.py# 

|   |   |── predict_api.py     # Flask API for Predictions

│   │   ├── predict.py         # Making Predictions using Best Model

│── API_DOCUMENTATION.md       # Overview and process how to run the Flask prediction

│── README.md                  # Project Documentation

│── Makefile                   

│── requirements.txt           # Dependencies & Libraries

│── Dockerfile.mlapp           # Dockerfile for ML Application

│── Dockerfile.mlflow          # Dockerfile for MLFlow Server

│── Docker-compose.yml          # Docker Compose file for Multi-Container Setup



Dataset:
- Food_Drive_2023.csv: Historical donation data used for training
- Food_Drive_2024.csvL New donation data for testing and evaluation
- External: Property_Assessment_Data__Current_Calendar_Year__20240925.csv: Assessed value & geographical information
