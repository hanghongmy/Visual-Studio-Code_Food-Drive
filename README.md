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
