# Food Drive Prediction API

## Overview
This API serves machine learning models to predict the number of donation bags collected based on input features such as time spent, doors in route, and assessed value. 

The API has two prediction models (v1 & v2) that can be accessed via Flask-based REST endpoints. V1 is the best model that we downloaded after performing evaluation and compare it with the other model that was tuned and it is the Random forest.
model_v1_path: "models/best_model.pkl" (applying for Linear_Regression)
model_v2_path: "models/Random_Forest.pkl"
## Base URL
This API runs on `http://127.0.0.1:5000` (Default).  
Make sure to replace `5000` with your actual **Flask port** if different.

## Available Endpoints**
| Method | Endpoint           | Description |
|--------|--------------------|-------------|
| `GET`  | `/food_drive_home` | Provides API details and usage information |
| `GET`  | `/health_status`   | Checks if the API is running correctly |
| `POST` | `/v1/predict`      | Uses Model v1 (best model) for prediction |
| `POST` | `/v2/predict`      | Uses Model v2 (Random Forest) for prediction |

## 1. Home Endpoint**
### Endpoint:** `/food_drive_home`
### Method:** `GET`
### Description:**  
Returns API usage information including available endpoints, required input format, and API description.

### **🔹 cURL Command**
curl -X GET http://127.0.0.1:5000/food_drive_home

#### Respone Format 
{
  "description": "This API serves machine learning models for predicting donation bags collected.",
  "endpoints": {
    "/v1/predict": "Predict using model version 1",
    "/v2/predict": "Predict using model version 2",
    "/health_status": "Check the health status of the API"
  },
  "request_format": {
    "time_spent": "float",
    "doors_in_route": "int",
    "assessed_value": "float"
  }
}

 ## 2. Health Check
🔹 Endpoint: /health_status
🔹 Method: GET
🔹 Description:
Checks if the API is running properly.

🔹 cURL Command
curl -X GET http://127.0.0.1:5000/health_status

#### Response Format 
{
  "status": "API is running and operational"
}

## 3. Prediction Using Model V1 - using best model to predict
🔹 Endpoint: /v1/predict
🔹 Method: POST
🔹 Description:
Takes input features in JSON format and returns a prediction using Model V1.

🔹 Required JSON Input Format
{
  "time_spent": 1.5,
  "doors_in_route": 10,
  "assessed_value": 100000
}

🔹 cURL Command
curl -X POST -H "Content-Type: application/json" -d '{"time_spent": 1.5, "doors_in_route": 10, "assessed_value": 100000}' http://127.0.0.1:5001/v1/predict

#### Response 
{
  "prediction": [
    4.621319940608169
  ]
}

## 4. Prediction Using Model V2 - Using Random Forest model to predict
🔹 Endpoint: /v2/predict
🔹 Method: POST
🔹 Description:
Takes input features in JSON format and returns a prediction using Model V2.

🔹 Required JSON Input Format
{
  "time_spent": 1.5,
  "doors_in_route": 10,
  "assessed_value": 100000
}

🔹 cURL Command

curl -X POST -H "Content-Type: application/json" -d '{"time_spent": 1.5, "doors_in_route": 10, "assessed_value": 100000}' http://127.0.0.1:5001/v2/predict

#### Response 
{
  "prediction": [
    12.201500237162701
  ]
}

## 5. Error Handling
Error Handling and Responses
HTTP Code	 Message	                Description
200	      - OK	                  - Successful prediction or health check.
400       - Bad Request	          - Missing fields or wrong data types in the request.
500	      - Internal Server Error	- Unexpected server error (e.g., model failure).

## 6. Running the API Locally
1. pip install -r requirements.txt
2. python predict_api.py
3. By default, the app will run at: http://127.0.0.1:5000
4. You can use the curl command to get the outcomes in the terminal and get the predictions. Change the values of the features according to choice but in the particular format. 
