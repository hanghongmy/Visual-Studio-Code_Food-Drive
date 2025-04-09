#!/usr/bin/env python3
"""
Simple script to check the Food Drive Prediction API directly
"""

import requests
import json
import argparse
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_api(data, endpoint, host="localhost", port=5000):
    """Check the API with input data"""
    url = f"http://{host}:{port}{endpoint}"
    headers = {'Content-Type': 'application/json'}

    logger.info(f"Sending request to {url} with data: {data}")
    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        logger.info(f"Status code: {response.status_code}")
        if response.status_code == 200:
            logger.info(f"Response:\n{json.dumps(response.json(), indent=2)}")
        else:
            logger.error(f"Error response:\n{response.text}")
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to the API: {e}")
    except requests.exceptions.Timeout as e:
        logger.error(f"Request timed out: {e}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Check Food Drive Prediction API')
    parser.add_argument('--data', required=True, 
                        help='Input data as a JSON string (e.g., \'{"time_spent": 30, "doors_in_route": 50, "assessed_value": 200000}\')')
    parser.add_argument('--endpoint', default='/v1/predict', 
                        help='API endpoint (e.g., /v1/predict or /v2/predict)')
    parser.add_argument('--host', default='localhost', 
                        help='API host (default: localhost)')
    parser.add_argument('--port', type=int, default=5000, 
                        help='API port (default: 5000)')
    
    args = parser.parse_args()

    # Parse input data
    try:
        input_data = json.loads(args.data)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid input data format: {e}")
        return

    # Validate input data
    required_fields = {"time_spent", "doors_in_route", "assessed_value"}
    missing_fields = required_fields - input_data.keys()
    if missing_fields:
        logger.error(f"Missing required fields: {missing_fields}")
        return

    # Check the API
    check_api(input_data, args.endpoint, args.host, args.port)

if __name__ == "__main__":
    main()