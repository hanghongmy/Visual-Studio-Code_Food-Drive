#!/usr/bin/env python3
"""
Reusable client for interacting with the Food Drive Prediction API

Usage:
python client.py --data '{"time_spent": 30, "doors_in_route": 50, "assessed_value": 200000}' --version v1
"""

import argparse
import requests
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Food Drive Prediction API Client')
    parser.add_argument('--data', type=str, required=True, 
                        help='Input data as a JSON string (e.g., \'{"time_spent": 30, "doors_in_route": 50, "assessed_value": 200000}\')')
    parser.add_argument('--version', type=str, default='v1', choices=['v1', 'v2'],
                        help='API version to use (v1 or v2, default: v1)')
    parser.add_argument('--url', type=str, default='http://localhost:5000',
                        help='API base URL (default: http://localhost:5000)')
    return parser.parse_args()

def send_prediction_request(data, version='v1', base_url='http://localhost:5000'):
    """Send a prediction request to the Food Drive Prediction API."""
    # Prepare API endpoint
    endpoint = f"{base_url}/{version}/predict"
    
    # Send request
    try:
        logger.info(f"Sending request to {endpoint} with data: {data}")
        response = requests.post(endpoint, json=data)
        response.raise_for_status()  # Raise exception for HTTP errors
        logger.info(f"Status code: {response.status_code}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            logger.error(f"Response status code: {e.response.status_code}")
            logger.error(f"Response text: {e.response.text}")
        return None

def main():
    """Main function."""
    args = parse_arguments()
    
    # Parse input data
    try:
        input_data = json.loads(args.data)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid input data format: {e}")
        return
    
    logger.info(f"Sending data to API: {input_data}")
    logger.info(f"Using API version: {args.version}")
    
    # Send prediction request
    result = send_prediction_request(
        input_data, 
        version=args.version,
        base_url=args.url
    )
    
    # Display results
    if result:
        logger.info("\nPrediction Results:")
        if result.get('prediction'):
            logger.info(f"Prediction: {result['prediction']}")
            print(f"Prediction: {result['prediction']}")
        else:
            logger.error(f"Error: {result.get('error', 'Unknown error')}")
            print(f"Error: {result.get('error', 'Unknown error')}")
    else:
        logger.error("Failed to get a response from the API")
        print("Failed to get a response from the API")

if __name__ == "__main__":
    main()