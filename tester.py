# tester.py
import requests
import json

# Sample SQL query for prediction
query_to_predict = "SELECT * FROM users WHERE username = 'admin' AND password = 'password' OR 1=1; --"

# URL of the prediction endpoint
predict_url = 'http://127.0.0.1:5000/predict'

# Data payload for the POST request
data = {'query': query_to_predict}

# Send a POST request to the prediction endpoint
response = requests.post(predict_url, json=data)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse and print the JSON response
    prediction_result = response.json()
    print("Prediction Result:", prediction_result)
else:
    # Print an error message if the request was not successful
    print(f"Error: {response.status_code}, {response.text}")
