# predict.py
import requests
import json

query_to_predict = "SELECT * FROM users WHERE username = 'admin' AND password = 'password' OR 1=1; --"
predict_url = 'http://127.0.0.1:5000/predict'

data = {'query': query_to_predict}

response = requests.post(predict_url, json=data)
print(response.json())
