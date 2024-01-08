# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('sql_injection_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_query(query):
    query_vectorized = vectorizer.transform([query])

    prediction = model.predict(query_vectorized)[0]
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        query = data['query']

        prediction = predict_query(query)
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
