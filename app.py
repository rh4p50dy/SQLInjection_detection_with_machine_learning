# app.py
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = joblib.load('sql_injection_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_query(query):
    """
    Predict whether the input query is a SQL injection or not.

    Parameters:
        query (str): Input SQL query.

    Returns:
        int: Prediction label (0 for non-SQL injection, 1 for SQL injection).
    """
    query_vectorized = vectorizer.transform([query])

    prediction = model.predict(query_vectorized)[0]
    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint for making predictions.

    Expects a JSON payload with the 'query' field representing the SQL query.

    Returns:
        JSON: Prediction result (0 or 1).
    """
    try:
        data = request.get_json()
        query = data['query']

        prediction = predict_query(query)
        return jsonify({'prediction': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)
