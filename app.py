from flask import Flask, request, render_template, jsonify
import joblib
import logging

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = joblib.load('sql_injection_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Configure logging
logging.basicConfig(level=logging.INFO)

def predict_query(query):
    """Predict whether the input query is a SQL injection or not."""
    try:
        query_vectorized = vectorizer.transform([query])
        prediction = model.predict(query_vectorized)[0]
        return int(prediction)  # Ensure it returns a JSON-compatible integer
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def home():
    """
    Renders the SQL Injection Detector UI.
    - Handles form submissions (POST)
    - Displays results only if a query is submitted
    """
    prediction = None
    error = None

    if request.method == 'POST':
        query = request.form.get('query', '').strip()

        if not query:
            error = "Please enter an SQL query."
        else:
            prediction = predict_query(query)
            if prediction is None:
                error = "Prediction failed."

    return render_template('index.html', prediction=prediction, error=error)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles predictions for both:
    - Web form submissions (HTML)
    - API calls (JSON)
    """
    try:
        if request.is_json:
            data = request.get_json()
            query = data.get('query', '').strip()
        else:
            query = request.form.get('query', '').strip()

        if not query:
            raise ValueError("No query provided")

        prediction = predict_query(query)
        if prediction is None:
            raise ValueError("Prediction failed")

        # Return JSON response for API calls
        if request.is_json:
            return jsonify({'prediction': prediction})

        # Render UI with result for web form
        return render_template('index.html', prediction=prediction, error=None)

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        if request.is_json:
            return jsonify({'error': str(e)}), 400
        return render_template('index.html', prediction=None, error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
