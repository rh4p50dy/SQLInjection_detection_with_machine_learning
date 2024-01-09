import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

def preprocess_data(X, y):
    """
    Preprocess the data if needed. (Placeholder function)

    Parameters:
        X (pd.Series): Input features.
        y (pd.Series): Target labels.

    Returns:
        pd.Series, pd.Series: Preprocessed input features and labels.
    """
    # Placeholder for additional preprocessing steps
    return X, y

def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier on the given data.

    Parameters:
        X_train (pd.Series): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        RandomForestClassifier, TfidfVectorizer: Trained model and vectorizer.
    """
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)

    model = RandomForestClassifier()
    model.fit(X_train_vectorized, y_train)

    return model, vectorizer

def evaluate_model(model, vectorizer, X_test, y_test):
    """
    Evaluate the trained model on the test set and print metrics.

    Parameters:
        model (RandomForestClassifier): Trained model.
        vectorizer (TfidfVectorizer): Trained vectorizer.
        X_test (pd.Series): Test features.
        y_test (pd.Series): Test labels.
    """
    X_test_vectorized = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_vectorized)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')

    # Additional metrics
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Load dataset
    dataset = load_data('sqli.csv')
    X = dataset['Query']
    y = dataset['Label']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess data
    X_train, y_train = preprocess_data(X_train, y_train)

    # Train the model
    model, vectorizer = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, vectorizer, X_test, y_test)

    # Save the model and vectorizer
    joblib.dump(model, 'sql_injection_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
