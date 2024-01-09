# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

# Function to load the dataset from a CSV file
def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

# Placeholder function for preprocessing data
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

# Function to train the RandomForestClassifier and TF-IDF vectorizer
def train_model(X_train, y_train):
    """
    Train a RandomForestClassifier on the given data and visualize the learning curve.

    Parameters:
        X_train (pd.Series): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        RandomForestClassifier, TfidfVectorizer: Trained model and vectorizer.
    """
    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)

    # Initialize the RandomForestClassifier
    model = RandomForestClassifier()

    # Track the learning curve during training
    train_sizes, train_scores, test_scores = learning_curve(model, X_train_vectorized, y_train, cv=5)

    # Plot the learning curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training Score')
    plt.plot(train_sizes, test_scores.mean(axis=1), label='Cross-Validation Score')
    plt.title('Learning Curve')
    plt.xlabel('Training Examples')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.show()

    # Train the model
    model.fit(X_train_vectorized, y_train)

    return model, vectorizer

# Function to evaluate the model on the test set and print metrics
def evaluate_model(model, vectorizer, X_test, y_test):
    """
    Evaluate the trained model on the test set and print metrics.

    Parameters:
        model (RandomForestClassifier): Trained model.
        vectorizer (TfidfVectorizer): Trained vectorizer.
        X_test (pd.Series): Test features.
        y_test (pd.Series): Test labels.
    """
    # Vectorize the test set
    X_test_vectorized = vectorizer.transform(X_test)
    # Make predictions
    y_pred = model.predict(X_test_vectorized)

    # Print accuracy and additional classification metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy * 100:.2f}%')
    print(classification_report(y_test, y_pred))

# Main block
if __name__ == "__main__":
    # Load dataset
    dataset = load_data('sqli.csv')
    X = dataset['Query']
    y = dataset['Label']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Placeholder for additional preprocessing steps
    X_train, y_train = preprocess_data(X_train, y_train)

    # Train the model and get the trained model and vectorizer
    model, vectorizer = train_model(X_train, y_train)

    # Evaluate the model on the test set
    evaluate_model(model, vectorizer, X_test, y_test)

    # Save the trained model and vectorizer for future use
    joblib.dump(model, 'sql_injection_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
