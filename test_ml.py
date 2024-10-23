import pytest
import pandas as pd
import numpy as np
from ml.data import process_data
from ml.model import train_model, compute_model_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Test data loading and processing
def test_process_data():
    """
    Test the process_data function to ensure that it correctly processes the
    input dataset and returns the expected types for features and labels.
    It checks that the output is a NumPy array for X, a NumPy array for y,
    and that the encoder and label binarizer are not None.
    """
    data = pd.read_csv("./data/census.csv")
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Call the process_data function
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label="salary", training=True)

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert encoder is not None
    assert lb is not None


# Test if model is a Logistic Regression
def test_train_model():
    """
    Test the train_model function to verify that it trains a Logistic
    Regression model on the training data. It ensures that the returned
    model is an instance of LogisticRegression.
    """
    data = pd.read_csv("./data/census.csv")
    train, _ = train_test_split(data, test_size=0.2, random_state=42)
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Call the process_data function
    X_train, y_train, _, _ = process_data(train, categorical_features=cat_features, label="salary", training=True)

    # Scale the training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train the model with scaled data
    model = train_model(X_train_scaled, y_train)
    assert isinstance(model, LogisticRegression)


# Test that the train and test datasets have the expected size
def test_data_split():
    """
    Test the data splitting functionality to ensure that the sizes of
    the training and testing datasets are as expected, following an
    80/20 split. It checks that the training dataset contains 80%
    of the original data and the test dataset contains 20%.
    """
    data = pd.read_csv("./data/census.csv")
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    expected_train_length = int(0.8 * len(data))
    expected_test_length = int(0.2 * len(data))

    # Assert that the lengths are either the expected size or one higher.
    # This accounts for one additional sample due to potential rounding.
    assert len(train) == expected_train_length or len(train) == expected_train_length + 1
    assert len(test) == expected_test_length or len(test) == expected_test_length + 1
