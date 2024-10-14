import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@pytest.fixture
def dataset():
    """Fixture for loading and splitting the dataset."""
    data = load_breast_cancer()
    X = data.data
    y = data.target
    return train_test_split(X, y, test_size=0.3, random_state=42)

@pytest.fixture
def model():
    """Fixture for creating the Logistic Regression model."""
    return LogisticRegression(max_iter=10000)

def test_dataset_shape(dataset):
    """Test that the dataset is split correctly and has the right shape."""
    X_train, X_test, y_train, y_test = dataset
    assert X_train.shape[0] > 0, "Training set is empty"
    assert X_test.shape[0] > 0, "Test set is empty"
    assert X_train.shape[1] == X_test.shape[1], "Feature dimensions are inconsistent between train and test sets"

def test_model_training(model, dataset):
    """Test that the model trains without errors."""
    X_train, X_test, y_train, y_test = dataset
    model.fit(X_train, y_train)
    assert model.coef_.shape[0] == 1, "The model should be a binary classifier"
    assert model.coef_.shape[1] == X_train.shape[1], "Model coefficient dimensions do not match input features"

def test_predictions_shape(model, dataset):
    """Test that the prediction shape matches the test data."""
    X_train, X_test, y_train, y_test = dataset
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    assert y_pred.shape == y_test.shape, "Predicted labels shape does not match true labels"

def test_accuracy(model, dataset):
    """Test that the model accuracy is reasonable."""
    X_train, X_test, y_train, y_test = dataset
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.8, f"Accuracy too low: {accuracy}"

def test_binary_output(model, dataset):
    """Test that the model predicts binary outcomes (0 or 1)."""
    X_train, X_test, y_train, y_test = dataset
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    unique_values = np.unique(y_pred)
    assert set(unique_values).issubset({0, 1}), "Model predictions are not binary"
