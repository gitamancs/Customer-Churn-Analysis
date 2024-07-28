
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from src.evaluate_models import evaluate_models
from sklearn.model_selection import train_test_split

@pytest.fixture
def sample_data():
    # Generate synthetic data for testing
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def test_evaluate_models(sample_data):
    X_train, X_test, y_train, y_test = sample_data

    # Initialize a sample model for testing
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    rank_table = evaluate_models([model], X_train, y_train, X_test, y_test)

    # Assert that the rank table has the expected structure
    assert isinstance(rank_table, list)
    assert all(isinstance(entry, list) and len(entry) == 4 for entry in rank_table)

    # Assert that accuracy, bias, and variance are within valid ranges
    for entry in rank_table:
        assert 0 <= entry[1] <= 100  # Accuracy
        assert 0 <= entry[2] <= 100  # Bias
        assert 0 <= entry[3] <= 100  # Variance

    # Add more specific assertions based on your actual implementation
