
import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_preprocessing import preprocess_data

@pytest.fixture
def sample_data():
    # Create a sample DataFrame for testing
    data = {
        'Feature1': [1, 2, 3, 4, 5],
        'Feature2': ['A', 'B', 'A', 'C', 'B'],
        'Target': [0, 1, 0, 1, 0]
    }
    df = pd.DataFrame(data)
    return df

def test_preprocess_data(sample_data):
    # Arrange
    x_train, x_test, y_train, y_test = preprocess_data(sample_data)

    # Assert
    assert isinstance(x_train, pd.DataFrame)
    assert isinstance(x_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

    # Add more specific assertions based on your preprocessing steps
    # For example, check the shape, data types, or specific values
    assert x_train.shape[1] == 2  # Assuming some columns were dropped during preprocessing
    assert 'Feature2' not in x_train.columns  # Assuming Feature2 was one-hot encoded
    assert x_test.shape[0] == 1/5 * sample_data.shape[0]  # Assuming a 80/20 split
