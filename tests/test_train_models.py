
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from src.train_models import train_models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

@pytest.fixture
def generate_dummy_data():
    # Generate dummy data for testing
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def test_train_models_generate_models(generate_dummy_data):
    x_train, x_test, y_train, y_test = generate_dummy_data
    models = train_models(x_train, y_train)

    # Check if models are of the correct type
    assert isinstance(models[0], LogisticRegression)
    assert isinstance(models[1], DecisionTreeClassifier)
    assert isinstance(models[2], RandomForestClassifier)
    assert isinstance(models[3], SVC)
    assert isinstance(models[4], KNeighborsClassifier)
    assert isinstance(models[5], GaussianNB)
    assert isinstance(models[6], MLPClassifier)
    assert isinstance(models[7], VotingClassifier)

def test_train_models_fit_models(generate_dummy_data):
    x_train, x_test, y_train, y_test = generate_dummy_data
    models = train_models(x_train, y_train)

    # Check if models are fitted
    for model in models:
        assert hasattr(model, 'classes_')

def test_train_models_ensemble_classifier(generate_dummy_data):
    x_train, x_test, y_train, y_test = generate_dummy_data
    models = train_models(x_train, y_train)

    # Check if ensemble classifier contains the specified models
    ensemble_classifier = models[7]
    assert 'logreg' in ensemble_classifier.named_estimators_
    assert 'dt' in ensemble_classifier.named_estimators_
    assert 'rf' in ensemble_classifier.named_estimators_
    assert 'svm' in ensemble_classifier.named_estimators_
    assert 'knn' in ensemble_classifier.named_estimators_
    assert 'nb' in ensemble_classifier.named_estimators_
    assert 'mlp' in ensemble_classifier.named_estimators_

    # Additional checks if needed