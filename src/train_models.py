
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import pickle

def train_models(x_train, y_train , save_to = None):
    # Initialize classifiers
    logreg = LogisticRegression()
    dt_classifier = DecisionTreeClassifier()
    rf_classifier = RandomForestClassifier()
    svm_classifier = SVC()
    knn_classifier = KNeighborsClassifier()
    nb_classifier = GaussianNB()
    mlp_classifier = MLPClassifier()

    # Create an ensemble of classifiers
    ensemble_classifier = VotingClassifier(estimators=[
        ('logreg', logreg),
        ('dt', dt_classifier),
        ('rf', rf_classifier),
        ('svm', svm_classifier),
        ('knn', knn_classifier),
        ('nb', nb_classifier),
        ('mlp', mlp_classifier)
    ], voting='hard')

    # Train models
    logreg.fit(x_train, y_train)
    dt_classifier.fit(x_train, y_train)
    rf_classifier.fit(x_train, y_train)
    svm_classifier.fit(x_train, y_train)
    knn_classifier.fit(x_train, y_train)
    nb_classifier.fit(x_train, y_train)
    mlp_classifier.fit(x_train, y_train)
    ensemble_classifier.fit(x_train, y_train)

    if save_to:
        with open(save_to, 'wb') as f:
            pickle.dump((logreg, dt_classifier, rf_classifier, svm_classifier, knn_classifier, nb_classifier, mlp_classifier, ensemble_classifier), f)

    return logreg, dt_classifier, rf_classifier, svm_classifier, knn_classifier, nb_classifier, mlp_classifier, ensemble_classifier
