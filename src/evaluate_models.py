
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_models(models,x_train, y_train, x_test, y_test):
    rank_table = []

    for clf in models:
        # Make predictions on the test data
        predictions = clf.predict(x_test)

        # Evaluate the model
        clf_ac = accuracy_score(predictions, y_test) * 100
        clf_cm = confusion_matrix(predictions, y_test)
        clf_cr = classification_report(predictions, y_test)

        # Get model-specific information
        model_name = type(clf).__name__

        # Print information
        print(f"Model: {model_name}")
        print(f"Accuracy: {clf_ac:.4f}")
        print("Classification Report:\n", clf_cr)
        print(f"{model_name}'s bias is ", clf.score(x_train, y_train))
        print(f"{model_name}'s variance is ", clf.score(x_test, y_test))

        # Plotting the Confusion Matrix
        plt.figure(figsize=(2, 2))
        sns.heatmap(clf_cm, annot=True, fmt='d', cmap='Blues_r')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion matrix for {model_name}')
        plt.show()

        rank_table.append([model_name, clf_ac, clf.score(x_train,y_train) * 100, clf.score(x_test, y_test) * 100])

    return rank_table
