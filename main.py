
from src.data_preprocessing import preprocess_data
from src.train_models import train_models
from src.evaluate_models import evaluate_models
import pickle
# Load and preprocess data
#x_train, x_test, y_train, y_test = preprocess_data('data/Churn_Modelling.csv', save_to='preprocessed_data.pkl')
x_train, x_test, y_train, y_test = preprocess_data('data/Churn_Modelling.csv')
#pickle.dump(x_train, x_test, y_train, y_test)

# Open the file in binary write mode
with open('data/preprocessed_data.pkl', 'wb') as file:
    # Use pickle.dump() to serialize and write the data to the file
    pickle.dump((x_train, x_test, y_train, y_test), file)
# Train models
models = train_models(x_train, y_train, save_to='trained_models.pkl')

# Evaluate models
rank_table = evaluate_models(models,x_train, y_train, x_test, y_test)

# Display results or save them as needed
print(rank_table)

