# app.py
import pandas as pd
from flask import Flask, render_template, request, jsonify
import pickle
from sklearn.preprocessing import LabelEncoder
# from src.data_preprocessing import preprocess_data

app = Flask(__name__)

# Load trained models from pickle file
with open('trained_models.pkl', 'rb') as f:
    models = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user input
            user_input = {
                'CreditScore': float(request.form['feature1']),
                # 'Geography': str(request.form['feature2']),
                'Gender': str(request.form['feature3']),
                'Age': float(request.form['feature4']),
                'Tenure': float(request.form['feature5']),
                'Balance': float(request.form['feature6']),
                'NumOfProducts': float(request.form['feature7']),
                'HasCrCard': float(request.form['feature8']),
                'IsActiveMember': float(request.form['feature9']),
                'EstimatedSalary': float(request.form['feature10']),
            }

            # Preprocess user input
            input_data = pd.DataFrame([user_input])

            # Extracting features
            x = input_data.iloc[:].values
            # Label encoding the 'gender' column
            label_encoder_age = LabelEncoder()
            x[:, 1] = label_encoder_age.fit_transform(x[:,1])


            # Use the ensemble classifier for prediction
            prediction = models[-1].predict(x)

            # Convert the prediction to an interpretable format (e.g., 'Churn' or 'Not Churn')
            result = 'Churn' if prediction[0] == 1 else 'Not Churn'

            # return jsonify({'prediction': result})
            return render_template('result.html', result=result)

        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
'''
# app.py
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from src.data_preprocessing import preprocess_data
from src.train_models import train_models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# app.py

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input
        #features = [float(request.form['feature1']), str(request.form['feature2']), str(request.form['feature3']), float(request.form['feature4']), float(request.form['feature5']), float(request.form['feature6']), float(request.form['feature7']), float(request.form['feature8']), float(request.form['feature9']), float(request.form['feature10'])]
        #preprocessed_data = np.array(features).reshape(1, -1)
        new_data = pd.DataFrame(
            {
                'CreditScore': float(request.form['feature1']),
                'Geography': str(request.form['feature2']),
                'Gender' : str(request.form['feature3']),
                'Age' : float(request.form['feature4']),
                'Tenure': float(request.form['feature5']),
                'Balance' : float(request.form['feature6']),
                'NumOfProducts' : float(request.form['feature7']),
                'HasCrCard' : float(request.form['feature8']),
                'IsActiveMember' : float(request.form['feature9']),
                'EstimatedSalary' : float(request.form['feature10']),
            },
            index=[0]
        )
        # Load and preprocess data
        #x_train, _, y_train, _ = preprocess_data('data/Churn_Modelling.csv')
        # Load preprocessed data from pickle file
        with open('preprocessed_data.pkl', 'rb') as f:
            # x_train, x_test, y_train, y_test, label_encoder_geography, label_encoder_gender = pickle.load(f)
            preprocessor = pickle.load(f)
            # x_train, x_test, y_train, y_test = pickle.load(f)
            new_data = preprocess_data(new_data)

        
        
        # Apply label encoding to 'Geography' and 'Gender'
        # new_data['Geography'] = label_encoder_geography.transform(new_data['Geography'])
        # new_data['Gender'] = label_encoder_gender.transform(new_data['Gender'])

        # Train models (if not already trained)
        #models = train_models(x_train, y_train)
        # Load trained models from pickle file
        with open('trained_models.pkl', 'rb') as f:
            models = pickle.load(f)

        # Use the ensemble classifier for prediction
        # prediction = models[-1].predict([features])[0]
        prediction = models[-1].predict(new_data.values)

        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
'''


