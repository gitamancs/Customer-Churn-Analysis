
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder
from sklearn.preprocessing import StandardScaler
# import pickle

def preprocess_data(input_file):
    # Load the dataset
    df = pd.read_csv(input_file)

    # Extracting features and target variable
    x = df.iloc[:, 3:-1].values
    y = df.iloc[:, -1].values

    # Label encoding the 'Gender' column
    label_encoder_age = LabelEncoder()
    x[:, 1] = label_encoder_age.fit_transform(x[:, 1])

    # Splitting the dataset into the Training set and Test set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    # Feature Scaling
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    # if save_to:
    #     with open(save_to, 'wb') as f:
    #         pickle.dump((x_train, x_test, y_train, y_test), f)

    return x_train, x_test, y_train, y_test
