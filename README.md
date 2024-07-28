# Customer Churn Analysis
## Overview
This repository contains the code for a machine learning project focused on predicting customer churn. Customer churn refers to the phenomenon of customers discontinuing their relationship with a business. The project employs various classification algorithms to analyze a dataset and make predictions regarding customer behavior.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Rank TAble](#rank-table)
- [Who Can Use It](#who-can-use-it)
- [Impact](#impact)
- [Contributing](#contributing)
- [License](#license)

## Introduction

### Data
The dataset (`Churn_Modelling.csv`) is located in the `data` directory.
## Data Preprocessing
The data preprocessing script (`src/data_preprocessing.py`) handles tasks such as label encoding and one-hot encoding.
## Model Training
The model training script (`src/train_models.py`) initializes classifiers, creates an ensemble, and trains each model.
## Model Evaluation
The model evaluation script (`src/evaluate_models.py`) evaluates the trained models, providing accuracy, bias, and variance metrics along with confusion matrices.
## Running the Analysis
Execute the main script (`main.py`) to run the complete analysis.
## CI/CD Workflow
The CI/CD workflow is defined in `.github/workflows/ci_cd_workflow.yml`. It runs on every push to the `main` branch.

## Project Structure

customer_churn_analysis/
│
├── data/
│   └── Churn_Modelling.csv
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train_models.py
│   └── evaluate_models.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data_preprocessing.py
│   ├── test_train_models.py
│   └── test_evaluate_models.py
│
├── .github/
│   └── workflows/
│       └── ci_cd_workflow.yml
│
├── README.md
├── requirements.txt
├── main.py
└── .gitignore


## Installation

To run this project locally, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/customer-churn-analysis.git`
2. Navigate to the project directory: `cd customer-churn-analysis`
3. Install the required dependencies: `pip install -r requirements.txt`
Install these libraries using:

## Usage

To use this project, follow these steps:

1. Ensure you have installed the required dependencies.
2. Run the Jupyter Notebook or Python script containing the code (`Customer Churn Analysis.ipynb`).
3. Explore the results and analyses provided in the notebook or script.

## Dependencies

The project relies on the following Python libraries:

- NumPy
- Pandas
- Matplotlib
- scikit-learn
- XGBoost
- LightGBM
- CatBoost
  
```bash
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm catboost
```

## Rank Table
### Without GridSearchCv
|index|Algorithm|Accuracy|Bias|Variance|
|---|---|---|---|---|
|0|LogisticRegression|78\.9|78\.9|78\.9|
|1|DecisionTreeClassifier|80\.2|100\.0|80\.2|
|2|RandomForestClassifier|86\.5|99\.9|86\.5|
|3|SVC|79\.7|79\.6|79\.7|
|4|KNeighborsClassifier|76\.4|81\.6|76\.4|
|5|GaussianNB|78\.4|78\.5|78\.4|
|6|MLPClassifier|78\.3|78\.9|78\.3|
|7|VotingClassifier|81\.2|85\.0|81\.2|

### With Grid Searchcv
|index|Algorithm|Accuracy|
|---|---|---|
|0|Logistic Regression|78\.9|
|1|Bernoulli Naive Bayes|78\.60|
|2|Gaussian Naive Bayes|78\.45|
|3|MultiNomial Naive Bayes|54\.40|
|4|Decision Trees|85\.15|
|5|Random Forest|86\.55|
|6|Support Vector Machines \(SVM\)|79\.75|
|7|k-Nearest Neighbors \(k-NN\)|77\.9|
|8|XGBoost|85\.25|
|9|LightGBM|86\.9|

### Who Can Use It

This code is intended for use by data scientists, machine learning engineers, or analysts responsible for customer analytics and retention strategies within a business or organization.

### Impact

The impact of this code lies in its potential to help businesses reduce customer churn. 
By predicting which customers are more likely to leave, companies can take proactive measures to retain them, leading to increased customer satisfaction and business profitability. 
Additionally, the code provides a framework for comparing and selecting the most suitable machine learning model for a given dataset.

## Contributing

Feel free to contribute to this project by forking the repository and submitting pull requests. Bug reports, suggestions, and feature requests are also welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

Replace the placeholders like `Your Project Name`, `Your Project Description`, etc., with the actual details relevant to your project. Additionally, if there are specific instructions for setting up, running, or contributing to the project, include them in the README.
