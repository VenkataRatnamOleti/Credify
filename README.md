# 💳 Crediy

Crediy is a machine learning based loan approval prediction system.

It predicts whether a loan application is likely to be approved based on applicant financial, employment and loan-related information.

## Live Link

👉 [Credify Streamlit Deployment](https://credify-ai.streamlit.app/)

## Features

- Applicant income analysis
- Coapplicant income
- Credit score
- Debt-to-income ratio
- Existing loans
- Savings
- Collateral value
- Loan amount
- Loan term
- Employment status
- Marital status
- Loan purpose
- Property area
- Education level
- Gender
- Employer category

## Machine Learning

The project uses supervised machine learning for binary classification.

Models explored:

- Logistic Regression
- K-Nearest Neighbors
- Gaussian Naive Bayes

The final deployed model is Logistic Regression.

## Model Evaluation

The final Logistic Regression model achieved:

- Accuracy: 87.50%
- Precision: 79.03%
- Recall: 80.33%
- F1 Score: 79.67%

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt