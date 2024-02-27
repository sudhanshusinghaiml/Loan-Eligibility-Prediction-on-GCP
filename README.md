# Loan-Eligibility-Prediction-on-GCP
This is a sample project for validating if the customer is eligible for a Loan. The model takes into account the important factors such as Crect Score, Loan Amount, Monthly Debt, Years of Credit History and many other features.  

## Installation
```
$ git clone https://github.com/sudhanshusinghaiml/Loan-Eligibility-Prediction-on-GCP.git
$ cd Loan-Eligibiltiy-Prediction-on-GCP
```

## Tech Stack:
 - **Language**: Python
 - **Libraries**: Flask, gunicorn, scipy, xgboost, joblib, seaborn, scikit_learn
 - **Services:** Flask, Docker, GCP, Gunicorn, Google Cloud Serverless Services

## Cloud Services:
 - Cloud Build
 - Cloud Run
 - Cloud Source Repository

## Project Approach:
 - 1. Create the repository in github in local
 - 2. Create a Flask app (app.py)
    - In this file we have created various modules\features on the webpage as given below:
        - Home Page - It briefly describes the Project
        - Project Description - This provides the techniques used for the modelling
        - Single User Prediction - It will accept the output from webpage for single user and predict whether application can be accepted or rejected based on the saved models
        - Multi User Prediction - It will accept a CSV file (with more than one user) and provide the Loan Status as additional column in the weboage it self. The same data is also saved in the server.
        - Model Training - If we have a different set of data on which we want to train our model, we can copy the files in the input folder in bit bucket and Train the model from this page.
 - 3. SingleUserLoanEligibilityPredictor module will serve the single user pipeline from the webpage.
 - 4. MultiUserLoanEligibilityPredictor module will serve the multiple user pipeline from the webpage.
 - 5. ModelTrainingEngine module will serve the model training on new data from the webpage.
 - 6. ML_Pipeline module is used for DataPreprocessing, Encoding and Outlier Treatment.