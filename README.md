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
 - Create the repository in github in local
 - **Application Code**
    - Flask app (app.py) is created various modules\features on the webpage as given below:
        - Home Page - It briefly describes the Project
        - Project Description - This provides the techniques used for the modelling
        - Single User Prediction - It will accept the output from webpage for single user and predict whether application can be accepted or rejected based on the saved models
        - Multi User Prediction - It will accept a CSV file (with more than one user) and provide the Loan Status as additional column in the weboage it self. The same data is also saved in the server.
        - Model Training - If we have a different set of data on which we want to train our model, we can copy the files in the input folder in bit bucket and Train the model from this page.
    - SingleUserLoanEligibilityPredictor module will serve the single user pipeline from the webpage.
    - MultiUserLoanEligibilityPredictor module will serve the multiple user pipeline from the webpage.
    - ModelTrainingEngine module will serve the model training on new data from the webpage.
    - ML_Pipeline module is used for DataPreprocessing, Encoding and Outlier Treatment.

 - **Docker**
    - **Prerequisite:** Install **Docker Desktop** and **Docker Toolbox**
    - Desktop or laptop should be Virtualization enabled
    - Dockerfile is created to run this application in the local desktop
    - Use the below commands to create Image and then run the container
    ```
    $ docker build -t loan_eligibility_app_image .
    $ docker run -p 5000:5000 loan_eligibility_app_image:latest
    ```
    - We can also run our codes in local without Docker
    ```
    python app.py
    ```