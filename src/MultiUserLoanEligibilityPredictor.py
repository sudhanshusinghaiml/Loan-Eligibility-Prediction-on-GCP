import joblib
import os
import numpy as np
import pandas as pd
from ML_Pipeline.DataPreProcessing import DataPreProcessing


def predictor(test_df, singleuser=None):
    try:
        
        # Loading datapreProcessor to disk for preprocessing the data
        filename = 'output/dataPreProcessing.pkl'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(os.getcwd())
        with open(filename, 'rb+') as f:
            preprocessor = joblib.load(f)
            
        # Performing data pre processing on test data
        preprocessed_df = preprocessor.transform(test_df)
        
        # Converting the columns to numeric for further processing - Starts
        numeric_columns_to_be_imputed = ['Credit Score', 'Years in current job', 'Annual Income', 'Monthly Debt', 'Months since last delinquent', 
                                 'Maximum Open Credit','Bankruptcies', 'Tax Liens']
                                 
        columns_for_outlier_treatment = ['Current Loan Amount', 'Credit Score', 'Annual Income', 'Monthly Debt', 'Current Credit Balance', 
                                 'Maximum Open Credit']
                                 
        for col in columns_for_outlier_treatment:
            preprocessed_df[col] = pd.to_numeric(preprocessed_df[col], errors='coerce')
            
        for col in numeric_columns_to_be_imputed:
            preprocessed_df[col] = pd.to_numeric(preprocessed_df[col], errors='coerce')
        # Converting the columns to numeric for further processing - Ends
            
            
        # Loading imputeNumericalValues to disk for imputing the values
        filename = 'output/imputeNumericalValues.pkl'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'rb+') as f:
            imputer = joblib.load(f)
        
        # Imputing the missing values in the dataset for prediction
        imputer_output_df = imputer.transform(preprocessed_df)
        
        # Loading outlierTreatment to disk for imputing the outlier values
        filename = 'output/outlierTreatment.pkl'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'rb+') as f:
            outlierProcessor = joblib.load(f)
        
        # Updating outlier values in the test dataset
        outlier_treatment_df = outlierProcessor.transform(imputer_output_df)

        # Loading categoricalEncoding to disk for Encoding the categorical values
        filename = 'output/categoricalEncoding.pkl'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'rb+') as f:
            encoder = joblib.load(f)
            
        # Encoding the test data so that it can be utilized for inference
        encoded_df = encoder.transform(outlier_treatment_df)
        
        # Loading XGB model to disk for prediction
        filename = 'output/xgb_threshold_model.pkl'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'rb+') as f:
            model = joblib.load(f)
        
        # Predicting the target variable using trained and loaded model
        test_prediction = model.predict(encoded_df)
        
        if singleuser is None:
        
            output_df = test_df.copy()
            
            output_df['Loan Status'] = test_prediction
            
            output_df['Loan Status'] = output_df['Loan Status'].replace({1: 'Loan Approved', 0: 'Loan Rejected'})
        else:
            return test_prediction[0]
        
    except Exception as e:
        print('Exception in MultiUserLoanEligibilityPrediction.predictor', e)
    else:
        return output_df
