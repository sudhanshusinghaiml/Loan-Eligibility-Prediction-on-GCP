import joblib
import os
import numpy as np
import pandas as pd
from ML_Pipeline import DataPreProcessing
from ML_Pipeline import ImputeNumericalValues
from ML_Pipeline import OutlierTreatment
from ML_Pipeline import FeatureEncoder

# Importing for logging purpose
import logging
from log_config import configure_logger
# Configure logger
configure_logger()
# Get logger
logger = logging.getLogger(__name__)

UPLOAD_FOLDER = 'output'

def predictor(test_df, singleuser=False):
    try:
        logger.debug('In the MultiUserLoanEligibilityPredictor.predictor')
        # Loading datapreProcessor to disk for preprocessing the data
        preprocessor_filename = os.path.join(UPLOAD_FOLDER, 'dataPreProcessing.pkl')
        preprocessor = joblib.load(preprocessor_filename)
        logger.debug('Loaded the preprocessor - ', preprocessor_filename)

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
        imputer_filename = os.path.join(UPLOAD_FOLDER, 'imputeNumericalValues.pkl')
        imputer = joblib.load(imputer_filename)
        logger.debug('Loaded the imputer -', imputer_filename)
        
        # Imputing the missing values in the dataset for prediction
        imputer_output_df = imputer.transform(preprocessed_df)
        
        # Loading outlierTreatment to disk for imputing the outlier values
        outlierProcessor_filename = os.path.join(UPLOAD_FOLDER, 'outlierTreatment.pkl')
        outlierProcessor = joblib.load(outlierProcessor_filename)
        logger.debug('Loaded the outlierProcessor - ',outlierProcessor_filename)

        # Updating outlier values in the test dataset
        outlier_treatment_df = outlierProcessor.transform(imputer_output_df)

        # Loading categoricalEncoding to disk for Encoding the categorical values
        encoder_filename = os.path.join(UPLOAD_FOLDER, 'categoricalEncoding.pkl')
        encoder = joblib.load(encoder_filename)
        logger.debug('Loaded the encoder - ', encoder_filename)

        # Encoding the test data so that it can be utilized for inference
        encoded_df = encoder.transform(outlier_treatment_df)
        
        # Loading XGB model to disk for prediction
        model_filename = os.path.join(UPLOAD_FOLDER, 'xgb_threshold_model.pkl')
        model = joblib.load(model_filename)
        logger.debug('Loaded the model -', model_filename)

        # Predicting the target variable using trained and loaded model
        test_prediction = model.predict(encoded_df)
        
        if not singleuser:
        
            output_df = test_df.copy()
            
            output_df['Loan Status'] = test_prediction
            
            output_df['Loan Status'] = output_df['Loan Status'].replace({1: 'Loan Approved', 0: 'Loan Rejected'})
        else:
            return test_prediction[0]
        
    except Exception as e:
        logger.debug('Exception in MultiUserLoanEligibilityPrediction.predictor', e)
    else:
        return output_df