import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

from ML_Pipeline.DataPreProcessing import DataPreProcessing
from ML_Pipeline.FeatureEncoder import FeatureEncoder
from ML_Pipeline.ImputeNumericalValues import  ImputeNumericalValues
from ML_Pipeline.OutlierTreatment import OutlierTreatment

# Importing for logging purpose
import logging
from log_config import configure_logger
# Configure logger
configure_logger()
# Get logger
logger = logging.getLogger(__name__)


def model_training_pipeline():
    """
        Calls CategoricalEncoder.py function for data cleaning and encoding.
        Calls AddFeatures.py to add more from other features in the dataset.
        Calls CustomScaler.py methods to train the data on Distance based ML Algorithms

        Returns
        -------
        True - on Successful execution of function
    """
    try:
        df = pd.read_csv('input/LoansTrainingSetV2.csv')

        columns_to_be_dropped = ['Loan ID', 'Customer ID']
        
        numeric_columns_to_be_imputed = ['Credit Score', 'Years in current job', 'Annual Income', 'Monthly Debt', 'Months since last delinquent', 
                                 'Maximum Open Credit','Bankruptcies', 'Tax Liens']
        
        columns_to_be_encoded = ['Term', 'Home Ownership', 'Purpose']
        
        columns_for_outlier_treatment = ['Current Loan Amount', 'Credit Score', 'Annual Income', 'Monthly Debt', 'Current Credit Balance', 
                                 'Maximum Open Credit']
        
        # categorical_columns_to_be_imputed = ['Loan Status', 'Term', 'Home Ownership', 'Purpose']

        # Applying Data PreProcessing on the input data
        logger.debug(f'Applying DataPreProcessing on the input data')
        preprocessor = DataPreProcessing(drop_columns=columns_to_be_dropped, 
                                 clean_numerical_columns= numeric_columns_to_be_imputed,
                                 clean_categorical_columns=columns_to_be_encoded
                                )
        preprocessed_df = preprocessor.fit_transform(df)

        # Converting the columns to numeric
        logger.debug(f'Applying the columns conversion to numeric')
        for col in numeric_columns_to_be_imputed:
            preprocessed_df[col] = pd.to_numeric(preprocessed_df[col], errors='coerce')

        for col in columns_for_outlier_treatment:
            preprocessed_df[col] = pd.to_numeric(preprocessed_df[col], errors='coerce')


        # Applying Imputation to preprocessed data
        logger.debug(f'Applying Imputation to preprocessed data')
        imputer = ImputeNumericalValues()
        imputer_output_df = imputer.fit_transform(preprocessed_df, columns= numeric_columns_to_be_imputed)
        imputer_output_df['Loan Status'] = imputer_output_df['Loan Status'].replace({'Loan Refused': 0, 'Loan Given':1})

        # Storing the data into X & Y
        X = imputer_output_df.iloc[:,1:]
        Y = imputer_output_df.iloc[:,0:1]

        # Applying Outlier Treatment on the imputed data
        logger.debug(f'Applying Outlier Treatment on the imputed data')
        outlierprocessor = OutlierTreatment()
        outlier_treatment_df = outlierprocessor.fit_transform(X, columns=columns_for_outlier_treatment)

        # Applying Encoding for the Categorical Columns
        logger.debug(f'Applying Encoding for the Categorical Columns')
        encoder = FeatureEncoder(label_encoder_cols=columns_to_be_encoded)
        encoded_df = encoder.fit_transform(outlier_treatment_df)

        # Saving the DataPreProcessing, ImputeNumericalValues, OutlierTreatment and FeatureEncoder Objects
        logger.debug(f'Saving the DataPreProcessing, ImputeNumericalValues, OutlierTreatment and FeatureEncoder Objects')
        
        joblib.dump(preprocessor, 'output/dataPreProcessing.pkl')
        logger.debug(f'Saved the object for DataPreProcessing')

        joblib.dump(imputer, 'output/imputeNumericalValues.pkl')
        logger.debug(f'Saved the object for ImputeNumericalValues')

        joblib.dump(outlierprocessor, 'output/outlierTreatment.pkl')
        logger.debug(f'Saved the object for outlierTreatment')

        joblib.dump(encoder, 'output/categoricalEncoding.pkl')
        logger.debug(f'Saved the object for categoricalEncoding')


        #Splitting the data for training and test
        logger.debug(f'Splitting the data for training and test')
        X_train, X_test, y_train, y_test = train_test_split(encoded_df, Y, test_size=0.3, random_state=1234, 
                                                            stratify=Y['Loan Status'])

        logger.debug(f'X_train.shape -{X_train.shape}')
        logger.debug(f'y_train.shape- {y_train.shape}')
        logger.debug(f'X_test.shape- {X_test.shape}')
        logger.debug(f'y_test.shape- {y_test.shape}')

        # Applying Extreme Gradient Boosting Classifier using Threshold
        logger.debug(f'Applying Extreme Gradient Boosting Classifier using Threshold')
        xgb_threshold_model = XGBClassifier(random_state=123)
        xgb_threshold_model.fit(X_train, y_train)

        y_xgb_train_prob_predicted = xgb_threshold_model.predict_proba(X_train)
        y_xgb_test_prob_predicted = xgb_threshold_model.predict_proba(X_test)

        y_xgb_train_predicted = [1 if i >= 0.20 else 0 for i in y_xgb_train_prob_predicted[:,1]]
        y_xgb_test_predicted =  [1 if i >= 0.20 else 0 for i in y_xgb_test_prob_predicted[:,1]]

        logger.info(f'Training Data Classification report -\n{classification_report(y_train, y_xgb_train_predicted)}')
        logger.info(f'Test Data Classification report -\n{classification_report(y_test, y_xgb_test_predicted)}')

        # Saving the model
        joblib.dump(xgb_threshold_model,'output/xgb_threshold_model.pkl')
        logger.debug(f'Saved the xgb_threshold_model')

    except Exception as e:
        logger.debug(f'Error in ModelTrainingEngine.model_training_pipeline {e}')
    else:
        return True
