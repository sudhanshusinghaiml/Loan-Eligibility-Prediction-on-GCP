from flask_cors import cross_origin
from flask import Flask, redirect, render_template, request, url_for
import pandas as pd
import numpy as np
from ModelTrainingEngine import model_training_pipeline
from SingleUserLoanEligibilityPredictor import predictor
from ML_Pipeline.DataPreProcessing import DataPreProcessing
from ML_Pipeline.ImputeNumericalValues import ImputeNumericalValues
from ML_Pipeline.OutlierTreatment import OutlierTreatment
from ML_Pipeline.FeatureEncoder import FeatureEncoder
import MultiUserLoanEligibilityPredictor

# To supress future warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=FutureWarning)

# Importing for logging purpose
import logging
from log_config import configure_logger
# Configure logger
configure_logger()
# Get logger
logger = logging.getLogger(__name__)

import os
import joblib
import six
import sys
# Setting up the environment for sklearn
sys.modules['sklearn.externals.six'] = six
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn.utils import _safe_indexing
sys.modules['sklearn.utils.safe_indexing'] = sklearn.utils._safe_indexing


# 1. Creating LoanEligibility Flask app
BankingLoanEligibilityapp = Flask(__name__)

UPLOAD_FOLDER = 'output'
BankingLoanEligibilityapp.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# 2. Index route, opens automatically on http://127.0.0.1:8000
@BankingLoanEligibilityapp.route('/')
def index():
    heading = 'Banking Loan Eligibility Prediction App'
    logger.debug('This is inside index() function.')
    return render_template("home.html", heading=heading)


# 3. Routing to Home Page of BankingLoanEligibilityapp
@BankingLoanEligibilityapp.route('/home.html')
def back_to_index():
    heading = 'Banking Loan Eligibility Prediction App'
    logger.debug('This is inside back_to_index() function.')
    return render_template("home.html", heading=heading)


# 4. Routing to Page that displays Project Descriptions
@BankingLoanEligibilityapp.route('/loanEligibilityProject.html')
@cross_origin()
def project_home():
    heading = 'Banking Loan Eligibility Prediction App'
    logger.debug('This is inside project_home() function.')
    return render_template('/loanEligibilityProject.html', heading=heading)


@BankingLoanEligibilityapp.route('/multiUserLoanEligibilityPrediction.html')
@cross_origin()
def upload_file():
    logger.debug('This is inside upload_file() function.')
    return render_template('/multiUserLoanEligibilityPrediction.html')


@BankingLoanEligibilityapp.route('/upload_processing', methods=['POST'])
def upload_processing():
    status = ' '
    logger.debug('Inside upload_processing function')
    if request.method == "POST":
        file = request.files['csvfile']
        if file:
            logger.debug('Detected test data file inside upload_processing function')
            filename = os.path.join(BankingLoanEligibilityapp.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            
            logger.info(f'The file {file.filename} is saved in {filename}')
            # Reading the uploaded CSV file in datframe
            df = pd.read_csv(filename)
            
            logger.debug('Calling MultiUserLoanEligibilityPredictor')
            # Calling the Loan Eligibility Prediction from the model
            predicted_df = MultiUserLoanEligibilityPredictor.predictor(df, False)

            logger.debug('Completed MultiUserLoanEligibilityPredictor')            
            # Save the updated DataFrame to a new CSV file
            processed_filename = os.path.join(BankingLoanEligibilityapp.config['UPLOAD_FOLDER'], 'processed_' + file.filename)
            predicted_df.to_csv(processed_filename, index=False)
            
            logger.info(f'Predicted data is saved in - {processed_filename}')
            
            file_name = 'processed_' + file.filename
            
            logger.debug('Redirected to the site to - "/display_multi_user_results/<filename>"')

            return redirect(url_for('display_multi_user_results', filename=file_name))

    logger.debug('Redirecting to the site - "/upload_processing" ')
    return redirect(url_for('upload_file'))


# 5. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return whether given customers is eligible for loan 
#    (http://127.0.0.1:8000/multiUserLoanEligibilityPredictionDisplay)
@BankingLoanEligibilityapp.route('/display_multi_user_results/<filename>')
def display_multi_user_results(filename):
    logger.debug('Inside display_multi_user_results function')
    df = pd.read_csv(os.path.join(BankingLoanEligibilityapp.config['UPLOAD_FOLDER'],filename))
    data = df.to_dict(orient='records')
    logger.debug('Displaying the static csv data to site')
    return render_template('/multiUserLoanEligibilityPredictionDisplay.html', data=data)

    
# 6. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return whether given customers is eligible for loan 
#    (http://127.0.0.1:8000/singleUserLoanEligibilityPrediction)
@BankingLoanEligibilityapp.route('/singleUserLoanEligibilityPrediction.html', methods=['GET', 'POST'])
@cross_origin()
def single_user_loan_eligibility_predictor():
    logger.debug('Inside single_user_loan_eligibility_predictor function')
    status = ' '
    if request.method == "POST":
        logger.debug('POST - Inside single_user_loan_eligibility_predictor function')
        to_predict_dict = request.form.to_dict()
        logger.debug('Calling SingleUserLoanEligibilityPredictor.predictor function')
        flag = predictor(to_predict_dict)
        logger.debug('The prediction for SingleUserLoanEligibilityPredictor is completed')
        if flag:
            status = "Loan Application Approved."
        else:
            status = "Loan Application is rejected."

    return render_template("/singleUserLoanEligibilityPrediction.html", prediction_text=status)

# 7. Expose the Model Training functionality with new set of Data
#    (http://127.0.0.1:8000/loanEligibilityModelTraining)
@BankingLoanEligibilityapp.route('/loanEligibilityModelTraining.html', methods=['GET', 'POST'])
@cross_origin()
def model_training():
    logger.debug('Inside model_training function')
    status = ' '
    if request.method == "POST":
        model_training_flag = float(request.form["RetrainModel"])
        if model_training_flag == 1:
            logger.debug('Calling ModelTrainingEngine.model_training_pipeline')
            flag = model_training_pipeline()
            if flag:
                status = "Model Training Completed"
                logger.debug('Model Training Completed')
            else:
                status = "Model Training completed with some errors"
                logger.debug('Model Training Completed with errors')
        else:
            status = "Model Training aborted"
            logger.debug('Model Training aborted')

    return render_template("/loanEligibilityModelTraining.html", model_training_status=status)


# 8. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    BankingLoanEligibilityapp.run(debug=True)
    BankingLoanEligibilityapp.config['TEMPLATES_AUTO_RELOAD'] = True