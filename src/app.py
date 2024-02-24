import MultiUserLoanEligibilityPredictor
from SingleUserLoanEligibilityPredictor import predictor
from ML_Pipeline.DataPreProcessing import DataPreProcessing
from ML_Pipeline.ImputeNumericalValues import ImputeNumericalValues
from ML_Pipeline.OutlierTreatment import OutlierTreatment
from ML_Pipeline.FeatureEncoder import FeatureEncoder

from flask_cors import cross_origin
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

# To supress future warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=FutureWarning)

import os
import joblib
import json
import traceback
import operator
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


# 2. Index route, opens automatically on http://127.0.0.1:8000
@BankingLoanEligibilityapp.route('/')
def index():
    heading = 'Banking Loan Eligibility Prediction App'
    return render_template("home.html", heading=heading)


# 3. Routing to Home Page of BankingLoanEligibilityapp
@BankingLoanEligibilityapp.route('/home.html')
def back_to_index():
    heading = 'Banking Loan Eligibility Prediction App'
    return render_template("home.html", heading=heading)


# 4. Routing to Page that displays Project Descriptions
@BankingLoanEligibilityapp.route('/loanEligibilityProject.html')
@cross_origin()
def project_home():
    heading = 'Banking Loan Eligibility Prediction App'
    return render_template('/loanEligibilityProject.html', heading=heading)

@BankingLoanEligibilityapp.route('/UploadMultiUserData') # , methods=['GET', 'POST'])
@cross_origin()
def upload_multi_user_data_for_predicton():
    status = ' '
    if request.method == "POST":
        to_predict_dict = request.form.to_dict()
        flag = predictor(to_predict_dict)
        if flag:
            status = "Loan Application can be accepted."
        else:
            status = "Loan Application is rejected."

    return render_template("/multiUserLoanEligibilityPrediction.html", model_training_status=status)


# 5. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return whether given customers is eligible for loan 
#    (http://127.0.0.1:8000/multi-user-loan-eligibility-prediction)
@BankingLoanEligibilityapp.route('/multiUserLoanEligibilityPredictionDisplay.html', methods=['GET', 'POST'])
def multi_user_loan_eligibility_predictor():
    if request.method == "POST":
        try:
            if 'csvfile' not in request.files:
                return 'No file has been uploaded', 400
            
            file = request.files['csvfile']
            if file.filename == '':
                return 'Invalid file was provided', 400
            
            if file:
                # Save the uploaded CSV file
                file.save(file.filename)
                
                # Process the CSV file
                df = pd.read_csv(file.filename)
                
                # Get the Loan Eligibility Prediction from the model
                predicted_data = MultiUserLoanEligibilityPredictor.predictor(df)
                
                # Save the modified data to a new CSV file
                output_filename = 'MultiUserLoanEligibilityPrediction.csv'
                predicted_data.to_csv(output_filename, index=False)
                
            else:
                return 'File upload failed', 400
                
        except Exception as e: # Exception for system errors
            print('Exception in multi_user_loan_eligibility_predictor', e)
        
    # Render the HTML template with the modified data
    # return render_template("/multiUserLoanEligibilityPrediction.html", data=predicted_data.to_html())
    if predicted_data is not None:
        return render_template("/multiUserLoanEligibilityPredictionDisplay.html", data=predicted_data.to_html())
    else:
        return render_template("/multiUserLoanEligibilityPrediction.html")

    
# 6. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return whether given customers is eligible for loan 
#    (http://127.0.0.1:8000/single-user-loan-eligibility-prediction)
@BankingLoanEligibilityapp.route('/singleUserLoanEligibilityPrediction.html', methods=['GET', 'POST'])
@cross_origin()
def single_user_loan_eligibility_predictor():
    status = ' '
    if request.method == "POST":
        to_predict_dict = request.form.to_dict()
        flag = predictor(to_predict_dict)
        if flag:
            status = "Loan Application Approved."
        else:
            status = "Loan Application is rejected."

    return render_template("/singleUserLoanEligibilityPrediction.html", prediction_text=status)

    
# 7. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    BankingLoanEligibilityapp.run(debug=True)
    BankingLoanEligibilityapp.config['TEMPLATES_AUTO_RELOAD'] = True