from flask_cors import cross_origin
from flask import Flask, redirect, render_template, request, url_for
import pandas as pd
import numpy as np

from SingleUserLoanEligibilityPredictor import predictor
from ML_Pipeline import DataPreProcessing
from ML_Pipeline import ImputeNumericalValues
from ML_Pipeline import OutlierTreatment
from ML_Pipeline import FeatureEncoder
from MultiUserLoanEligibilityPredictor import predictor as multiuser_predictor


# To supress future warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=FutureWarning)

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

@BankingLoanEligibilityapp.route('/multiUserLoanEligibilityPrediction.html')
@cross_origin()
def upload_file():
    return render_template('/multiUserLoanEligibilityPrediction.html')

@BankingLoanEligibilityapp.route('/upload_processing', methods=['POST'])
def upload_processing():
    status = ' '
    if request.method == "POST":
        file = request.files['csvfile']
        if file:
            filename = os.path.join(BankingLoanEligibilityapp.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            # Reading the uploaded CSV file in datframe
            df = pd.read_csv(filename)
            
            # Calling the Loan Eligibility Prediction from the model
            predicted_df = multiuser_predictor(df, False)
            
            # Save the updated DataFrame to a new CSV file
            processed_filename = os.path.join(BankingLoanEligibilityapp.config['UPLOAD_FOLDER'], 'processed_' + file.filename)
            predicted_df.to_csv(processed_filename, index=False)

            return redirect(url_for('display_multi_user_results', filename=processed_filename))
        
    return redirect(url_for('upload_file'))

# 5. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return whether given customers is eligible for loan 
#    (http://127.0.0.1:8000/multi-user-loan-eligibility-prediction)
@BankingLoanEligibilityapp.route('/display_multi_user_results/<filename>')
def display_multi_user_results(filename):
    df = pd.read_csv(filename)
    data = df.to_dict(orient='records')
    return render_template('/multiUserLoanEligibilityPredictionDisplay.html', data=data)

    
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