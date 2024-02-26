import pandas as pd
import MultiUserLoanEligibilityPredictor
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

def predictor(input_dict):
    try:
        LoanID = str(input_dict['LoanID'])
        CustomerId = str(input_dict['CustomerId'])
        CurrentLoanAmount = int(input_dict['CurrentLoanAmount'])
        Term = str(input_dict['Term'])
        HomeOwnership = str(input_dict['HomeOwnership'])
        Purpose = str(input_dict['Purpose'])
        CreditScore = float(input_dict['CreditScore'])
        Yearsincurrentjob = float(input_dict['Yearsincurrentjob'])
        AnnualIncome = float(input_dict['AnnualIncome'])
        MonthlyDebt = float(input_dict['MonthlyDebt'])
        YearsofCreditHistory = float(input_dict['YearsofCreditHistory'])
        Monthssincelastdelinquent = float(input_dict['Monthssincelastdelinquent'])
        NumberofOpenAccounts = int(input_dict['NumberofOpenAccounts'])
        NumberofCreditProblems = int(input_dict['NumberofCreditProblems'])
        CurrentCreditBalance = int(input_dict['CurrentCreditBalance'])
        MaximumOpenCredit = int(input_dict['MaximumOpenCredit'])
        Bankruptcies = float(input_dict['Bankruptcies'])
        TaxLiens = float(input_dict['TaxLiens'])

        df = [[ LoanID, CustomerId, CurrentLoanAmount, Term,  CreditScore, Yearsincurrentjob, HomeOwnership, AnnualIncome, Purpose, MonthlyDebt, YearsofCreditHistory, 
               Monthssincelastdelinquent, NumberofOpenAccounts, NumberofCreditProblems, CurrentCreditBalance, MaximumOpenCredit, Bankruptcies, TaxLiens ]]
        columns_list = ['Loan ID', 'Customer ID', 'Current Loan Amount', 'Term',  'Credit Score','Years in current job', 'Home Ownership', 'Annual Income', 'Purpose',
                        'Monthly Debt', 'Years of Credit History','Months since last delinquent','Number of Open Accounts','Number of Credit Problems','Current Credit Balance',
                        'Maximum Open Credit', 'Bankruptcies','Tax Liens']
        test_df = pd.DataFrame(df, columns=columns_list)

        logger.debug('Created DataFrame for the given Input')
        
        logger.debug('Calling MultiUserLoanEligibilityPredictor.predictor function for prediction on the data')
        # Calling MultiUserLoanEligibilityPredictor for prediction
        predicted_value = MultiUserLoanEligibilityPredictor.predictor(test_df, singleuser=True)

        logger.debug('Successfully executed MultiUserLoanEligibilityPredictor.predictor function for prediction on the data')

    except Exception as e:
        logger.debug(f'Exception in SingleUserLoanEligibilityPredictor.predictor {e} ')
    else:
        return predicted_value
