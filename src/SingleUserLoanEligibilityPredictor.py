import pandas as pd
import MultiUserLoanEligibilityPredictor


def predictor(input_dict):
    try:
        LoanID = str(input_dict['LoanID'])
        CustomerId = str(input_dict['CustomerId'])
        CurrentLoanAmount = float(input_dict['CurrentLoanAmount'])
        Term = str(input_dict['Term'])
        HomeOwnership = str(input_dict['HomeOwnership'])
        Purpose = str(input_dict['Purpose'])
        CreditScore = float(input_dict['CreditScore'])
        Yearsincurrentjob = float(input_dict['Yearsincurrentjob'])
        AnnualIncome = float(input_dict['AnnualIncome'])
        MonthlyDebt = float(input_dict['MonthlyDebt'])
        YearsofCreditHistory = int(input_dict['YearsofCreditHistory'])
        Monthssincelastdelinquent = int(input_dict['Monthssincelastdelinquent'])
        NumberofOpenAccounts = int(input_dict['NumberofOpenAccounts'])
        NumberofCreditProblems = int(input_dict['NumberofCreditProblems'])
        CurrentCreditBalance = float(input_dict['CurrentCreditBalance'])
        MaximumOpenCredit = int(input_dict['MaximumOpenCredit'])
        Bankruptcies = int(input_dict['Bankruptcies'])
        TaxLiens = float(input_dict['TaxLiens'])

        df = [[ LoanID, CustomerId, CurrentLoanAmount, Term,  CreditScore, Yearsincurrentjob, HomeOwnership, AnnualIncome, Purpose, MonthlyDebt, YearsofCreditHistory, 
               Monthssincelastdelinquent, NumberofOpenAccounts, NumberofCreditProblems, CurrentCreditBalance, MaximumOpenCredit, Bankruptcies, TaxLiens ]]
        columns_list = ['Loan ID', 'Customer ID', 'Current Loan Amount', 'Term',  'Credit Score','Years in current job', 'Home Ownership', 'Annual Income', 'Purpose',
                        'Monthly Debt', 'Years of Credit History','Months since last delinquent','Number of Open Accounts','Number of Credit Problems','Current Credit Balance',
                        'Maximum Open Credit', 'Bankruptcies','Tax Liens']
        test_df = pd.DataFrame(df, columns=columns_list)
        
        # Calling MultiUserLoanEligibilityPredictor for prediction
        predicted_value = MultiUserLoanEligibilityPredictor.predictor(test_df, singleuser=True)

    except Exception as e:
        print('Exception in SingleUserLoanEligibilityPredictor.predictor', e)
    else:
        return predicted_value
