from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd

class OutlierTreatment(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.lower_limits = {}
        self.upper_limits = {}

    def fit(self, X, columns=None):
        if columns is None:
            return self  # No fitting is applied
        
        # getting the lower nd upper limit range for 
        # outlier treatment
        for col in columns:
            data = pd.Series(sorted(X[col]))
            Q1 = data.min()
            Q3 = data.quantile(0.90)
            IQR = Q3 - Q1
            llimit = Q1 - (1.5 * IQR)
            hlimit = Q3 + (1.5 * IQR)
            print('Columns -', col)
            self.lower_limits[col] = llimit
            self.upper_limits[col] = hlimit
            print('self.lower_limit -', self.lower_limits[col])
            print('self.upper_limit -', self.upper_limits[col])
            
        return self

    def transform(self, X):
               
        X_transformed = X.copy()  # Create a copy of the original data
        
        # Apply outlier treatment for each specified column
        for col in self.lower_limits.keys():
            
            ll = self.lower_limits[col]
            ul = self.upper_limits[col]
            
            # Apply transformation using lower limit and upper limit
            X_transformed[col] = np.where(X_transformed[col] > ul, ul, X_transformed[col])
            X_transformed[col] = np.where(X_transformed[col] < ll, ll, X_transformed[col])
        return X_transformed


    def fit_transform(self, X, columns=None):
        # Fit the preprocessor and transform the data
        return self.fit(X, columns).transform(X)