from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd

class ImputeNumericalValues(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mv_median = None

    def fit(self, X, columns=None):
        Xo = X.copy()
        
        self.mv_median = {}
        
        if columns is None:
            return self  # No fitting is applied
        
        # getting the lower nd upper limit range for 
        # outlier treatment
        for col in columns:
            self.mv_median[col] = Xo[col].median()
            
        return self

    def transform(self, X):
               
        X_transformed = X.copy()  # Create a copy of the original data
        for idx, median in self.mv_median.items():
            # Apply transformation using lower limit and upper limit
            X_transformed[idx] = X_transformed[idx].fillna(median)
        return X_transformed


    def fit_transform(self, X, columns=None):
        # Fit the preprocessor and transform the data
        return self.fit(X, columns).transform(X)
