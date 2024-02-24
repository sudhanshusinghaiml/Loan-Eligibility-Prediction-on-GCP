from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd

class ImputeCategoricalValues(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mv_mode = None

    def fit(self, X, columns=None):
        
        self.mv_mode = {}
    
        Xo = X.copy()
        if columns is None:
            return self  # No fitting is applied
        
        # getting the lower nd upper limit range for 
        # outlier treatment
        for col in columns:
            self.mv_mode[col] = Xo[col].mode()[0]
            
        return self

    def transform(self, X):
               
        X_transformed = X.copy()  # Create a copy of the original data
        for idx, mode in self.mv_mode.items():
            # Apply transformation using lower limit and upper limit
            X_transformed[idx] = X_transformed[idx].fillna(mode)
        return X_transformed


    def fit_transform(self, X, columns=None):
        # Fit the preprocessor and transform the data
        return self.fit(X, columns).transform(X)