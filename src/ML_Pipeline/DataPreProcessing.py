from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
import re

class DataPreProcessing(BaseEstimator, TransformerMixin):
    """
    PreProcess the data. It includes processes such as:
    Missing Values - Updating missing values for training data
    Outlier Treatment - It will treate outliers in the manner as it does for training data
    Feature Update - It Updates feature values with unique values so that data encoding can be done easily

    """

    def __init__(self, drop_columns=None, clean_numerical_columns=None, clean_categorical_columns=None):
        """
        Parameters
        ----------
        cols : list of str
        drop_columns:                 Columns name that are not needed for our models such as customer id, loan id
        clean_numerical_columns:      Columns name that should have only numeric values
        clean_categorical_columns:    Columns name that should have only categorical values
        """

        if isinstance(drop_columns, str):
            self.drop_columns = [drop_columns]
        else:
            self.drop_columns = drop_columns

        if isinstance(clean_numerical_columns, str):
            self.clean_numerical_columns = [clean_numerical_columns]
        else:
            self.clean_numerical_columns = clean_numerical_columns

        if isinstance(clean_categorical_columns, str):
            self.clean_categorical_columns = [clean_categorical_columns]
        else:
            self.clean_categorical_columns = clean_categorical_columns


    def fit(self, X, columns=None):
        """Fit to X and y

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values.

        Returns
        -------
        self : encoder
            Returns self.
        """
        
        # Return the fit object
        return self
    
    def transform(self, X, columns=None):
        """Perform Data Cleaning.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to label encode

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """
        
        # Xo = pd.DataFrame(X)
        Xo = X.copy()
        
        # Dropping the columns that are not needed for our models
        for col in self.drop_columns:
            Xo.drop(columns = col, inplace=True)
        
        # Remove special characters from specified columns
        if self.clean_numerical_columns:
            for col in self.clean_numerical_columns:
               
                Xo[col] = Xo[col].astype(str)
                Xo[col] = Xo[col].apply(lambda x: re.sub(r'[^0-9\s]', '', x))
                Xo[col] = Xo[col].str.strip()
                Xo[col] = Xo[col].replace('#VALUE!', np.nan, regex=True)
        
        # Replace categories in specified columns
        if self.clean_categorical_columns:
            for col in self.clean_categorical_columns:
                if col == 'Purpose':
                    Xo[col] = Xo[col].str.replace('other', 'Other', regex=True)
                elif col == 'Home Ownership':
                    Xo[col] = Xo[col].str.replace('HaveMortgage', 'Home Mortgage', regex=True)

        # Return encoded DataFrame
        return Xo

    def fit_transform(self, X, columns=None):
        """Fit and transform the data via label/one-hot/target encoding.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """

        return self.fit(X).transform(X)