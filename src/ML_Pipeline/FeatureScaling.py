from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np


class FeatureScaling(BaseEstimator, TransformerMixin):
    """
    A custom standard scaler class with the ability to apply scaling on selected columns
    """

    def __init__(self, scaling_cols=None):
        """
        Parameters
        ----------
        scaling_cols : list of str
            Columns on which to perform scaling and normalization. Default is to scale all numerical columns

        """
        self.scaling_maps = None
        self.scaling_cols = scaling_cols

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to scale
        """

        # Scaling all non-categorical columns if user doesn't provide the list of columns to scale
        if self.scaling_cols is None:
            self.scaling_cols = [c for c in X if
                                 ((str(X[c].dtype).find('float') != -1) or (str(X[c].dtype).find('int') != -1))]

        # Create mapping corresponding to scaling and normalization
        self.scaling_maps = dict()
        for col in self.scaling_cols:
            self.scaling_maps[col] = dict()
            self.scaling_maps[col]['mean'] = np.mean(X[col].values).round(2)
            self.scaling_maps[col]['std_dev'] = np.std(X[col].values).round(2)

        # Return fit object
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to scale
        """
        Xo = X.copy()

        # Map transformation to respective columns
        for col in self.scaling_cols:
            Xo[col] = (Xo[col] - self.scaling_maps[col]['mean']) / self.scaling_maps[col]['std_dev']

        # Return scaled and normalized DataFrame
        return Xo

    def fit_transform(self, X, y=None):
        """
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to scale
        """
        # Fit and return transformed dataframe
        return self.fit(X).transform(X)