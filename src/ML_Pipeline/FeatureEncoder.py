from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd


class FeatureEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categorical columns using LabelEncoding, OneHotEncoding and TargetEncoding.
    LabelEncoding is used for binary categorical columns
    """

    def __init__(self, cols=None, label_encoder_cols=None):
        """
        Parameters
        ----------
        cols : list of str
            Columns to encode.  Default is to one-hot/target/label encode all categorical columns in the DataFrame.
        """
        self.label_encoder_maps = None

        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols

        if isinstance(label_encoder_cols, str):
            self.label_encoder_cols = [label_encoder_cols]
        else:
            self.label_encoder_cols = label_encoder_cols

        # self.reduce_df = reduce_df

    def fit(self, X, y):
        """Fit label/one-hot/target encoder to X and y

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
        # Encode all categorical cols by default
        if self.cols is None:
            self.cols = [c for c in X if str(X[c].dtype) == 'object']

        # Check columns are in X
        for col in self.cols:
            if col not in X:
                raise ValueError('Column \'' + col + '\' not in X')

        # Separating out lcols, ohecols and tcols
        if self.label_encoder_cols is None:
            self.label_encoder_cols = [c for c in self.cols if X[c].nunique() <= 5]

        # Create Label Encoding mapping
        self.label_encoder_maps = dict()
        for col in self.label_encoder_cols:
            self.label_encoder_maps[col] = dict(zip(X[col].values, X[col].astype('category').cat.codes.values))

        # Return the fit object
        return self

    def transform(self, X, y=None):
        """Perform label/one-hot/target encoding transformation.

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
        # Perform label encoding transformation
        for col, lmap in self.label_encoder_maps.items():
            # Map the column
            # print(col)
            # print(lmap)
            # print(type(Xo))
            Xo[col] = Xo[col].map(lmap)
            Xo[col].fillna(-1, inplace=True)  # Filling new values with -1

        # Return encoded DataFrame
        return Xo

    def fit_transform(self, X, y=None):
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

        return self.fit(X, y).transform(X, y)
