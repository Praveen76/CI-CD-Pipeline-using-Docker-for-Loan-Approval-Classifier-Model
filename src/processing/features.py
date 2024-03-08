from typing import List
import sys
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class MissValImputer(BaseEstimator, TransformerMixin):
    """Custom imputer for handling missing values in numerical and categorical variables."""
    print("================== MissValImputer started for numerical variables==================")
    def __init__(self, num_vars: list, cat_vars: list):
        if not all(isinstance(var, str) for var in num_vars + cat_vars):
            raise ValueError("Variables should be a list of strings.")

        self.num_vars = num_vars
        self.cat_vars = cat_vars

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        # Store mean and standard deviation for numerical variables
        self.numerical_fill_values = {}

        for var in self.num_vars:
            mean_value = X[var].mean()
            std_value = X[var].std()
            self.numerical_fill_values[var] = (mean_value, std_value)

        # Store mode value for categorical variables
        self.categorical_fill_values = {}

        for var in self.cat_vars:
            mode_value = X[var].mode().values[0]
            self.categorical_fill_values[var] = mode_value

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        np.random.seed(0)

        # Impute missing values for numerical variables with mean
        for var in self.num_vars:
            mean_value, std_value = self.numerical_fill_values[var]

            # Impute missing values using a random distribution
            null_count = X[var].isnull().sum()
            null_random_list = np.random.randint(mean_value - std_value, mean_value + std_value, size=null_count)
            X.loc[np.isnan(X[var]), var] = null_random_list
            X[var] = X[var].astype(X[var].dtype)  # Ensure the correct data type

        # Impute missing values for categorical variables with mode
        for var in self.cat_vars:
            mode_value = self.categorical_fill_values[var]
            X[var].fillna(mode_value, inplace=True)

        return X


class OutlierHandler:
    print("================== OutlierHandler started for numerical variables==================")
    def __init__(self, num_vars: list):
        self.num_vars = num_vars
        self.outlier_info = {}

    def identify_outliers(self, df, clmn):
        print('DataFrame columns:', df.columns)
        print('Outliers check for feature:', clmn)
        # Outliers check
        Q1 = df[clmn].quantile(0.25)
        Q3 = df[clmn].quantile(0.75)
        IQR = Q3 - Q1
        Lower_Whisker = Q1 - (1.5 * IQR)
        Upper_Whisker = Q3 + (1.5 * IQR)
        # Store outlier information
        self.outlier_info[clmn] = {'Lower_Whisker': Lower_Whisker, 'Upper_Whisker': Upper_Whisker}

    def treat_outliers(self, df, clmn, method='mean'):
        if method == 'mean':
            mean_value = round(df[clmn].mean(), 2)
            df[clmn] = df[clmn].apply(lambda x: mean_value if
                                      (x < self.outlier_info[clmn]['Lower_Whisker'] or x > self.outlier_info[clmn]['Upper_Whisker'])
                                      else x)

    def fit(self, X, y=None):
        # Identify and store outlier information during fit
        for col in self.num_vars:
            self.identify_outliers(X, col)

        return self

    def transform(self, X):
        # Treat outliers during transform
        X = X.copy()
        for col in self.num_vars:
            self.treat_outliers(X, col)
        print("Outliers treatment is done")
        return X

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    print("================== Encoding started for categorical variables==================")
    def __init__(self, cat_vars: list):
        self.cat_vars = cat_vars
        self.encoder = None

    def fit(self, X, y=None):
        # One-Hot Encoding
        encoder = OneHotEncoder(drop='first', sparse=False)
        self.encoder = encoder.fit(X[self.cat_vars])

        return self

    def transform(self, X):
        if len(self.cat_vars) == 0:
            return X

        # Prefix encoded values with column names
        OHE_df = pd.DataFrame(self.encoder.transform(X[self.cat_vars]),
                              columns=self.encoder.get_feature_names_out(self.cat_vars))

        num_df = X.drop(self.cat_vars, axis=1).reset_index(drop=True)

        df_encoded = pd.concat([OHE_df, num_df], axis=1)

        return df_encoded
