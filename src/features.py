import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# We should handle missing values, encode categorical values, make extra functional transformation to features in this features.py
## (1) Missing Value Handling
    ### If there are many columns missing, we should use a list to include all columns involved then do a general missing values handling
    ### sometimes we also need to do binning before categorical features encoding
class MissingValueHandler(BaseEstimator, TransformerMixin):
    """
    Handle missing values for Titanic-like dataset. The columns are fixed while strategies might differ.

    - Age: fill with median (default) or mean, depending on `age_fill_strategy`.
    - Embarked: fill with most frequent value (mode).
    - Cabin: create an indicator column (1 if missing, else 0).

    Future updates are: (1) filling age with cluster technique (using median is not convincing)
    """
    def __init__(self, age_fill_strategy):
        self.age_fill_strategy = strategy
    def fit(self, X, y=None):
        if self.age_fill_strategy == 'median':
            self.age_fill_ = X['Age'].median()
        elif self.age_fill_strategy == 'mean':
            self.age_fill_ = X['Age'].mean()
            
        self.embarked_fill_ = X['Embarked'].mode()[0]
        return self
    def transform(self, X):
        X = X.copy()
        # Age: fill missing values with median
        X['Age'] = X['Age'].fillna(self.age_fill_)
        # Embarked: filling missing values with mode
        X['Embarked'] = X['Embarked'].fillna(self.embarked_fill_)
        # Cabin: create a missing value indicator column
        X['Cabin_missing_indicator'] = X['Cabin'].isnull().astype(int) # This is optional
        return X

## (2) categorical features encoding
    ### The reason why I don't simply use OneHotEncoding pipeline is to change features according to different columns
class CategoricalFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    One-hot encode categorical features (Sex, Pclass, Embarked).

    - Drops original categorical columns.
    - Keeps dummy variables (drop="first" to avoid dummy trap).
    - Ignores unknown categories at transform time.

    Future updates: (1) 
    """
    def fit(self, X, y=None):
        self.categorical_features = ["Sex", "Pclass", "Embarked"] # usually unchanged
        self.enc = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
        self.enc.fit(X[self.categorical_features])
        return self
    def transform(self, X):
        X = X.copy()
        cat_array = self.enc.transform(X[self.categorical_features])
        cat_df = pd.DataFrame(cat_array, 
                              columns=self.enc.get_feature_names_out(self.categorical_features),
                              index=X.index)
        X = X.drop(columns=self.categorical_features)
        X = pd.concat([X, cat_df], axis=1)
        return X

## (3) Feature Generator
    ### We could generate new features based on existing features
class FeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # We could generate as many features as we can, because we could do feature selection in the final stage
        # filling missing values have many options, if we keep all of them it would be very confusing
        X = X.copy()
        # Fare: use log(1+Fare) instead as it has a heavy tail
        X['Fare_log'] = np.log1p(X['Fare'])
        # Sibsp, Parch might be used for 'family_size' or 'is_alone'
        X['family_size'] = X['Sibsp'] + X['Parch'] + 1
        X['is_alone'] = np.where(X["family_size"] == 1, 1, 0)

        
        return X

## (4) numerical features standardlization
    ### data standardlization is necessary? for logistic model
class NumericalFeatureTransformer(BaseEstimator, TransformerMixin):
    '''

    '''
    def __init__(self, normalize):
        self.normalize = normalize
        self.scaler = StandardScaler() if normalize else None
        self.numerical_features = ["Age", "Fare", "SibSp", "Parch", "family_size", "Fare_log"]
    def fit(self, X, y=None):
        if self.normalize:
            self.scaler.fit(X[self.numerical_features])
        return self
    def transform(self, X):
        X = X.copy()
        if self.normalize:
            X[self.numerical_features] = self.scaler.transform(X[self.numerical_features])
        return X

# We have Cartesian product results for the combination of 4 classes above. We have several options
## (a) We use gridsearch (b) We merge certain combination as a big class
# v1 - baseline; v2 - improved; v3 - advanced
# small/ simple changes to functions would create a new version but #comment.
# to avoid the situation when stable version stops working because of changing. We should
## (a) do branch management ; (b) for the very stable version we first create v1_1 / v2_0. After the result show good, we could replace the original one.
# What is the relations between features.py and 02_FeatureEngineering.ipynb? Is the latter one necessary?
## I guess I'll merge 02 into 01, because it is all about exploration. And also we could explore different filling strategy, combination of features.

class FeaturePreprocessor_v1(BaseEstimator, TransformerMixin):
    def __init__(self, age_fill_strategy = 'median', normalize = True):
        self.missing = MissingValueHandler(age_fill_strategy = age_fill_strategy)
        self.cat = CategoricalFeatureTransformer()
        self.feature_gen = FeatureGenerator()
        self.num = NumericalFeatureTransformer()
        
        self.feature_names_ = None

    def fit(self, X, y=None):
        X = X.copy()
        X = self.missing.fit_transform(X, y)
        X = self.cat.fit_transform(X, y)
        X = self.feature_gen.fit_transform(X, y)
        X = self.num.fit_transform(X, y)

        self.feature_names_ = list(X.columns)
        
        return self
    def transform(self, X):
        X = X.copy()
        X = self.missing.transform(X)
        X = self.cat.transform(X)
        X = self.feature_gen.transform(X)
        X = self.num.transform(X)

        return pd.DataFrame(X, columns=self.feature_names_)

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features = None):
        self.features = features
    def fit(self, X, y=None):
        if self.features is None:
            self.features_ = list(X.columns)
        else:
            missing = [f for f in self.features if f not in X.columns]
            if missing:
                raise ValueError(f"Features not found in the input data: {missing}")
            self.features_ = self.features
        return self
    def transform(self, X):
        X = X.copy()
        return X[self.features_]