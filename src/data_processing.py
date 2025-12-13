import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class AggregateFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        agg = X.groupby('CustomerId').agg(
            total_amount=('Amount', 'sum'),
            avg_amount=('Amount', 'mean'),
            transaction_count=('Amount', 'count'),
            std_amount=('Amount', 'std')
        ).reset_index()
        return X.merge(agg, on='CustomerId', how='left')


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
        X['trans_hour'] = X['TransactionStartTime'].dt.hour
        X['trans_day'] = X['TransactionStartTime'].dt.day
        X['trans_month'] = X['TransactionStartTime'].dt.month
        X['trans_year'] = X['TransactionStartTime'].dt.year
        return X


def build_feature_pipeline(numerical_features, categorical_features):

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_features),
        ('cat', cat_pipeline, categorical_features)
    ])

    return Pipeline([
        ('aggregate', AggregateFeatures()),
        ('datetime', DatetimeFeatures()),
        ('preprocessing', preprocessor)
    ])
