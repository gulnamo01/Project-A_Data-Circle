from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
import joblib

# Custom transformer for applying LabelEncoder to multiple columns (using integer indices)
class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.encoders = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for i, col in enumerate(X.columns):
            le = LabelEncoder()
            le.fit(X[col])
            self.encoders[i] = le
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for i, col in enumerate(X.columns):
            X[col] = self.encoders[i].transform(X[col])
        return X

def build_full_pipeline(
    ordinal_cols,
    binary_cols,
    numerical_cols,
    categorical_cols,
    doctor_rec_cols,
    employment_info_cols,
    ordinal_enc_cols,
    binary_enc_cols,
    nominal_enc_cols,
    enc_order
):
    # Helper: columns to impute but NOT encode
    ordinal_noenc = [col for col in ordinal_cols if col not in ordinal_enc_cols]
    binary_noenc = [col for col in binary_cols if col not in binary_enc_cols]
    categorical_noenc = [
        col for col in categorical_cols
        if col not in nominal_enc_cols and col not in ordinal_enc_cols and col not in binary_enc_cols
    ]

    # Pipelines
    ordinal_enc_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(categories=enc_order, handle_unknown='use_encoded_value', unknown_value=-1))
    ])
    ordinal_noenc_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    binary_enc_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', MultiColumnLabelEncoder())
    ])
    binary_noenc_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    nominal_enc_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])
    categorical_noenc_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])
    doctor_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=9999))
    ])
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    emp_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Not Applicable'))
    ])

    preprocessor = ColumnTransformer([
        ('ord_enc', ordinal_enc_pipe, ordinal_enc_cols),
        ('ord_noenc', ordinal_noenc_pipe, ordinal_noenc),
        ('bin_enc', binary_enc_pipe, binary_enc_cols),
        ('bin_noenc', binary_noenc_pipe, binary_noenc),
        ('nom_enc', nominal_enc_pipe, nominal_enc_cols),
        ('cat_noenc', categorical_noenc_pipe, categorical_noenc),
        ('doctor', doctor_pipe, doctor_rec_cols),
        ('num', num_pipe, numerical_cols),
        ('emp', emp_pipe, employment_info_cols)
    ], remainder='passthrough')

    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    return pipeline

def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)

def load_pipeline(path):
    return joblib.load(path)
