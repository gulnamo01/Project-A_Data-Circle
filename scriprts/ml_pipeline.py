import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
import joblib

def build_full_pipeline(
    ordinal_cols,
    binary_cols,
    numerical_cols,
    categorical_cols,
    doctor_rec_cols,
    employment_info_cols
):
    """
    Builds a full pipeline: imputation, encoding, and classifier.
    """
    # Imputation step
    imputer = ColumnTransformer([
        ('ordinal_mode', SimpleImputer(strategy='most_frequent'), ordinal_cols),
        ('binary_mode', SimpleImputer(strategy='most_frequent'), [col for col in binary_cols if col not in doctor_rec_cols]),
        ('binary_zero', SimpleImputer(strategy='constant', fill_value=0),doctor_rec_cols ),
        ('numerical_median', SimpleImputer(strategy='median'), numerical_cols),
        ('cat_mode', SimpleImputer(strategy='most_frequent'), [col for col in categorical_cols if col not in     employment_info_cols]),
        ('cat_missing', SimpleImputer(strategy='constant', fill_value='Not Applicable'), employment_info_cols)
    ], remainder='passthrough')

    # Encoding 
    # encoder = ColumnTransformer([
    #     ('ord', ordinal_encoder, ordinal_cols),
    #     ('bin', binary_encoder, binary_cols),
    #     ('nom', nominal_encoder, nominal_cols)
    # ], remainder='passthrough')

    # Classifier
    #classifier = RandomForestClassifier(random_state=42)

    # Full pipeline
    pipeline = Pipeline([
        ('imputer', imputer),
        # ('encoder', encoder),
        # ('classifier', classifier)
    ])

    return pipeline

def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)

def load_pipeline(path):
    return joblib.load(path)
