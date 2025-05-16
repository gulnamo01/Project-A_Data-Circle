from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

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
    """
    Builds a full pipeline: imputation, encoding, and classifier.
    """
    # Imputation step
    imputer = ColumnTransformer([
    ('ordinal_mode', SimpleImputer(strategy='most_frequent'), ordinal_cols),  # Mode for low-missing ordinal (e.g., h1n1_concern = 2)
    ('binary_mode', SimpleImputer(strategy='most_frequent'), [col for col in binary_cols if col not in doctor_rec_cols]),  # Mode for health_insurance (1), health_worker (0), etc.
    ('binary_zero', SimpleImputer(strategy='constant', fill_value=9999), doctor_rec_cols),  # 9999 for doctor recs
    ('numerical_mean', SimpleImputer(strategy='mean'), numerical_cols),
    ('cat_mode', SimpleImputer(strategy='most_frequent'), [col for col in categorical_cols if col not in  employment_info_cols]),  # Mode for education, income_poverty, etc.
    ('cat_missing', SimpleImputer(strategy='constant', fill_value='Not Applicable'),  employment_info_cols)  # Not Applicable for ~50% missing
], remainder='passthrough')  # Keep respondent_id, targets

    # Encoding
    ordinal_encoder = OrdinalEncoder(categories=enc_order, handle_unknown='use_encoded_value', unknown_value=-1)
    binary_encoder = LabelEncoder()
    nominal_encoder = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

    encoder = ColumnTransformer([
        ('ord', ordinal_encoder, ordinal_enc_cols),
        ('bin', binary_encoder, binary_enc_cols),
        ('nom', nominal_encoder, nominal_enc_cols)
    ], remainder='passthrough')


    # Classifier
    #classifier = RandomForestClassifier(random_state=42)

    # Full pipeline
    pipeline = Pipeline([
        ('imputer', imputer),
        ('encoder', encoder),
        # ('classifier', classifier)
    ])
    return pipeline

def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)

def load_pipeline(path):
    return joblib.load(path)
