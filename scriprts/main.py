import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
import joblib
from sklearn.model_selection import train_test_split
from ml_pipeline import build_full_pipeline, save_pipeline, load_pipeline


#open the datasets
#Test
submission = pd.read_csv("../data/submission_format.csv")
test_features = pd.read_csv("../data/test_set_features.csv")
#Train
train_features = pd.read_csv("../data/training_set_features.csv")
train_labels = pd.read_csv("../data/training_set_labels.csv")

# Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    train_features, train_labels, 
    test_size=0.2,           
)


# Column definitions
ordinal_cols = [
    'h1n1_concern', 'h1n1_knowledge', 'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
    'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc'
]

binary_cols = [
    'behavioral_antiviral_meds', 'behavioral_avoidance', 'behavioral_face_mask', 'behavioral_wash_hands',
    'behavioral_large_gatherings', 'behavioral_outside_home', 'behavioral_touch_face', 'chronic_med_condition',
    'child_under_6_months', 'health_worker', 'health_insurance', 'doctor_recc_h1n1', 'doctor_recc_seasonal'
]
numerical_cols = ['household_adults', 'household_children']
categorical_cols = [
    'age_group', 'education', 'race', 'sex', 'income_poverty', 'marital_status', 'rent_or_own',
    'employment_status', 'hhs_geo_region', 'census_msa', 'employment_industry', 'employment_occupation'
]
doctor_rec_cols = ['doctor_recc_h1n1', 'doctor_recc_seasonal']
employment_info_cols = ['employment_industry', 'employment_occupation']


# Build pipeline
pipeline = build_full_pipeline(
    ordinal_cols=ordinal_cols,
    binary_cols=binary_cols,
    numerical_cols=numerical_cols,
    categorical_cols=categorical_cols,
    doctor_rec_cols = doctor_rec_cols,
    employment_info_cols= employment_info_cols
)

# Fit pipeline
pipeline.fit(X_train, y_train)

# Save pipeline
save_pipeline(pipeline, 'full_train_pipeline.pkl')
pipeline.fit(X_train, y_train)

# Fit validation pipeline
pipeline.fit(X_test, y_test)
# Save validation pipeline
save_pipeline(pipeline, 'full_test_pipeline.pkl')
print("done the imputation pipeline")

# Predict on new data
# pipeline = load_pipeline('full_train_pipeline.pkl')
# X_new = test_features
# y_pred = pipeline.predict(X_new)
