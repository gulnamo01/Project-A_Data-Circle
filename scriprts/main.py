import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline
from ml_pipeline import build_full_pipeline
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings

# Optional: XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

warnings.filterwarnings("ignore")

# Load datasets
train_features = pd.read_csv("data/training_set_features.csv")
train_labels = pd.read_csv("data/training_set_labels.csv")
test_features = pd.read_csv("data/test_set_features.csv")

# Define columns (ensure no overlaps!)
ordinal_cols = [
    'h1n1_concern', 'h1n1_knowledge', 'opinion_h1n1_vacc_effective', 'opinion_h1n1_risk',
    'opinion_h1n1_sick_from_vacc', 'opinion_seas_vacc_effective', 'opinion_seas_risk', 'opinion_seas_sick_from_vacc'
]
binary_cols = [
    'behavioral_antiviral_meds', 'behavioral_avoidance', 'behavioral_face_mask', 'behavioral_wash_hands',
    'behavioral_large_gatherings', 'behavioral_outside_home', 'behavioral_touch_face', 'chronic_med_condition',
    'child_under_6_months', 'health_worker', 'health_insurance'
]
numerical_cols = ['household_adults', 'household_children']
categorical_cols = [
    'age_group', 'education', 'race', 'sex', 'income_poverty', 'marital_status', 'rent_or_own',
    'employment_status', 'hhs_geo_region', 'census_msa', 'employment_industry', 'employment_occupation'
]
doctor_rec_cols = ['doctor_recc_h1n1', 'doctor_recc_seasonal']
employment_info_cols = ['employment_industry', 'employment_occupation']

ordinal_enc_cols = ['age_group', 'education', 'income_poverty']
binary_enc_cols = ['sex', 'marital_status', 'rent_or_own']
nominal_enc_cols = ['race', 'hhs_geo_region', 'census_msa', 'employment_industry', 'employment_occupation', 'employment_status']

# Remove overlaps
categorical_cols = [col for col in categorical_cols if col not in doctor_rec_cols and col not in employment_info_cols]
binary_cols = [col for col in binary_cols if col not in doctor_rec_cols]
nominal_enc_cols = [col for col in nominal_enc_cols if col not in employment_info_cols]

# Custom orderings for ordinal columns
age_order = ['18 - 34 Years', '35 - 44 Years', '45 - 54 Years', '55 - 64 Years', '65+ Years']
edu_order = ['< 12 Years', '12 Years', 'Some College', 'College Graduate']
income_order = ['Below Poverty', '<= $75,000, Above Poverty', '> $75,000']
enc_order = [age_order, edu_order, income_order]

# Build preprocessing pipeline
preprocessor = build_full_pipeline(
    ordinal_cols=ordinal_cols,
    binary_cols=binary_cols,
    numerical_cols=numerical_cols,
    categorical_cols=categorical_cols,
    doctor_rec_cols=doctor_rec_cols,
    employment_info_cols=employment_info_cols,
    ordinal_enc_cols=ordinal_enc_cols,
    binary_enc_cols=binary_enc_cols,
    nominal_enc_cols=nominal_enc_cols,
    enc_order=enc_order
)

# Define models to try
# weight balancing : class_weight='balanced'
model_defs = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=200, random_state=42, n_jobs=-1)
}
if XGBClassifier is not None:
    model_defs["XGBoost"] = XGBClassifier(class_weight='balanced', n_estimators=200, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
if LGBMClassifier is not None:
    model_defs["LightGBM"] = LGBMClassifier(class_weight='balanced', n_estimators=200, random_state=42, n_jobs=-1)

results = []

# For each target, train and evaluate each model with stratified split
for target in ['h1n1_vaccine', 'seasonal_vaccine']:
    # Stratified split for this target only
    X_train, X_test, y_train_target, y_test_target = train_test_split(
        train_features, train_labels[target], test_size=0.2, random_state=42,
        stratify=train_labels[target]
    )
    for model_name, model in model_defs.items():
        # Use balancing algorithem only for the imbalanced target
        if target == 'h1n1_vaccine':
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('smote', SMOTE(random_state=42)),
                # ('over', RandomOverSampler( random_state=42)),
                # ('under', RandomUnderSampler(random_state=42)),
                ('classifier', model)
            ])
        else:
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
        pipeline.fit(X_train, y_train_target)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test_target, y_pred)
        results.append({
            "Target": target,
            "Model": model_name,
            "Validation Accuracy": acc
        })
        print(f"{model_name} ({target}): Validation accuracy: {acc:.4f}")

# Present results as a summary table
results_df = pd.DataFrame(results)
print("\nSummary Report:")
print(results_df.pivot(index="Model", columns="Target", values="Validation Accuracy").to_string())
