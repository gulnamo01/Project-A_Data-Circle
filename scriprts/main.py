import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from ml_pipeline import build_full_pipeline
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

from scipy.stats import mode

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

# Build preprocessing pipeline (shared)
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
model_defs = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
}
if XGBClassifier is not None:
    model_defs["XGBoost"] = XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
if LGBMClassifier is not None:
    model_defs["LightGBM"] = LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1)

# Store results
results = []

# For each target, train and evaluate each model with stratified split
for target in ['h1n1_vaccine', 'seasonal_vaccine']:
    # Stratified split for this target only
    X_train, X_test, y_train_target, y_test_target = train_test_split(
        train_features, train_labels[target], test_size=0.2, random_state=42,
        stratify=train_labels[target]
    )
    # Fit and evaluate supervised models
    for model_name, model in model_defs.items():
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
    
    # --- KMeans as a baseline ---
    # Preprocess the data
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)
    # Set number of clusters to number of classes (2 for binary)
    n_clusters = len(pd.unique(y_train_target))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_train_proc)
    # Assign each cluster to the most common class in the training set
    cluster_to_class = {}
    for cluster in range(n_clusters):
        labels_in_cluster = y_train_target[kmeans.labels_ == cluster]
        if len(labels_in_cluster) == 0:
            cluster_to_class[cluster] = mode(y_train_target, keepdims=True).mode[0]
        else:
            cluster_to_class[cluster] = mode(labels_in_cluster, keepdims=True).mode[0]
    # Predict clusters for test data, then map to class
    test_clusters = kmeans.predict(X_test_proc)
    y_pred_kmeans = [cluster_to_class[c] for c in test_clusters]
    acc_kmeans = accuracy_score(y_test_target, y_pred_kmeans)
    results.append({
        "Target": target,
        "Model": "KMeans (cluster baseline)",
        "Validation Accuracy": acc_kmeans
    })
    print(f"KMeans (cluster baseline) ({target}): Validation accuracy: {acc_kmeans:.4f}")

# Present results as a summary table
results_df = pd.DataFrame(results)
print("\nSummary Report:")
print(results_df.pivot(index="Model", columns="Target", values="Validation Accuracy").to_string())
