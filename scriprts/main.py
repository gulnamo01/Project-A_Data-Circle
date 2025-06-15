import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, RocCurveDisplay
from ml_pipeline import build_full_pipeline, save_pipeline
import joblib
import matplotlib.pyplot as plt
import warnings
import re
import os

SAVE_MODELS = True  # Set to False to skip saving models

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

# -------------------------------
# Load datasets
# -------------------------------
train_features = pd.read_csv("data/training_set_features.csv")
train_labels = pd.read_csv("data/training_set_labels.csv")
test_features = pd.read_csv("data/test_set_features.csv")

# -------------------------------
# Define columns
# -------------------------------
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

# -------------------------------
# Build preprocessing pipeline
# -------------------------------
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
preprocessor.fit(train_features)  # Fit on training data
save_pipeline(preprocessor, 'ML_pipeline.pkl')

# -------------------------------
# Define models with class_weight='balanced'
# -------------------------------
model_defs = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
}
if XGBClassifier is not None:
    model_defs["XGBoost"] = XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
if LGBMClassifier is not None:
    model_defs["LightGBM"] = LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')

# -------------------------------
# Define hyperparameter grids
# -------------------------------
param_grids = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10],
        'solver': ['liblinear', 'lbfgs'],
        'penalty': ['l2']
    },
    "Random Forest": {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
}
if XGBClassifier is not None:
    param_grids["XGBoost"] = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'n_estimators': [100, 200]
    }
if LGBMClassifier is not None:
    param_grids["LightGBM"] = {
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 63],
        'n_estimators': [100, 200]
    }

def params_to_filename(params):
    if not params:
        return "default"
    items = []
    for k, v in sorted(params.items()):
        if isinstance(v, (list, tuple)):
            v = '-'.join(map(str, v))
        items.append(f"{k}-{v}")
    s = "_".join(items)
    s = re.sub(r'[^A-Za-z0-9_\-\.]', '', s)
    return s

# -------------------------------
# Preprocess ALL data once
# -------------------------------
print("Preprocessing all features...")
X_preprocessed = preprocessor.fit_transform(train_features)
test_preprocessed = preprocessor.transform(test_features)  # For future predictions

if SAVE_MODELS and not os.path.exists("models"):
    os.makedirs("models")
if not os.path.exists("roc_curves"):
    os.makedirs("roc_curves")

results = []

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for target in ['h1n1_vaccine', 'seasonal_vaccine']:
    print(f"\nTraining models for target: {target}")
    y = train_labels[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X_preprocessed, y, 
        test_size=0.2, random_state=42, 
        stratify=y
    )
    # Only apply SMOTE to h1n1_vaccine
    if target == 'h1n1_vaccine':
        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        
        # ros = RandomOverSampler( random_state=42)
        # X_train, y_train = ros.fit_resample(X_train, y_train)

        # rus = RandomUnderSampler( random_state=42)
        # X_train, y_train = rus.fit_resample(X_train, y_train)



    for model_name, model in model_defs.items():
        param_grid = param_grids.get(model_name, {})
        grid = GridSearchCV(
            model,
            param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        if hasattr(grid, "predict_proba"):
            y_proba = grid.predict_proba(X_test)[:, 1]
        else:
            y_proba = grid.decision_function(X_test)
        auc = roc_auc_score(y_test, y_proba)
        best_params = grid.best_params_
        # Save model
        if SAVE_MODELS:
            params_str = params_to_filename(best_params)
            filename = f"{model_name.replace(' ', '')}_{params_str}_{target}.joblib"
            filename = filename.replace('__', '_').replace(' ', '').replace('/', '_')
            joblib.dump(grid.best_estimator_, f"models/{filename}")
            print(f"Model saved to models/{filename}")
        # Plot and save ROC curve
        fig, ax = plt.subplots()
        RocCurveDisplay.from_estimator(grid, X_test, y_test, ax=ax)
        plt.title(f"ROC Curve: {model_name} ({target})")
        img_name = f"roc_curves/ROC_{model_name.replace(' ', '')}_{target}.png"
        plt.savefig(img_name, bbox_inches='tight')
        plt.close(fig)
        print(f"ROC curve saved to {img_name}")

        # Cross-validation on full data with best params
        model_for_cv = model.set_params(**best_params)
        # For SMOTE, CV needs to be handled with a pipeline; here we report CV on preprocessed (not resampled) data
        auc_scores = cross_val_score(
            model_for_cv, X_preprocessed, y, 
            scoring='roc_auc', cv=cv_strategy, n_jobs=-1
        )
        f1_scores = cross_val_score(
            model_for_cv, X_preprocessed, y, 
            scoring='f1', cv=cv_strategy, n_jobs=-1
        )
        acc_scores = cross_val_score(
            model_for_cv, X_preprocessed, y, 
            scoring='accuracy', cv=cv_strategy, n_jobs=-1
        )

        results.append({
            "Target": target,
            "Model": model_name,
            "Validation Accuracy": acc,
            "F1 Score": f1,
            "AUC": auc,
            "CV Accuracy Mean": acc_scores.mean(),
            "CV Accuracy Std": acc_scores.std(),
            "CV F1 Mean": f1_scores.mean(),
            "CV F1 Std": f1_scores.std(),
            "CV AUC Mean": auc_scores.mean(),
            "CV AUC Std": auc_scores.std(),
            "Best Params": str(best_params),
            "ROC Curve Image": img_name
        })
        print(f"\n{model_name} ({target})")
        print(f"Best params: {best_params}")
        print(f"Validation accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"CV mean AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
        print(f"CV mean F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
        print(f"CV mean Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")

# -------------------------------
# Results Summary
# -------------------------------
results_df = pd.DataFrame(results)
print("\n=== Summary Report ===")
print("Metrics reported: Validation Accuracy, F1 Score, ROC AUC, Cross-Validation Means/Stds, ROC Curve Image Path\n")
print(results_df[[
    "Model", "Target", "Validation Accuracy", "F1 Score", "AUC",
    "CV Accuracy Mean", "CV Accuracy Std", "CV F1 Mean", "CV F1 Std", "CV AUC Mean", "CV AUC Std",
    "ROC Curve Image"
]].to_string(index=False))
print("\nBest Parameters for each model/target:")
print(results_df[['Model', 'Target', 'Best Params']].to_string(index=False))
