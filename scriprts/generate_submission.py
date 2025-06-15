import pandas as pd
import joblib

# Load test data
test_features = pd.read_csv("data/test_set_features.csv")
respondent_ids = test_features["respondent_id"]

# Load pipeline and models
preprocessor = joblib.load("ML_pipeline.pkl")
X_test = preprocessor.transform(test_features) 


lgbm_h1n1 = joblib.load("models/XGBoost_learning_rate-0.1_max_depth-5_n_estimators-100_h1n1_vaccine.joblib")
lgbm_seasonal = joblib.load("models/XGBoost_learning_rate-0.1_max_depth-3_n_estimators-200_seasonal_vaccine.joblib")

# Predict probabilities
h1n1_preds = lgbm_h1n1.predict_proba(X_test)[:, 1]
seasonal_preds = lgbm_seasonal.predict_proba(X_test)[:, 1]

# Build submission
submission = pd.DataFrame({
    "respondent_id": respondent_ids,
    "h1n1_vaccine": h1n1_preds,
    "seasonal_vaccine": seasonal_preds
})

submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
