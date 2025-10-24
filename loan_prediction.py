# loan_prediction.py

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# 1) Load dataset
data_path = "loan.csv"
if data_path.endswith(".csv"):
    data = pd.read_csv(data_path)
else:
    data = pd.read_excel(data_path)
print("✅ Data loaded successfully")

# 2) Clean and preprocess
if "Dependents" in data.columns:
    data["Dependents"] = data["Dependents"].replace("3+", 3)
    data["Dependents"] = data["Dependents"].fillna(0).astype(int)

# Fill numerics with median, categoricals with mode
data.fillna(data.median(numeric_only=True), inplace=True)
data.fillna(data.mode().iloc[0], inplace=True)

# Drop id if present
if "Loan_ID" in data.columns:
    data.drop(["Loan_ID"], axis=1, inplace=True)

# 3) Encode categoricals
cat_cols = data.select_dtypes(include=["object"]).columns
encoder = LabelEncoder()
for col in cat_cols:
    data[col] = encoder.fit_transform(data[col])

# 4) Split
X = data.drop(["Loan_Status"], axis=1)
y = data["Loan_Status"]
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Scale for Logistic only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6) Fit models
log_model = LogisticRegression(max_iter=200, random_state=42)
log_model.fit(X_train_scaled, y_train)

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.07,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="logloss",
    random_state=42
)
xgb_model.fit(X_train, y_train)

# 7) Probabilities and metrics
y_proba_log = log_model.predict_proba(X_test_scaled)[:, 1]
y_pred_log = (y_proba_log >= 0.5).astype(int)

y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
y_pred_xgb = (y_proba_xgb >= 0.5).astype(int)

print(f"Logistic AUC: {roc_auc_score(y_test, y_proba_log):.4f}")
print(f"XGBoost  AUC: {roc_auc_score(y_test, y_proba_xgb):.4f}")

# 8) Simple average ensemble
w = 0.5  # tune on validation if needed
y_proba_ens = w * y_proba_log + (1 - w) * y_proba_xgb
y_pred_ens = (y_proba_ens >= 0.5).astype(int)

print(f"Ensemble  AUC: {roc_auc_score(y_test, y_proba_ens):.4f}")
print(f"Ensemble  F1 : {f1_score(y_test, y_pred_ens):.4f}")
print(f"Ensemble  Acc: {accuracy_score(y_test, y_pred_ens):.4f}")

# 9) Save artifacts for app.py
os.makedirs("model", exist_ok=True)
joblib.dump(log_model, "model/logistic_model.pkl")
joblib.dump(xgb_model, "model/xgboost_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("✅ Models and scaler saved in model/")
