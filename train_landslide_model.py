"""
train_landslide_model.py
Trains a Random Forest / XGBoost classifier on the Himachal Pradesh
landslide susceptibility dataset and exports the model for web use.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             confusion_matrix)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ------------------------------------------------------------------
# 1. LOAD DATA
#    Download the modelling CSV from Mendeley Data (dataset mxpg845z62)
#    and place it in the same folder, or adjust the path below.
# ------------------------------------------------------------------
DATA_PATH = "himachal_modelling_dataset.csv"   # <-- change if needed
df = pd.read_csv(DATA_PATH)

# The target column is typically named 'landslide' or 'label' (1/0).
# Adjust these two lines to match your CSV header.
TARGET_COL = "class"
FEATURE_COLS = [
    "Aspect_z", "Dist_River_z", "Elevation_z",
    "FlowAccumulation_z", "NDVI_z", "Rainfall_z", "Slope_z"
]  # These are the 7 standardised predictors in the HP dataset

X = df[FEATURE_COLS].copy()
y = df[TARGET_COL].copy()

# ------------------------------------------------------------------
# 2. PREPROCESSING
# ------------------------------------------------------------------
# Drop rows with missing values (or impute with the median)
X = X.fillna(X.median())

# Standardise features (required for XGBoost; optional for RF but good practice)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------------------------------------------------------
# 3. TRAIN RANDOM FOREST
# ------------------------------------------------------------------
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ------------------------------------------------------------------
# 4. TRAIN XGBOOST (for comparison)
# ------------------------------------------------------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# ------------------------------------------------------------------
# 5. EVALUATE
# ------------------------------------------------------------------
def evaluate(model, name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC  : {roc_auc_score(y_test, y_prob):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Landslide", "Landslide"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

evaluate(rf, "Random Forest")
evaluate(xgb_model, "XGBoost")

# ------------------------------------------------------------------
# 6. FEATURE IMPORTANCE (from Random Forest)
# ------------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("\nFeature Importances (Random Forest):")
print(importances)

# ------------------------------------------------------------------
# 7. CROSS-VALIDATION (5-fold stratified)
# ------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_scaled, y, cv=cv, scoring="roc_auc")
print(f"\n5-Fold CV AUC-ROC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ------------------------------------------------------------------
# 8. EXPORT THE MODEL FOR THE WEB APP
#    Save the scaler and the Random Forest model as .pkl files.
#    In production, wrap these in a small Flask/FastAPI service,
#    or convert the RF to TensorFlow.js using tfjs (optional).
# ------------------------------------------------------------------
joblib.dump(scaler, "landslide_scaler.pkl")
joblib.dump(rf, "landslide_rf_model.pkl")
joblib.dump(xgb_model, "landslide_xgb_model.pkl")
print("\nModels saved: landslide_scaler.pkl, landslide_rf_model.pkl, landslide_xgb_model.pkl")