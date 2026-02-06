# =========================
# IMPORT LIBRARIES
# =========================
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


# =========================
# LOAD DATA
# =========================
DATA_PATH = os.path.join("data", "Traveling_Dataset.csv")
df = pd.read_csv(DATA_PATH)


# =========================
# FEATURE SELECTION
# =========================
numerical_features = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "PitchSatisfactionScore", "NumberOfChildrenVisiting", "MonthlyIncome"
]

categorical_features = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Passport", "OwnCar", "Designation"
]

X = df[numerical_features + categorical_features]
y = df["ProdTaken"]


# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)


# =========================
# PREPROCESSING
# =========================
numeric_transformer = MinMaxScaler()

product_encoder = OrdinalEncoder(
    categories=[["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]]
)

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("product", product_encoder, ["ProductPitched"]),
        ("cat", categorical_transformer,
         ["TypeofContact", "Occupation", "Gender",
          "MaritalStatus", "Passport", "OwnCar", "Designation"])
    ]
)


# =========================
# PIPELINE WITH SMOTE
# =========================
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])


# =========================
# HYPERPARAMETER TUNING
# =========================
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 12, 16],
    "model__min_samples_split": [2, 4],
    "model__min_samples_leaf": [1, 2],
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_pipeline = grid_search.best_estimator_


# =========================
# EVALUATION
# =========================
y_pred = best_pipeline.predict(X_test)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# =========================
# SAVE ARTIFACTS
# =========================
os.makedirs("pkl", exist_ok=True)

joblib.dump(best_pipeline.named_steps["model"], "pkl/tourism_model.pkl")
joblib.dump(best_pipeline.named_steps["preprocessor"], "pkl/preprocessor.pkl")

print("✅ Model and preprocessor saved successfully")
