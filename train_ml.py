import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("data/fraud_dataset.csv")

# Features and target
X = df[["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]]
y = df["isFraud"]

# Column types
cat_cols = ["type"]
num_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

# Preprocessing pipeline
preprocess = ColumnTransformer(
    [
        ("cat", OneHotEncoder(), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# Full ML pipeline
model = Pipeline(
    [
        ("prep", preprocess),
        ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", scale_pos_weight=(y==0).sum()/(y==1).sum()))
    ]
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print(classification_report(y_test, pred))

# Save model
joblib.dump(model, "ml_model.pkl")
print("ML model saved as ml_model.pkl")