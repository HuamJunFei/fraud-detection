from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(title="Fraud Detection API")

# Load trained ML model
ml_model = joblib.load("ml_model.pkl")

# Load transaction graph & suspicious accounts
G = joblib.load("transaction_graph.pkl")
suspicious_accounts = joblib.load("suspicious_accounts.pkl")

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(transaction: dict):
    """
    transaction = {
        "type": "TRANSFER",
        "amount": 1000,
        "oldbalanceOrg": 5000,
        "newbalanceOrig": 4000,
        "oldbalanceDest": 0,
        "newbalanceDest": 1000,
        "sender": "user123",
        "receiver": "accountX"
    }
    """
    # Prepare ML input
    df = pd.DataFrame([{
        "type": transaction["type"],
        "amount": transaction["amount"],
        "oldbalanceOrg": transaction["oldbalanceOrg"],
        "newbalanceOrig": transaction["newbalanceOrig"],
        "oldbalanceDest": transaction["oldbalanceDest"],
        "newbalanceDest": transaction["newbalanceDest"]
    }])

    # ML prediction
    ml_score = ml_model.predict_proba(df)[0][1]

    # Graph prediction
    receiver = transaction["receiver"]
    graph_score = 0.0
    if receiver in suspicious_accounts:
        graph_score = 0.8  # flag high risk

    # Combine scores
    final_score = 0.7 * ml_score + 0.3 * graph_score

    # Determine risk level
    if final_score > 0.75:
        risk_level = "HIGH"
    elif final_score > 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "ml_score": ml_score,
        "graph_score": graph_score,
        "final_score": final_score,
        "risk_level": risk_level
    }