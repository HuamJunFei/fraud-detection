from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import joblib
import pickle
import xgboost as xgb

# Import your engine logic
# Assuming you have simple wrapper functions in your engine files
from app.engines.behavioral import predict_behavior
from app.engines.network import get_network_risk

app = FastAPI(title="Fraud Detection API")

# 1. Load Models at Startup (Memory efficient)
# In production, these should be loaded once, not on every request
ml_model = xgb.Booster()
ml_model.load_model("models/behavioral.json")

with open("models/graph.pkl", "rb") as f:
    fraud_graph = pickle.load(f)

# 2. Define Request Schema
class Transaction(BaseModel):
    transaction_id: str
    user_id: str
    target_account: str
    amount: float
    time_hour: int
    account_age: int

# 3. Main Endpoint
@app.post("/v1/analyze")
async def analyze_transaction(tx: Transaction):
    # Run ML and Graph engines in parallel
    ml_task = predict_behavior(ml_model, tx)
    graph_task = get_network_risk(fraud_graph, tx)
    
    ml_score, graph_score = await asyncio.gather(ml_task, graph_task)
    
    # Risk Scoring Formula
    final_score = (0.7 * ml_score) + (0.3 * graph_score)
    
    # Decision Logic
    decision = "APPROVE"
    if final_score >= 0.75:
        decision = "BLOCK"
    elif final_score >= 0.60:
        decision = "REVIEW"
        
    return {
        "transaction_id": tx.transaction_id,
        "final_score": round(final_score, 2),
        "decision": decision,
        "details": {
            "ml_behavior": round(ml_score, 2),
            "graph_network": round(graph_score, 2)
        }
    }