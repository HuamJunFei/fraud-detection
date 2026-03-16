from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import pickle
import xgboost as xgb
import networkx as nx

app = FastAPI(title="Fraud Detection API")

# --- 1. Load Models Once at Startup ---
ml_model = xgb.Booster()
ml_model.load_model("models/behavioral.json")

with open("models/graph.pkl", "rb") as f:
    fraud_graph = pickle.load(f)

# --- 2. Define Request Schema (Matches IEEE-CIS features) ---
class Transaction(BaseModel):
    transaction_id: str
    card1: int
    addr1: int
    amount: float
    time_hour: int

# --- 3. Engine Logics ---

async def get_ml_score(data: Transaction):
    # Prepare features as a DMatrix for XGBoost
    # In production, use the exact feature list used during training
    features = [[data.amount, data.time_hour]] 
    # Use your model to get probability
    score = ml_model.predict(xgb.DMatrix(features))[0]
    return float(score)

async def get_graph_score(data: Transaction):
    # Check centrality in the graph
    card_node = f"card_{data.card1}"
    addr_node = f"addr_{data.addr1}"
    
    # Calculate risk: If the node exists, use its pagerank
    if fraud_graph.has_node(card_node):
        pagerank = fraud_graph.nodes[card_node].get('pagerank', 0)
        # Normalize (e.g., if pagerank > 0.05, it's high risk)
        return min(pagerank * 10, 1.0)
    return 0.1

# --- 4. Main API Endpoint ---



@app.post("/v1/analyze")
async def analyze_transaction(tx: Transaction):
    # Run both engines in parallel for performance
    ml_task = get_ml_score(tx)
    graph_task = get_graph_score(tx)
    
    ml_score, graph_score = await asyncio.gather(ml_task, graph_task)
    
    # Final Formula: 0.7 ML + 0.3 Graph
    final_score = (0.7 * ml_score) + (0.3 * graph_score)
    
    # Thresholds
    decision = "APPROVE"
    if final_score >= 0.75:
        decision = "BLOCK"
    elif final_score >= 0.60:
        decision = "REVIEW"
        
    return {
        "transaction_id": tx.transaction_id,
        "final_score": round(final_score, 2),
        "decision": decision,
        "model_signals": {
            "behavioral": round(ml_score, 2),
            "network": round(graph_score, 2)
        }
    }