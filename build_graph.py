import pandas as pd
import networkx as nx
import joblib

# Load dataset
df = pd.read_csv("data/fraud_dataset.csv")

# Create directed graph from transactions
G = nx.from_pandas_edgelist(
    df,
    source="nameOrig",
    target="nameDest",
    create_using=nx.DiGraph()
)

# Optionally store accounts that receive many transactions
suspicious_accounts = [node for node in G.nodes() if G.in_degree(node) > 2]

# Save graph and suspicious accounts
joblib.dump(G, "transaction_graph.pkl")
joblib.dump(suspicious_accounts, "suspicious_accounts.pkl")

print(f"Graph saved with {len(G.nodes())} nodes and {len(G.edges())} edges")
print(f"Suspicious accounts (>2 incoming tx): {len(suspicious_accounts)}")