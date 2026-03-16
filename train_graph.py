import pandas as pd
import networkx as nx
import pickle

def build_fraud_graph():
    # 1. Load Data
    df = pd.read_csv('data/dataset.csv')
    
    # 2. Build Graph (User -> Target Account)
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['user_id'], row['target_account'])
    
    # 3. Save Graph
    with open('models/graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    print("Graph model saved!")

if __name__ == "__main__":
    build_fraud_graph()