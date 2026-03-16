import pandas as pd
import networkx as nx
import pickle

def build_fraud_graph():
    print("Loading IEEE-CIS transaction data...")
    # We only need the transaction data for the graph
    df = pd.read_csv('data/train_transaction.csv', usecols=['TransactionID', 'card1', 'addr1', 'isFraud'])
    
    # 1. Initialize a Graph
    # We will use an Undirected Graph to see which cards frequent which locations
    G = nx.Graph()
    
    # 2. Add edges
    # Node A = card1 (Credit Card ID), Node B = addr1 (Location)
    # The weight will be the number of transactions shared between these nodes
    print("Building nodes and edges...")
    for _, row in df.iterrows():
        # Nodes: card1 and addr1
        card = f"card_{int(row['card1'])}"
        addr = f"addr_{int(row['addr1'])}"
        
        if G.has_edge(card, addr):
            G[card][addr]['weight'] += 1
        else:
            G.add_edge(card, addr, weight=1, is_fraud=row['isFraud'])

    # 3. Calculate Graph Features (PageRank)
    # High PageRank in this graph identifies "hub" locations or "active" cards
    print("Calculating graph metrics (PageRank)...")
    pagerank_scores = nx.pagerank(G, weight='weight')
    
    # Add PageRank as an attribute to the nodes
    nx.set_node_attributes(G, pagerank_scores, 'pagerank')

    # 4. Save the graph for use in FastAPI
    with open('models/graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    
    print("Graph model saved to models/graph.pkl!")

if __name__ == "__main__":
    build_fraud_graph()