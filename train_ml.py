import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_merge_data():
    # Load the two main files
    # Note: Ensure you have these files in your 'data/' folder
    trans = pd.read_csv('data/train_transaction.csv')
    iden = pd.read_csv('data/train_identity.csv')
    
    # Merge on TransactionID
    df = pd.merge(trans, iden, on='TransactionID', how='left')
    
    # Clean up memory
    del trans, iden
    return df

def preprocess_data(df):
    # Label encode categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Fill missing values with -999 (XGBoost handles this well)
    df = df.fillna(-999)
    return df

def train_behavioral_model():
    print("Loading and merging IEEE-CIS data...")
    df = load_and_merge_data()
    
    # Define features (excluding target and ID)
    target = 'isFraud'
    features = [c for c in df.columns if c not in [target, 'TransactionID']]
    
    df = preprocess_data(df)
    
    X = df[features]
    y = df[target]
    
    # Train/Test split
    X_train, _, y_train,_ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist', # Faster for large datasets
        scale_pos_weight=25 # Essential for IEEE-CIS imbalance
    )
    
    model.fit(X_train, y_train)
    
    # Save model
    model.save_model('models/behavioral.json')
    print("Model saved to models/behavioral.json!")

if __name__ == "__main__":
    train_behavioral_model()