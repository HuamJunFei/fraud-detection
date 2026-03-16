import pandas as pd
import xgboost as xgb
import joblib

def train_behavioral_model():
    # 1. Load Data
    df = pd.read_csv('data/dataset.csv')
    X = df[['amount', 'time_hour', 'account_age']] # Example features
    y = df['is_fraud']
    
    # 2. Train Model
    model = xgb.XGBClassifier(n_estimators=100)
    model.fit(X, y)
    
    # 3. Save Model
    model.save_model('models/behavioral.json')
    print("Behavioral ML model saved!")

if __name__ == "__main__":
    train_behavioral_model()