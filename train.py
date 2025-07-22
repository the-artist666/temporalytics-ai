import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from api.market_data_fetcher import MarketDataFetcher
from core.feature_engine import FeatureEngine
from core.tdm_field_processor import TDMFieldProcessor
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model():
    fetcher = MarketDataFetcher()
    feature_engine = FeatureEngine()
    tdm_processor = TDMFieldProcessor()
    
    # Fetch historical data
    df = fetcher.fetch_historical_data(coin_id="bitcoin", vs_currency="usd", days=90)
    if df is None or df.empty:
        logger.error("Failed to fetch historical data.")
        return
    
    # Compute features
    df = feature_engine.compute_technical_indicators(df)
    if df is None:
        logger.error("Failed to compute technical indicators.")
        return
    
    df = tdm_processor.compute_tdm_metrics(df)
    if df is None:
        logger.error("Failed to compute TDM metrics.")
        return
    
    # Prepare features and target
    feature_columns = ['sma_20', 'ema_12', 'rsi', 'stoch', 'atr', 'vwap', 'trend', 'momentum', 'conviction', 'stability']
    X = df[feature_columns]
    y = df['close'].shift(-1)  # Predict next hour's price
    
    # Drop NaN values
    valid_idx = X.dropna().index.intersection(y.dropna().index)
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_scaled, y)
    
    # Save model and scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_predictor.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    
    # Save data for reference
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/btcusd_1h.csv")
    
    logger.info("Model training completed and saved.")

if __name__ == "__main__":
    train_model()
