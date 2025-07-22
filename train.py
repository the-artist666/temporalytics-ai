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
    
    COINS = ["bitcoin", "ethereum", "solana", "binancecoin", "xrp", "cardano", "dogecoin", "polkadot", "chainlink", "polygon"]
    coin_data = {}
    
    # Fetch data for all coins
    for coin in COINS:
        logger.info(f"Fetching data for {coin}...")
        df = fetcher.fetch_historical_data(coin_id=coin, vs_currency="usd", days=90)
        if df is None or df.empty:
            logger.error(f"Failed to fetch data for {coin}.")
            continue
        coin_data[coin] = df
    
    # Process features
    for coin, df in coin_data.items():
        df = feature_engine.compute_technical_indicators(df, coin_data)
        if df is None:
            logger.error(f"Failed to compute indicators for {coin}.")
            continue
        df = tdm_processor.compute_tdm_metrics(df)
        if df is None:
            logger.error(f"Failed to compute TDM metrics for {coin}.")
            continue
        coin_data[coin] = df
    
    # Train models
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    for coin, df in coin_data.items():
        feature_columns = ['sma_20', 'ema_12', 'rsi', 'stoch', 'atr', 'vwap', 'bb_upper', 'bb_lower', 'macd', 'obv', 'ichimoku_tenkan', 'ichimoku_kijun', 'adx', 'volatility', 'sharpe', 'trend', 'momentum', 'conviction', 'stability']
        X = df[feature_columns]
        y = df['close'].shift(-1)
        
        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model.fit(X_scaled, y)
        
        joblib.dump(model, f"models/xgb_{coin}.pkl")
        joblib.dump(scaler, f"models/scaler_{coin}.pkl")
        df.to_csv(f"data/{coin}usd_1h.csv")
        
        logger.info(f"Model and data saved for {coin}.")

if __name__ == "__main__":
    train_model()
