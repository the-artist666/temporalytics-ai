import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
import pickle
import logging
import os
from api.market_data_fetcher import MarketDataFetcher
from core.feature_engine import FeatureEngine
from core.kalman_filter import KalmanFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

COIN_ID = "bitcoin"
CURRENCY = "usd"
DAYS = 90
FEATURE_COLS = ['close', 'MA_short', 'MA_long', 'EMA', 'Volatility', 'RSI', 'Stoch_K', 'Stoch_D', 'ATR', 'VWAP']
TARGET_COL = 'close'
SEQUENCE_LENGTH = 30
DATA_DIR = "data"
MODELS_DIR = "models"
DATA_FILE = os.path.join(DATA_DIR, f"{COIN_ID}_{CURRENCY}_{DAYS}d.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "xgb_predictor.pkl")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.pkl")

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length), :-1]
        y = data[i + seq_length, -1]
        xs.append(x.flatten())
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    logging.info("Fetching Data...")
    fetcher = MarketDataFetcher()
    df = fetcher.fetch_historical_data(COIN_ID, CURRENCY, DAYS)
    if df is None:
        logging.error("Failed to fetch data. Exiting.")
        return
    df.to_csv(DATA_FILE)

    logging.info("Applying Kalman Filter...")
    kalman = KalmanFilter()
    df['close'] = kalman.filter_series(df['close'].values)

    logging.info("Generating Features...")
    feature_engine = FeatureEngine()
    df_features = feature_engine.generate_features(df)

    cols = [col for col in FEATURE_COLS if col != TARGET_COL] + [TARGET_COL]
    df_model_data = df_features[cols]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_data = scaler.fit_transform(df_model_data)

    with open(SCALER_FILE, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Scaler saved to {SCALER_FILE}")

    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)
    logging.info(f"Created {len(X)} sequences for training.")

    logging.info("Training XGBoost Model...")
    model = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='reg:squarederror',
        random_state=42
    )
    model.fit(X, y)
    logging.info("Model training completed.")

    logging.info("Saving Model...")
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    logging.info(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    main()
