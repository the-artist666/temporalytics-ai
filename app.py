import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import logging
from api.market_data_fetcher import MarketDataFetcher
from core.feature_engine import FeatureEngine
from core.tdm_field_processor import TDMFieldProcessor
from core.kalman_filter import KalmanFilter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
st.set_page_config(layout="wide", page_title="Temporalytics AI")

@st.cache_resource
def load_resources():
    model_path = "models/xgb_predictor.pkl"
    scaler_path = "models/scaler.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        st.warning("Model or scaler file not found! Please run `python train.py` locally and include `models/xgb_predictor.pkl` and `models/scaler.pkl` in the repository.")
        return None, None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Failed to load model or scaler: {e}")
        return None, None

def run_analysis(coin_id, currency, days):
    fetcher = MarketDataFetcher()
    df = fetcher.fetch_historical_data(coin_id, currency, days)
    if df is None:
        st.error("Could not fetch market data.")
        return None
    
    # Apply Kalman Filter
    kalman = KalmanFilter()
    df['close'] = kalman.filter_series(df['close'].values)
    
    feature_engine = FeatureEngine()
    df_features = feature_engine.generate_features(df.copy())
    
    if df_features.empty:
        st.warning("Not enough data to generate features.")
        return None
    
    tdm_processor = TDMFieldProcessor()
    tdm_pattern = tdm_processor.get_symbolic_pattern(df_features['close'].values, df_features['Volatility'].values)
    tdm_metrics = tdm_processor.calculate_field_metrics(tdm_pattern)
    
    if model is None or scaler is None:
        return df_features, tdm_metrics, None
    
    last_sequence_unscaled = df_features[['close', 'MA_short', 'MA_long', 'EMA', 'Volatility', 'RSI', 'Stoch_K', 'Stoch_D', 'ATR', 'VWAP']].values[-30:]
    full_feature_df = pd.DataFrame(last_sequence_unscaled, columns=['close', 'MA_short', 'MA_long', 'EMA', 'Volatility', 'RSI', 'Stoch_K', 'Stoch_D', 'ATR', 'VWAP'])
    scaler_feature_order = ['close', 'MA_short', 'MA_long', 'EMA', 'Volatility', 'RSI', 'Stoch_K', 'Stoch_D', 'ATR', 'VWAP']
    full_feature_df = full_feature_df[scaler_feature_order]
    scaled_sequence = scaler.transform(full_feature_df)
    
    X_pred = scaled_sequence[:, :-1].flatten().reshape(1, -1)
    
    prediction_scaled = model.predict(X_pred)[0]
    dummy_row = np.zeros((1, len(scaler_feature_order)))
    dummy_row[0, -1] = prediction_scaled
    prediction = scaler.inverse_transform(dummy_row)[0, -1]
    
    return df_features, tdm_metrics, prediction

st.title("📈 Temporalytics AI")
st.caption("An Enhanced AI for Predicting Crypto Prices with XGBoost and Kalman Filtering")

model, scaler = load_resources()

if model is None and scaler is None:
    st.error("Cannot proceed without model and scaler. Please follow setup instructions in README.md.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        coin_id = st.text_input("Coin ID (e.g., bitcoin)", value="bitcoin")
    with col2:
        currency = st.text_input("Currency", value="usd")
    with col3:
        days = st.slider("Days of Historical Data", 30, 365, 90)

    if st.button("Analyze Market"):
        with st.spinner("Analyzing market..."):
            analysis_results = run_analysis(coin_id, currency, days)
        
        if analysis_results:
            df, tdm_metrics, prediction = analysis_results

            st.header("Price Chart")
            st.line_chart(df[['close', 'MA_short', 'MA_long', 'EMA', 'VWAP']])

            st.header("Additional Indicators")
            st.line_chart(df[['RSI', 'Stoch_K', 'Stoch_D', 'ATR']])

            if prediction is not None:
                st.header("Prediction")
                st.metric("Next 24h Price Prediction", f"${prediction:,.2f}")
            else:
                st.warning("Prediction unavailable due to missing model files.")

            st.header("Market Dynamics (TDM Metrics)")
            st.json(tdm_metrics)
