import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from api.market_data_fetcher import MarketDataFetcher
from api.realtime_price_fetcher import RealtimePriceFetcher
from core.feature_engine import FeatureEngine
from core.tdm_field_processor import TDMFieldProcessor
from core.kalman_filter import KalmanFilter
import joblib
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Temporalytics AI", layout="wide")

st.title("Temporalytics AI")
st.write("Cryptocurrency Price Prediction and Analysis")

# Initialize components
fetcher = MarketDataFetcher()
realtime_fetcher = RealtimePriceFetcher()
feature_engine = FeatureEngine()
tdm_processor = TDMFieldProcessor()
kalman = KalmanFilter()

# Load model and scaler
try:
    model = joblib.load("models/xgb_predictor.pkl")
    scaler = joblib.load("models/scaler.pkl")
except Exception as e:
    st.error(f"Failed to load model or scaler: {e}")
    st.stop()

# Sidebar for user inputs
st.sidebar.header("Input Parameters")
coin_id = st.sidebar.text_input("Coin ID", value="bitcoin")
vs_currency = st.sidebar.text_input("Currency", value="usd")
days = st.sidebar.number_input("Days of Historical Data", min_value=1, max_value=365, value=90)
analyze_button = st.sidebar.button("Analyze Market")

# Real-time price display
st.header("Real-Time Price")
try:
    price = realtime_fetcher.fetch_realtime_price(coin_id, vs_currency)
    if price is not None:
        st.metric(f"{coin_id.upper()} Price ({vs_currency.upper()})", f"${price:,.2f}")
    else:
        st.warning(f"Unable to fetch real-time price for {coin_id}.")
except Exception as e:
    st.error(f"Error fetching real-time price: {e}")

if analyze_button:
    st.header("Market Analysis")
    
    # Fetch historical data
    df = fetcher.fetch_historical_data(coin_id, vs_currency, days)
    if df is None or df.empty:
        st.error(f"Failed to fetch historical data for {coin_id}.")
        st.stop()
    
    # Apply Kalman filter
    df['smoothed_close'] = kalman.filter(df, 'close')
    
    # Compute technical indicators
    df = feature_engine.compute_technical_indicators(df)
    if df is None:
        st.error("Failed to compute technical indicators.")
        st.stop()
    
    # Compute TDM metrics
    df = tdm_processor.compute_tdm_metrics(df)
    if df is None:
        st.error("Failed to compute TDM metrics.")
        st.stop()
    
    # Prepare features for prediction
    feature_columns = ['sma_20', 'ema_12', 'rsi', 'stoch', 'atr', 'vwap', 'trend', 'momentum', 'conviction', 'stability']
    X = df[feature_columns].tail(1)
    X_scaled = scaler.transform(X)
    
    # Make prediction
    try:
        prediction = model.predict(X_scaled)[0]
        st.subheader("24-Hour Price Prediction")
        st.write(f"Predicted {coin_id.upper()} price in {vs_currency.upper()}: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
    
    # Plotting
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Price and Smoothed Price")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Price', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df.index, y=df['smoothed_close'], name='Smoothed Price', line=dict(color='orange')))
        fig.update_layout(xaxis_title="Date", yaxis_title=f"Price ({vs_currency.upper()})")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Technical Indicators")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')))
        fig.add_trace(go.Scatter(x=df.index, y=df['stoch'], name='Stochastic', line=dict(color='green')))
        fig.update_layout(xaxis_title="Date", yaxis_title="Value")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("TDM Metrics")
    col3, col4, col5, col6 = st.columns(4)
    with col3:
        st.metric("Trend", f"{df['trend'].iloc[-1]:.2f}")
    with col4:
        st.metric("Momentum", f"{df['momentum'].iloc[-1]:.4f}")
    with col5:
        st.metric("Conviction", f"{df['conviction'].iloc[-1]:.2f}")
    with col6:
        st.metric("Stability", f"{df['stability'].iloc[-1]:.4f}")
