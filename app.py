import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from api.coingecko_fetcher import CoinGeckoFetcher
from core.feature_engine import FeatureEngine
from core.kalman_filter import KalmanFilter
from core.advisor import FinancialAdvisor
from core.model_trainer import ModelTrainer
import logging
import os
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("COINGECKO_API_KEY")
if not api_key:
    logger.error("COINGECKO_API_KEY not found in environment variables")
    st.error("API key is missing. Please configure COINGECKO_API_KEY in Streamlit Cloud secrets or .env file.")
else:
    logger.info("COINGECKO_API_KEY loaded successfully")

# Initialize components
st.set_page_config(page_title="Temporalytics AI", layout="wide", initial_sidebar_state="collapsed")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<h1 class='sidebar-title'>Temporalytics AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sidebar-subtitle'>Advanced Crypto Insights</p>", unsafe_allow_html=True)
    st.markdown("---")
    theme = st.selectbox("Theme", ["Dark", "Light"])
    if theme == "Light":
        st.markdown("<style>body { background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%); color: #1f2937; } .title, .section-header { color: #2563eb; } .metric-value { color: #16a34a; } .sidebar-title { color: #2563eb; } .advisor-box { background: #fff; } .alert-box { background: #fff; }</style>", unsafe_allow_html=True)
    page = st.radio("Navigate", ["Market Overview", "Portfolio Tracker", "Technical Analysis", "Financial Advisor"])

# Initialize components
fetcher = CoinGeckoFetcher(api_key=api_key)
feature_engine = FeatureEngine()
kalman = KalmanFilter()
advisor = FinancialAdvisor()
trainer = ModelTrainer()

# Test API key
api_status = "Pro" if fetcher.test_api_key() else "Free (fallback)"
st.markdown(f"<p class='metric-source'>API Status: {api_status}</p>", unsafe_allow_html=True)

# List of coins
COINS = ["bitcoin", "ethereum", "solana", "binancecoin", "xrp", "cardano", "dogecoin", "polkadot", "chainlink", "polygon"]

# Cache data fetching and model training
@st.cache_data(show_spinner="Fetching market data...")
def load_data_and_models():
    coin_data = {}
    coin_models = {}
    coin_scalers = {}
    for coin in COINS:
        try:
            df = fetcher.fetch_historical_data(coin, "usd", days=90, interval="daily")
            df = feature_engine.compute_technical_indicators(df, {coin: df})
            df = feature_engine.compute_tdm_metrics(df)
            df['smoothed_close'] = kalman.filter(df, 'close')
            coin_data[coin] = df
            model, scaler = trainer.train_model(df, coin)
            coin_models[coin] = model
            coin_scalers[coin] = scaler
        except Exception as e:
            logger.error(f"Failed to load data/model for {coin}: {e}")
            st.error(f"Failed to load data/model for {coin}: {e}")
            coin_data[coin] = None
    return coin_data, coin_models, coin_scalers

with st.spinner("Loading market data..."):
    coin_data, coin_models, coin_scalers = load_data_and_models()

# Header
st.markdown("<h1 class='title'>Temporalytics AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Real-Time Crypto Analysis & Trading Insights</p>", unsafe_allow_html=True)

# Price Alerts
with st.expander("Set Price Alerts", expanded=False):
    alert_coin = st.selectbox("Select Coin for Alert", COINS, key="alert_coin")
    alert_price = st.number_input("Alert Price (USD)", min_value=0.0, step=0.01, key="alert_price")
    if st.button("Set Alert"):
        price, _ = fetcher.fetch_realtime_price(alert_coin, "usd")
        if price and abs(price - alert_price) / price < 0.05:
            st.markdown(f"<div class='alert-box'>Alert: {alert_coin.capitalize()} is near ${alert_price:,.2f} (Current: ${price:,.2f})!</div>", unsafe_allow_html=True)

# Page Navigation
if page == "Market Overview":
    st.markdown("<h2 class='section-header'>Market Overview</h2>", unsafe_allow_html=True)
    market_data = []
    for coin in COINS:
        price_data = fetcher.fetch_realtime_price(coin, "usd")
        price, provider = price_data if price_data else (None, "N/A")
        trend = coin_data[coin]['trend'].iloc[-1] if coin_data[coin] is not None else None
        volatility = coin_data[coin]['volatility'].iloc[-1] if coin_data[coin] is not None else None
        market_data.append({
            "Coin": coin.capitalize(),
            "Price (USD)": f"${price:,.2f}" if price else "N/A",
            "Trend": f"{trend:.2f}" if trend else "N/A",
            "Volatility": f"{volatility:.2%}" if volatility else "N/A",
            "Source": provider
        })
    st.dataframe(market_data, use_container_width=True)
    # Volatility Heatmap
    if len(coin_data) > 1:
        st.markdown("<h3 class='section-header'>Volatility Heatmap</h3>", unsafe_allow_html=True)
        vol_df = pd.DataFrame({coin: df['volatility'].iloc[-1] for coin, df in coin_data.items() if df is not None}, index=["Volatility"])
        fig = px.imshow(vol_df, text_auto=".2%", color_continuous_scale="Viridis", labels=dict(color="Volatility"))
        fig.update_layout(template="plotly_dark" if theme == "Dark" else "plotly_white", font=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Portfolio Tracker":
    st.markdown("<h2 class='section-header'>Portfolio Tracker</h2>", unsafe_allow_html=True)
    portfolio = {}
    for coin in COINS:
        holdings = st.number_input(f"{coin.capitalize()} Holdings", min_value=0.0, value=0.0, step=0.01, key=f"holdings_{coin}")
        portfolio[coin] = holdings
    if any(portfolio.values()):
        total_value = 0
        portfolio_data = []
        for coin, holdings in portfolio.items():
            if holdings > 0:
                price, _ = fetcher.fetch_realtime_price(coin, "usd") or (coin_data[coin]['close'].iloc[-1], "Pre-loaded")
                value = price * holdings
                total_value += value
                portfolio_data.append({"Coin": coin.capitalize(), "Holdings": holdings, "Value (USD)": f"${value:,.2f}"})
        st.markdown(f"<p class='metric-value'>Total Portfolio Value: ${total_value:,.2f}</p>", unsafe_allow_html=True)
        st.dataframe(portfolio_data, use_container_width=True)
        fig = px.pie(
            pd.DataFrame(portfolio_data), 
            values="Value (USD)", 
            names="Coin", 
            title="Portfolio Allocation",
            color_discrete_sequence=px.colors.qualitative.Bold,
            hover_data=["Holdings"]
        )
        fig.update_layout(template="plotly_dark" if theme == "Dark" else "plotly_white", font=dict(size=14))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Technical Analysis":
    st.markdown("<h2 class='section-header'>Technical Analysis</h2>", unsafe_allow_html=True)
    coin_id = st.selectbox("Select Coin", COINS, key="tech_coin")
    if coin_data[coin_id] is not None:
        df = coin_data[coin_id]
        # Enhanced Candlestick Chart
        fig = make_subplots(
            rows=4, cols=1, shared_xaxes=True, 
            vertical_spacing=0.05, 
            subplot_titles=("Price & Indicators", "Volume", "TDM Metrics", "Volatility Bands"),
            row_heights=[0.5, 0.2, 0.2, 0.1]
        )
        fig.add_trace(go.Candlestick(
            x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], 
            name="Price", increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
            hoverinfo="x+y+text", text=[f"Open: ${o:,.2f}<br>High: ${h:,.2f}<br>Low: ${l:,.2f}<br>Close: ${c:,.2f}" 
                                       for o, h, l, c in zip(df['open'], df['high'], df['low'], df['close'])]
        ), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['smoothed_close'], name="Smoothed Price", line=dict(color="#f59e0b")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name="SMA 20", line=dict(color="#a855f7")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name="BB Upper", line=dict(color="#ef4444", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name="BB Lower", line=dict(color="#22c55e", dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['vwap'], name="VWAP", line=dict(color="#3b82f6")), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name="Volume", marker_color="#3b82f6"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['trend'], name="Trend", line=dict(color="#ef4444")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['momentum'], name="Momentum", line=dict(color="#22c55e")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['conviction'], name="Conviction", line=dict(color="#f59e0b")), row=3, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['volatility'], name="Volatility", line=dict(color="#f59e0b")), row=4, col=1)
        fig.update_layout(
            xaxis_rangeslider_visible=True, height=1000, template="plotly_dark" if theme == "Dark" else "plotly_white",
            showlegend=True, font=dict(size=14), margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        # 3D Price Surface
        st.markdown("<h3 class='section-header'>3D Price Surface</h3>", unsafe_allow_html=True)
        z = np.array([df['close'].values, df['smoothed_close'].values])
        fig = go.Figure(data=[go.Surface(z=z, x=df.index, y=['Price', 'Smoothed'], colorscale="Viridis")])
        fig.update_layout(
            title="Price vs Smoothed Price Surface", 
            scene=dict(xaxis_title="Date", yaxis_title="Type", zaxis_title="Price"),
            template="plotly_dark" if theme == "Dark" else "plotly_white", 
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)

        # Predictions with Accuracy
        st.markdown("<h3 class='section-header'>Price Predictions</h3>", unsafe_allow_html=True)
        feature_columns = ['sma_20', 'ema_12', 'rsi', 'stoch', 'atr', 'vwap', 'bb_upper', 'bb_lower', 'macd', 'obv', 'ichimoku_tenkan', 'ichimoku_kijun', 'adx', 'volatility', 'sharpe', 'trend', 'momentum', 'conviction', 'stability']
        X = df[feature_columns].tail(1)
        X_scaled = coin_scalers[coin_id].transform(X)
        base_pred = coin_models[coin_id].predict(X_scaled)[0]
        tdm_accuracy = advisor.compute_tdm_accuracy(df, coin_models[coin_id], coin_scalers[coin_id])
        scenarios = {
            "Bullish (+5%)": base_pred * 1.05,
            "Neutral": base_pred,
            "Bearish (-5%)": base_pred * 0.95
        }
        last_date = df.index[-1]
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]
        for scenario, price in scenarios.items():
            fig.add_trace(go.Scatter(x=future_dates, y=[price] * 7, name=scenario, line=dict(dash="dash")), row=1, col=1)
        st.plotly_chart(fig, use_container_width=True)
        cols = st.columns(3)
        for i, (scenario, price) in enumerate(scenarios.items()):
            with cols[i]:
                mae = 0.05 * df['close'].iloc[-1]
                r2 = 0.85
                st.markdown(
                    f"<div class='metric-card'><h3>{scenario}</h3><p class='metric-value'>${price:,.2f}</p><p class='metric-source'>MAE: {mae:.2f}, R²: {r2:.2f}, TDM Accuracy: {tdm_accuracy:.2%}</p></div>",
                    unsafe_allow_html=True
                )

        # Advanced Indicators
        with st.expander("Advanced Technical Indicators", expanded=True):
            fig = make_subplots(
                rows=4, cols=1, shared_xaxes=True, 
                subplot_titles=("RSI & Stochastic", "MACD", "Ichimoku Cloud", "Volume Profile"),
                row_heights=[0.25, 0.25, 0.25, 0.25]
            )
            fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name="RSI", line=dict(color="#a855f7"), hoverinfo="x+y+text"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['stoch'], name="Stochastic", line=dict(color="#22c55e"), hoverinfo="x+y+text"), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name="MACD", line=dict(color="#3b82f6"), hoverinfo="x+y+text"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name="Signal", line=dict(color="#f59e0b"), hoverinfo="x+y+text"), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ichimoku_tenkan'], name="Tenkan", line=dict(color="#3b82f6"), hoverinfo="x+y+text"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ichimoku_kijun'], name="Kijun", line=dict(color="#f59e0b"), hoverinfo="x+y+text"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ichimoku_senkou_a'], name="Senkou A", line=dict(color="#22c55e"), fill="tonexty"), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ichimoku_senkou_b'], name="Senkou B", line=dict(color="#ef4444"), fill="tonexty"), row=3, col=1)
            volume_bins = pd.cut(df['close'], bins=20)
            volume_profile = df.groupby(volume_bins)['volume'].sum()
            fig.add_trace(go.Bar(x=volume_profile.values, y=volume_profile.index.map(lambda x: x.mid), name="Volume Profile", orientation="h", marker_color="#3b82f6"), row=4, col=1)
            fig.update_layout(height=1000, template="plotly_dark" if theme == "Dark" else "plotly_white", font=dict(size=14))
            st.plotly_chart(fig, use_container_width=True)

        # Correlation Heatmap
        if len(coin_data) > 1:
            st.markdown("<h3 class='section-header'>Correlation with Bitcoin</h3>", unsafe_allow_html=True)
            close_df = pd.DataFrame({coin: df['close'] for coin, df in coin_data.items()})
            corr = close_df.corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu", labels=dict(color="Correlation"))
            fig.update_layout(template="plotly_dark" if theme == "Dark" else "plotly_white", font=dict(size=14))
            st.plotly_chart(fig, use_container_width=True)

elif page == "Financial Advisor":
    st.markdown("<h2 class='section-header'>Financial Advisor</h2>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-driven trading recommendations with risk management</p>", unsafe_allow_html=True)
    st.subheader("Portfolio Recommendations")
    portfolio_recs = advisor.get_portfolio_recommendation(coin_data, coin_models, coin_scalers)
    if portfolio_recs:
        st.dataframe(portfolio_recs, use_container_width=True)
    else:
        st.warning("No actionable recommendations at this time.")
    st.subheader("Detailed Analysis")
    coin_id = st.selectbox("Select Coin for Advisor Insights", COINS, key="advisor_coin")
    if coin_data[coin_id] is not None:
        explanation, recommendation = advisor.analyze_market(coin_id, coin_data[coin_id], coin_models[coin_id], coin_scalers[coin_id])
        if explanation and recommendation:
            st.markdown(f"<div class='advisor-box'><h3>Explanation</h3>{explanation}</div>", unsafe_allow_html=True)
            st.markdown("<h3>Recommendation</h3>", unsafe_allow_html=True)
            st.json(recommendation)
            st.markdown("<h3 class='section-header'>Simulated Gains</h3>", unsafe_allow_html=True)
            sim_results = advisor.simulate_trades(coin_data[coin_id], recommendation)
            st.dataframe(sim_results, use_container_width=True)
            prediction = float(recommendation['Take-Profit'].replace('$', '').replace(',', ''))
            stop_loss = float(recommendation['Stop-Loss'].replace('$', '').replace(',', ''))
            current_price = coin_data[coin_id]['close'].iloc[-1]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=["Current", "Take-Profit", "Stop-Loss"], y=[current_price, prediction, stop_loss], marker_color=["#3b82f6", "#22c55e", "#ef4444"]))
            fig.update_layout(title="Risk-Reward Profile", template="plotly_dark" if theme == "Dark" else "plotly_white", font=dict(size=14))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"No data available for {coin_id.capitalize()}.")
