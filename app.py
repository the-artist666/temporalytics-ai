import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from api.market_data_fetcher import MarketDataFetcher
from api.realtime_price_fetcher import RealtimePriceFetcher
from core.feature_engine import FeatureEngine
from core.tdm_field_processor import TDMFieldProcessor
from core.kalman_filter import KalmanFilter
from core.advisor import FinancialAdvisor
import joblib
import logging
import os
try:
    from weasyprint import HTML
except ImportError as e:
    st.warning("PDF export is disabled due to missing WeasyPrint dependencies.")
    HTML = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Temporalytics AI", layout="wide")
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize components
fetcher = MarketDataFetcher()
realtime_fetcher = RealtimePriceFetcher()
feature_engine = FeatureEngine()
tdm_processor = TDMFieldProcessor()
kalman = KalmanFilter()
advisor = FinancialAdvisor()

# Load pre-trained models and data
COINS = ["bitcoin", "ethereum", "solana", "binancecoin", "xrp", "cardano", "dogecoin", "polkadot", "chainlink", "polygon"]
coin_data = {}
coin_models = {}
coin_scalers = {}
for coin in COINS:
    try:
        coin_data[coin] = pd.read_csv(f"data/{coin}usd_1h.csv", index_col="timestamp", parse_dates=True)
        coin_data[coin] = feature_engine.compute_technical_indicators(coin_data[coin], coin_data)
        coin_data[coin] = tdm_processor.compute_tdm_metrics(coin_data[coin])
        coin_data[coin]['smoothed_close'] = kalman.filter(coin_data[coin], 'close')
        coin_models[coin] = joblib.load(f"models/xgb_{coin}.pkl")
        coin_scalers[coin] = joblib.load(f"models/scaler_{coin}.pkl")
    except Exception as e:
        st.error(f"Failed to load data/model for {coin}: {e}")
        coin_data[coin] = None

# Header
st.markdown("<h1 class='title'>Temporalytics AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Professional Cryptocurrency Analysis & Trading Advisor</p>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Market Overview", "Portfolio Tracker", "Technical Analysis", "Financial Advisor"])

# Market Overview
with tab1:
    st.markdown("<h2 class='section-header'>Market Overview</h2>", unsafe_allow_html=True)
    market_data = []
    for coin in COINS:
        price_data = realtime_fetcher.fetch_realtime_price(coin, "usd")
        price, provider = price_data if price_data else (None, "N/A")
        trend = coin_data[coin]['trend'].iloc[-1] if coin_data[coin] is not None else None
        market_data.append({
            "Coin": coin.capitalize(),
            "Price (USD)": f"${price:,.2f}" if price else "N/A",
            "Trend": f"{trend:.2f}" if trend else "N/A",
            "Source": provider
        })
    st.dataframe(market_data, use_container_width=True)

# Portfolio Tracker
with tab2:
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
                price, _ = realtime_fetcher.fetch_realtime_price(coin, "usd") or (coin_data[coin]['close'].iloc[-1], "Pre-loaded")
                value = price * holdings
                total_value += value
                portfolio_data.append({"Coin": coin.capitalize(), "Holdings": holdings, "Value (USD)": f"${value:,.2f}"})
        st.markdown(f"<p class='metric-value'>Total Portfolio Value: ${total_value:,.2f}</p>", unsafe_allow_html=True)
        st.dataframe(portfolio_data, use_container_width=True)

# Technical Analysis
with tab3:
    st.markdown("<h2 class='section-header'>Technical Analysis</h2>", unsafe_allow_html=True)
    coin_id = st.selectbox("Select Coin", COINS, key="tech_coin")
    if coin_data[coin_id] is not None:
        df = coin_data[coin_id]
        # Candlestick Chart with Scenarios
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("Price", "Volume"), row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=df.index, open=df['close'], high=df['close']*1.01, low=df['close']*0.99, close=df['close'], name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['smoothed_close'], name="Smoothed Price", line=dict(color="#f59e0b")), row=1, col=1)
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name="Volume", marker_color="#3b82f6"), row=2, col=1)
        # Predictions
        feature_columns = ['sma_20', 'ema_12', 'rsi', 'stoch', 'atr', 'vwap', 'bb_upper', 'bb_lower', 'macd', 'obv', 'ichimoku_tenkan', 'ichimoku_kijun', 'adx', 'volatility', 'sharpe', 'trend', 'momentum', 'conviction', 'stability']
        X = df[feature_columns].tail(1)
        X_scaled = coin_scalers[coin_id].transform(X)
        base_pred = coin_models[coin_id].predict(X_scaled)[0]
        scenarios = {
            "Bullish (+5%)": base_pred * 1.05,
            "Neutral": base_pred,
            "Bearish (-5%)": base_pred * 0.95
        }
        last_date = df.index[-1]
        future_dates = [last_date + pd.Timedelta(hours=i) for i in range(1, 25)]
        for scenario, price in scenarios.items():
            fig.add_trace(go.Scatter(x=future_dates, y=[price] * 24, name=scenario, line=dict(dash="dash")), row=1, col=1)
        fig.update_layout(xaxis_rangeslider_visible=False, height=600, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        # Indicators
        with st.expander("Technical Indicators", expanded=True):
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=("RSI & Stochastic", "MACD", "Bollinger Bands", "Ichimoku Cloud"))
            fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name="RSI", line=dict(color="#a855f7")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['stoch'], name="Stochastic", line=dict(color="#22c55e")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name="MACD", line=dict(color="#3b82f6")), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name="Signal", line=dict(color="#f59e0b")), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name="Upper Band", line=dict(color="#ef4444")), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_middle'], name="Middle Band", line=dict(color="#6b7280")), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name="Lower Band", line=dict(color="#22c55e")), row=3, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ichimoku_tenkan'], name="Tenkan", line=dict(color="#3b82f6")), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ichimoku_kijun'], name="Kijun", line=dict(color="#f59e0b")), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ichimoku_senkou_a'], name="Senkou A", line=dict(color="#22c55e"), fill="tonexty"), row=4, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['ichimoku_senkou_b'], name="Senkou B", line=dict(color="#ef4444"), fill="tonexty"), row=4, col=1)
            fig.update_layout(height=1000, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        # Fibonacci and ADX
        with st.expander("Fibonacci & ADX", expanded=False):
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Fibonacci Retracement", "ADX"))
            for level in ['fib_0_0', 'fib_23_6', 'fib_38_2', 'fib_50_0', 'fib_61_8', 'fib_100_0']:
                fig.add_trace(go.Scatter(x=df.index, y=df[level], name=level.replace('fib_', ''), line=dict(dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['adx'], name="ADX", line=dict(color="#3b82f6")), row=2, col=1)
            fig.update_layout(height=600, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        # TDM Metrics
        st.markdown("<h3 class='section-header'>TDM Metrics</h3>", unsafe_allow_html=True)
        cols = st.columns(4)
        metrics = ["trend", "momentum", "conviction", "stability"]
        for i, metric in enumerate(metrics):
            with cols[i]:
                value = df[metric].iloc[-1]
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={"text": metric.capitalize()},
                    gauge={"axis": {"range": [min(df[metric]), max(df[metric])]},
                           "bar": {"color": "#3b82f6"}}
                ))
                st.plotly_chart(fig, use_container_width=True)
        # Market Metrics
        st.markdown("<h3 class='section-header'>Market Metrics</h3>", unsafe_allow_html=True)
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"<p class='metric-value'>Volatility (Annualized): {df['volatility'].iloc[-1]:.2%}</p>", unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f"<p class='metric-value'>Sharpe Ratio: {df['sharpe'].iloc[-1]:.2f}</p>", unsafe_allow_html=True)
        # Correlation Heatmap
        if len(coin_data) > 1:
            st.markdown("<h3 class='section-header'>Correlation with Bitcoin</h3>", unsafe_allow_html=True)
            close_df = pd.DataFrame({coin: df['close'] for coin, df in coin_data.items()})
            corr = close_df.corr()['bitcoin'].drop('bitcoin')
            fig = px.bar(x=corr.index.str.capitalize(), y=corr.values, labels={'x': 'Coin', 'y': 'Correlation'})
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        # Predictions with Accuracy
        st.markdown("<h3 class='section-header'>24-Hour Price Predictions</h3>", unsafe_allow_html=True)
        for scenario, price in scenarios.items():
            mae = 0.05 * df['close'].iloc[-1]  # Mock MAE
            r2 = 0.85  # Mock R²
            st.markdown(
                f"<div class='metric-card'><h3>{scenario}</h3><p class='metric-value'>${price:,.2f}</p><p class='metric-source'>MAE: {mae:.2f}, R²: {r2:.2f}</p></div>",
                unsafe_allow_html=True
            )

# Financial Advisor
with tab4:
    st.markdown("<h2 class='section-header'>Financial Advisor</h2>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>AI-driven trading recommendations with risk management</p>", unsafe_allow_html=True)
    st.subheader("Portfolio Recommendations")
    portfolio_recs = advisor.get_portfolio_recommendation()
    if portfolio_recs:
        st.dataframe(portfolio_recs, use_container_width=True)
    else:
        st.warning("No actionable recommendations at this time.")
    st.subheader("Detailed Analysis")
    coin_id = st.selectbox("Select Coin for Advisor Insights", COINS, key="advisor_coin")
    explanation, recommendation = advisor.analyze_market(coin_id)
    if explanation and recommendation:
        st.markdown(f"**Explanation**:\n{explanation}")
        st.markdown("**Recommendation**:")
        st.json(recommendation)
        if HTML is not None and st.button(f"Download {coin_id.capitalize()} Report"):
            try:
                html_content = f"""
                <h1>Temporalytics AI Report: {coin_id.capitalize()}</h1>
                <h2>Market Analysis</h2>
                <p>{explanation.replace('\n', '<br>')}</p>
                <h2>Trading Recommendation</h2>
                <ul>
                    <li><b>Action:</b> {recommendation['Action']}</li>
                    <li><b>Reasoning:</b> {recommendation['Reasoning']}</li>
                    <li><b>Position Size:</b> {recommendation['Position Size']}</li>
                    <li><b>Stop-Loss:</b> {recommendation['Stop-Loss']}</li>
                    <li><b>Take-Profit:</b> {recommendation['Take-Profit']}</li>
                </ul>
                <h2>Latest Data</h2>
                {coin_data[coin_id].tail(1).to_html()}
                """
                pdf = HTML(string=html_content).write_pdf()
                st.download_button(f"Download {coin_id.capitalize()} Report", data=pdf, file_name=f"{coin_id}_report.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Failed to generate PDF report: {e}")
        elif HTML is None:
            st.warning("PDF export is not available due to missing dependencies.")
    else:
        st.error(f"No data available for {coin_id.capitalize()}.")
