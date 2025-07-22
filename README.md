Temporalytics AI
An AI-powered system for predicting cryptocurrency prices using XGBoost, Kalman filtering, and technical indicators, with real-time data from multiple APIs.
Setup Instructions

Clone the repository:git clone https://github.com/your-username/temporalytics-ai.git
cd temporalytics-ai


Create and activate a virtual environment:python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Train the model:python train.py


Run the Streamlit app locally:streamlit run app.py



Deployment
Deploy to Streamlit Community Cloud:

Push the repository to GitHub.
Sign up at share.streamlit.io.
Connect your GitHub account and deploy the app.py file from this repository.

Project Structure

api/: Contains market_data_fetcher.py for historical data and realtime_price_fetcher.py for real-time prices using Coinranking, Finnhub, Alpha Vantage, and xAI APIs.
core/: Includes feature_engine.py, tdm_field_processor.py, and kalman_filter.py for data processing.
models/: Stores trained model (xgb_predictor.pkl) and scaler (scaler.pkl).
data/: Stores fetched data (e.g., btcusd_1h.csv).
app.py: Streamlit dashboard for interactive analysis.
train.py: Script to train the XGBoost model.
requirements.txt: Lists required Python libraries.

Features

Real-Time Price: Fetches current prices using multiple APIs with automatic switching to avoid rate limits.
Historical Analysis: Visualizes price trends with smoothed data and indicators (RSI, Stochastic, ATR, VWAP).
Price Prediction: Uses XGBoost for 24-hour price forecasts.
Market Dynamics: Computes TDM metrics (trend, momentum, conviction, stability).

