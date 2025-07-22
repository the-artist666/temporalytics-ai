Temporalytics AI
A professional Streamlit app for cryptocurrency analysis, supporting 10 coins, advanced indicators, portfolio tracking, multi-scenario predictions, and a Grok-like financial advisor, with a grok.com-inspired design.
Setup Instructions

Clone the repository:git clone https://github.com/your-username/temporalytics-ai.git
cd temporalytics-ai


Create and activate a virtual environment:python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Install WeasyPrint dependencies (Ubuntu):sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0


Train models and generate data:python train.py


Run the app locally:streamlit run app.py



Deployment
Deploy to Streamlit Community Cloud:

Push to GitHub, including data/ and models/:git add .
git commit -m "Initial commit"
git push origin main


At share.streamlit.io, deploy app.py from main branch.
Set app URL (e.g., temporalytics-ai-yourname).
Reboot if updating.

Features

Supported Coins: Bitcoin, Ethereum, Solana, Binance Coin, XRP, Cardano, Dogecoin, Polkadot, Chainlink, Polygon.
Pre-Loaded Data: Historical data with indicators and predictions.
Indicators: RSI, Stochastic, MACD, Bollinger Bands, Ichimoku Cloud, Fibonacci, ADX, Volatility, Sharpe Ratio, Correlation.
Portfolio Tracker: Track holdings and total value.
Market Overview: Real-time prices and trends.
Predictions: Bullish, Neutral, Bearish scenarios with MAE and R².
Financial Advisor: AI-driven trading recommendations with risk management (stop-loss, take-profit, position sizing).
Export: Download analysis as PDF.
Design: Professional, grok.com-inspired UI.

Troubleshooting

API Errors: Skip xAI with API_KEYS[3]["calls"] = 10000 in API modules.
Data Missing: Run python train.py.
WeasyPrint Issues: Install dependencies or skip PDF export.

