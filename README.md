Temporalytics AI
A Streamlit-based cryptocurrency trading advisor with real-time data from CoinGecko's Pro API, advanced technical analysis, and AI-driven trading recommendations with simulated gains.
Installation

Clone the repository:
git clone https://github.com/your-username/temporalytics-ai.git
cd temporalytics-ai


Create a .env file with your CoinGecko API key:
COINGECKO_API_KEY=CG-mCypHSj3Ci4JpGH96VmHoayY


Create a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt


Run the app:
streamlit run app.py



Deployment on Streamlit Cloud

Push the repository to GitHub.
Add your CoinGecko API key as a secret in Streamlit Cloud:
Go to "Manage app" > "Secrets" and add:COINGECKO_API_KEY = "CG-mCypHSj3Ci4JpGH96VmHoayY"




In Streamlit Cloud, create a new app, link to your repository, and enable "Use custom Dockerfile."
Reboot the app to deploy.

Features

Real-time data via CoinGecko Pro API with authenticated requests.
TDM hybrid metrics (Trend, Direction, Momentum) with accuracy.
Enhanced visualizations: interactive candlestick charts, volume profiles, correlation heatmaps, risk-reward charts.
Short- and long-term trading recommendations with simulated gains.
Risk management: 5% stop-loss, 10% take-profit, max 10% position size.
Modern UI inspired by TradingView with sidebar navigation and dynamic charts.
