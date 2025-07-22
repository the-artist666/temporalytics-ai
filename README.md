Temporalytics AI
A Streamlit-based cryptocurrency trading advisor that fetches real-time data from CoinGecko's Pro API, provides advanced technical analysis, and offers AI-driven trading recommendations with simulated gains.
Installation

Clone the repository:
git clone https://github.com/your-username/temporalytics-ai.git
cd temporalytics-ai


Create a .env file with your CoinGecko API key:
COINGECKO_API_KEY=your-api-key-here


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
Go to "Manage app" > "Secrets" and add:COINGECKO_API_KEY = "your-api-key-here"




In Streamlit Cloud, create a new app, link to your repository, and enable "Use custom Dockerfile."
Reboot the app to deploy.

Features

Real-time data via CoinGecko Pro API.
TDM hybrid metrics (Trend, Direction, Momentum) with accuracy.
Advanced visualizations (candlestick, volume profile, correlation heatmap).
Short- and long-term trading recommendations with simulated gains.
Risk management: 5% stop-loss, 10% take-profit, max 10% position size.
