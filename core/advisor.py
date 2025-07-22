import pandas as pd
import numpy as np
import joblib
import os
import logging

logger = logging.getLogger(__name__)

class FinancialAdvisor:
    def __init__(self, data_dir="data", model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.coins = [
            "bitcoin", "ethereum", "solana", "binancecoin", "xrp",
            "cardano", "dogecoin", "polkadot", "chainlink", "polygon"
        ]
        self.data = {}
        self.models = {}
        self.scalers = {}
        self.load_data_and_models()

    def load_data_and_models(self):
        for coin in self.coins:
            try:
                csv_path = os.path.join(self.data_dir, f"{coin}usd_1h.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, index_col="timestamp", parse_dates=True)
                    self.data[coin] = df
                    model_path = os.path.join(self.model_dir, f"xgb_{coin}.pkl")
                    scaler_path = os.path.join(self.model_dir, f"scaler_{coin}.pkl")
                    if os.path.exists(model_path) and os.path.exists(scaler_path):
                        self.models[coin] = joblib.load(model_path)
                        self.scalers[coin] = joblib.load(scaler_path)
                    else:
                        logger.warning(f"Model or scaler not found for {coin}")
                else:
                    logger.warning(f"Data not found for {coin}")
            except Exception as e:
                logger.error(f"Failed to load data/model for {coin}: {e}")

    def analyze_market(self, coin):
        df = self.data.get(coin)
        if df is None or df.empty:
            logger.error(f"No data available for {coin}")
            return None, None
        latest = df.iloc[-1]
        indicators = {
            "Price": latest["close"],
            "RSI": latest["rsi"],
            "MACD": latest["macd"],
            "ADX": latest["adx"],
            "Volatility (ATR)": latest["atr"],
            "Trend": latest["trend"],
            "Momentum": latest["momentum"],
            "Conviction": latest["conviction"],
            "Stability": latest["stability"]
        }
        explanation = self.generate_explanation(coin, indicators)
        recommendation = self.generate_recommendation(coin, indicators)
        return explanation, recommendation

    def generate_explanation(self, coin, indicators):
        explanations = [f"**{coin.capitalize()} Market Analysis**"]
        price = indicators["Price"]
        explanations.append(f"- **Current Price**: ${price:,.2f}")
        rsi = indicators["RSI"]
        if rsi > 70:
            explanations.append("- **RSI**: Overbought ({rsi:.1f}), suggesting a potential price correction.")
        elif rsi < 30:
            explanations.append("- **RSI**: Oversold ({rsi:.1f}), indicating a possible price rebound.")
        else:
            explanations.append("- **RSI**: Neutral ({rsi:.1f}), showing stable momentum.")
        macd = indicators["MACD"]
        if macd > 0:
            explanations.append("- **MACD**: Positive ({macd:.2f}), indicating bullish momentum.")
        else:
            explanations.append("- **MACD**: Negative ({macd:.2f}), suggesting bearish momentum.")
        adx = indicators["ADX"]
        if adx > 25:
            explanations.append("- **ADX**: Strong trend ({adx:.1f}), suggesting reliable price movement.")
        else:
            explanations.append("- **ADX**: Weak trend ({adx:.1f}), advising caution.")
        trend = indicators["Trend"]
        if trend > 0:
            explanations.append("- **Trend**: Upward ({trend:.2f}), indicating positive price direction.")
        else:
            explanations.append("- **Trend**: Downward ({trend:.2f}), suggesting negative price direction.")
        return "\n".join(explanations)

    def generate_recommendation(self, coin, indicators):
        price = indicators["Price"]
        atr = indicators["Volatility (ATR)"]
        rsi = indicators["RSI"]
        macd = indicators["MACD"]
        adx = indicators["ADX"]
        trend = indicators["Trend"]
        stop_loss_pct = 0.05  # 5% stop-loss
        take_profit_pct = 0.10  # 10% take-profit
        volatility_threshold = np.percentile(self.data[coin]["atr"].dropna(), 75)
        position_size = min(0.1 / (atr / price), 0.1)  # Max 10% of portfolio
        if rsi < 30 and macd > 0 and adx > 25 and trend > 0:
            action = "Buy"
            stop_loss = price * (1 - stop_loss_pct)
            take_profit = price * (1 + take_profit_pct)
            reasoning = f"Buy {coin.capitalize()} due to oversold RSI, bullish MACD, strong trend (ADX), and positive TDM trend."
        elif rsi > 70 or macd < 0 or trend < 0:
            action = "Sell"
            stop_loss = price * (1 + stop_loss_pct)
            take_profit = price * (1 - take_profit_pct)
            reasoning = f"Sell {coin.capitalize()} due to overbought RSI, bearish MACD, or negative TDM trend."
        else:
            action = "Hold"
            stop_loss = None
            take_profit = None
            reasoning = f"Hold {coin.capitalize()} as indicators show mixed signals or weak trend."
        if atr > volatility_threshold:
            action = "Hold"
            reasoning = f"Avoid trading {coin.capitalize()} due to high volatility (ATR)."
        recommendation = {
            "Action": action,
            "Reasoning": reasoning,
            "Position Size": f"{position_size*100:.1f}% of portfolio",
            "Stop-Loss": f"${stop_loss:,.2f}" if stop_loss else "N/A",
            "Take-Profit": f"${take_profit:,.2f}" if take_profit else "N/A"
        }
        return recommendation

    def get_portfolio_recommendation(self):
        portfolio = []
        total_weight = 0
        for coin in self.coins:
            _, rec = self.analyze_market(coin)
            if rec and rec["Action"] != "Hold":
                weight = float(rec["Position Size"].split("%")[0]) / 100
                total_weight += weight
                portfolio.append({**rec, "Coin": coin.capitalize()})
        if total_weight > 1:
            for rec in portfolio:
                rec["Position Size"] = f"{float(rec['Position Size'].split('%')[0]) / total_weight:.1f}% of portfolio"
        return portfolio
