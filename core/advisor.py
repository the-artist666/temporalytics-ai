import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

class FinancialAdvisor:
    def compute_tdm_accuracy(self, df, model, scaler):
        feature_columns = ['sma_20', 'ema_12', 'rsi', 'stoch', 'atr', 'vwap', 'bb_upper', 'bb_lower', 'macd', 'obv', 'ichimoku_tenkan', 'ichimoku_kijun', 'adx', 'volatility', 'sharpe', 'trend', 'momentum', 'conviction', 'stability']
        X = df[feature_columns].dropna()
        y = df['close'].shift(-1).dropna()
        X = X.iloc[:len(y)]
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        tdm_signals = np.where(df['trend'].iloc[:len(y)] > 0, 1, -1)
        correct_signals = np.sum(np.sign(y_pred - X['close']) == np.sign(y - X['close']))
        return correct_signals / len(y)

    def analyze_market(self, coin_id, df, model, scaler):
        try:
            latest_data = df.iloc[-1]
            feature_columns = ['sma_20', 'ema_12', 'rsi', 'stoch', 'atr', 'vwap', 'bb_upper', 'bb_lower', 'macd', 'obv', 'ichimoku_tenkan', 'ichimoku_kijun', 'adx', 'volatility', 'sharpe', 'trend', 'momentum', 'conviction', 'stability']
            X = df[feature_columns].tail(1)
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
            action = "Buy" if latest_data['trend'] > 0 and latest_data['rsi'] < 70 else "Sell" if latest_data['trend'] < 0 and latest_data['rsi'] > 30 else "Hold"
            explanation = f"""
            **{coin_id.capitalize()} Analysis**:
            - **Trend**: {'Uptrend' if latest_data['trend'] > 0 else 'Downtrend'} (SMA20: {latest_data['sma_20']:.2f}, EMA12: {latest_data['ema_12']:.2f}).
            - **Momentum**: {latest_data['momentum']:.2%} (5-period ROC).
            - **Conviction**: RSI at {latest_data['rsi']:.2f}, indicating {'overbought' if latest_data['rsi'] > 70 else 'oversold' if latest_data['rsi'] < 30 else 'neutral'} conditions.
            - **Volatility**: {latest_data['volatility']:.2%} (annualized).
            - **Prediction**: Next 24-hour price predicted at ${prediction:,.2f}.
            """
            recommendation = {
                "Action": action,
                "Reasoning": f"Based on TDM metrics and technical indicators.",
                "Position Size": f"{min(0.1, 1 / (1 + latest_data['volatility'])):.2%} of portfolio",
                "Stop-Loss": f"${prediction * 0.95:,.2f} (5% below prediction)",
                "Take-Profit": f"${prediction * 1.10:,.2f} (10% above prediction)"
            }
            return explanation, recommendation
        except Exception as e:
            print(f"Error analyzing {coin_id}: {e}")
            return None, None

    def get_portfolio_recommendation(self, coin_data, coin_models, coin_scalers):
        recs = []
        for coin in coin_data:
            if coin_data[coin] is not None:
                _, rec = self.analyze_market(coin, coin_data[coin], coin_models[coin], coin_scalers[coin])
                if rec:
                    recs.append({"Coin": coin.capitalize(), **rec})
        return pd.DataFrame(recs) if recs else None

    def simulate_trades(self, df, recommendation):
        prediction = float(recommendation['Take-Profit'].replace('$', '').replace(',', ''))
        stop_loss = float(recommendation['Stop-Loss'].replace('$', '').replace(',', ''))
        current_price = df['close'].iloc[-1]
        position_size = 0.1  # 10% of portfolio
        short_term_gain = (prediction - current_price) / current_price * position_size * 100 if recommendation['Action'] == "Buy" else (current_price - prediction) / current_price * position_size * 100
        long_term_gain = short_term_gain * 1.5  # Assume 50% higher for 7 days
        return pd.DataFrame({
            "Horizon": ["Short-Term (24h)", "Long-Term (7d)"],
            "Simulated Gain (%)": [f"{short_term_gain:.2f}%", f"{long_term_gain:.2f}%"],
            "Stop-Loss": [f"${stop_loss:,.2f}", f"${stop_loss * 0.98:,.2f}"],
            "Take-Profit": [f"${prediction:,.2f}", f"${prediction * 1.05:,.2f}"]
        })
