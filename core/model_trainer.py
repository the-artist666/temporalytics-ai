from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import joblib
import os

class ModelTrainer:
    def train_model(self, df, coin):
        model_path = f"models/xgb_{coin}.pkl"
        scaler_path = f"models/scaler_{coin}.pkl"
        os.makedirs('models', exist_ok=True)
        feature_columns = ['sma_20', 'ema_12', 'rsi', 'stoch', 'atr', 'vwap', 'bb_upper', 'bb_lower', 'macd', 'obv', 'ichimoku_tenkan', 'ichimoku_kijun', 'adx', 'volatility', 'sharpe', 'trend', 'momentum', 'conviction', 'stability']
        X = df[feature_columns].dropna()
        y = df['close'].shift(-1).dropna()
        X = X.iloc[:len(y)]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        model.fit(X_scaled, y)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        return model, scaler
