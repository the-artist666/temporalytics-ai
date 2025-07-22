import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class FeatureEngine:
    def __init__(self):
        pass

    def compute_technical_indicators(self, df: pd.DataFrame, coin_dfs: Dict[str, pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        try:
            if df is None or df.empty:
                logger.error("Input DataFrame is None or empty.")
                return None

            df = df.copy()
            df['returns'] = df['close'].pct_change()

            # Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Stochastic Oscillator
            low_14 = df['close'].rolling(window=14).min()
            high_14 = df['close'].rolling(window=14).max()
            df['stoch'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
            
            # ATR
            high_low = df['close'].shift(-1) - df['close']
            df['atr'] = high_low.abs().rolling(window=14).mean()
            
            # VWAP
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            
            # Bollinger Bands
            df['bb_middle'] = df['sma_20']
            df['bb_std'] = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
            df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # OBV
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
            
            # Ichimoku Cloud
            df['ichimoku_tenkan'] = (df['close'].rolling(window=9).max() + df['close'].rolling(window=9).min()) / 2
            df['ichimoku_kijun'] = (df['close'].rolling(window=26).max() + df['close'].rolling(window=26).min()) / 2
            df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
            df['ichimoku_senkou_b'] = (df['close'].rolling(window=52).max() + df['close'].rolling(window=52).min()) / 2
            df['ichimoku_senkou_b'] = df['ichimoku_senkou_b'].shift(26)
            
            # Fibonacci Retracement (last 90 days)
            high = df['close'].max()
            low = df['close'].min()
            diff = high - low
            df['fib_0_0'] = high
            df['fib_23_6'] = high - 0.236 * diff
            df['fib_38_2'] = high - 0.382 * diff
            df['fib_50_0'] = high - 0.5 * diff
            df['fib_61_8'] = high - 0.618 * diff
            df['fib_100_0'] = low
            
            # ADX
            tr = pd.concat([
                (df['close'].shift(-1) - df['close']).abs(),
                (df['close'].shift(-1) - df['close'].shift(1)).abs(),
                (df['close'] - df['close'].shift(1)).abs()
            ], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()
            plus_dm = (df['close'].shift(-1) - df['close']).where(df['close'].shift(-1) > df['close'], 0)
            minus_dm = (df['close'].shift(1) - df['close']).where(df['close'].shift(1) > df['close'], 0)
            plus_di = 100 * plus_dm.rolling(window=14).mean() / atr
            minus_di = 100 * minus_dm.rolling(window=14).mean() / atr
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            df['adx'] = dx.rolling(window=14).mean()
            
            # Volatility (annualized)
            df['volatility'] = df['returns'].rolling(window=14).std() * np.sqrt(365 * 24)
            
            # Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
            df['sharpe'] = df['returns'].rolling(window=14).mean() / df['returns'].rolling(window=14).std() * np.sqrt(365 * 24)
            
            # Correlation (if multiple coin data provided)
            if coin_dfs:
                close_df = pd.DataFrame({coin: df['close'] for coin, df in coin_dfs.items()})
                df['correlation'] = close_df.corr()['bitcoin']  # Correlation with Bitcoin
            
            df.dropna(inplace=True)
            logger.info("Computed indicators: SMA, EMA, RSI, Stochastic, ATR, VWAP, Bollinger Bands, MACD, OBV, Ichimoku, Fibonacci, ADX, Volatility, Sharpe.")
            return df

        except Exception as e:
            logger.error(f"Failed to compute technical indicators: {e}")
            return None
