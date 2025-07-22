import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class FeatureEngine:
    def __init__(self):
        pass

    def compute_technical_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute technical indicators for the given price data."""
        try:
            if df is None or df.empty:
                logger.error("Input DataFrame is None or empty.")
                return None

            df = df.copy()
            df['returns'] = df['close'].pct_change()

            # Moving Averages
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            
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
            df['vwap'] = (df['close'] * df['close'].shift(-1)).cumsum() / df['close'].shift(-1).cumsum()
            
            df.dropna(inplace=True)
            logger.info("Computed technical indicators: SMA, EMA, RSI, Stochastic, ATR, VWAP.")
            return df

        except Exception as e:
            logger.error(f"Failed to compute technical indicators: {e}")
            return None
