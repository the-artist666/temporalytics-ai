import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class FeatureEngine:
    def __init__(self, rsi_period: int = 14, short_ma: int = 7, long_ma: int = 21, ema_span: int = 20,
                 stoch_k: int = 14, stoch_d: int = 3, atr_period: int = 14):
        self.rsi_period = rsi_period
        self.short_ma = short_ma
        self.long_ma = long_ma
        self.ema_span = ema_span
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.atr_period = atr_period
        logger.info("Feature Engine initialized with enhanced indicators.")

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in df.columns:
            raise ValueError("Input DataFrame must contain a 'close' column.")
        
        logger.info("Generating enhanced technical indicator features...")
        
        # Basic indicators
        df['MA_short'] = df['close'].rolling(window=self.short_ma).mean()
        df['MA_long'] = df['close'].rolling(window=self.long_ma).mean()
        df['EMA'] = df['close'].ewm(span=self.ema_span, adjust=False).mean()
        df['Volatility'] = df['close'].rolling(window=self.long_ma).std()
        df['RSI'] = self._compute_rsi(df['close'])
        
        # New indicators
        df['Stoch_K'], df['Stoch_D'] = self._compute_stochastic(df)
        df['ATR'] = self._compute_atr(df)
        df['VWAP'] = self._compute_vwap(df)
        
        logger.info("Features generated. Dropping rows with NaN values.")
        return df.dropna()

    def _compute_rsi(self, series: pd.Series) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _compute_stochastic(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        low_min = df['close'].rolling(window=self.stoch_k).min()
        high_max = df['close'].rolling(window=self.stoch_k).max()
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-8)
        stoch_d = stoch_k.rolling(window=self.stoch_d).mean()
        return stoch_k, stoch_d

    def _compute_atr(self, df: pd.DataFrame) -> pd.Series:
        high_low = df['close'].shift(-1) - df['close'].shift(1)
        high_close = abs(df['close'].shift(-1) - df['close'])
        low_close = abs(df['close'].shift(1) - df['close'])
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=self.atr_period, min_periods=1).mean()
        return atr

    def _compute_vwap(self, df: pd.DataFrame) -> pd.Series:
        typical_price = df['close']
        volume_proxy = df['close'].diff().abs()
        cumulative_pv = (typical_price * volume_proxy).cumsum()
        cumulative_volume = volume_proxy.cumsum()
        vwap = cumulative_pv / (cumulative_volume + 1e-8)
        return vwap
