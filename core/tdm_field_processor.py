import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class TDMFieldProcessor:
    def __init__(self):
        pass

    def compute_tdm_metrics(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute TDM (Trend, Dynamics, Momentum) metrics."""
        try:
            if df is None or df.empty:
                logger.error("Input DataFrame is None or empty.")
                return None

            df = df.copy()
            
            # Trend: Difference between short-term and long-term EMA
            df['trend'] = df['ema_12'] - df['close'].ewm(span=50, adjust=False).mean()
            
            # Momentum: Rate of change of returns
            df['momentum'] = df['returns'].diff().rolling(window=14).mean()
            
            # Conviction: Combination of RSI and Stochastic
            df['conviction'] = (df['rsi'] + df['stoch']) / 2
            
            # Stability: Inverse of ATR
            df['stability'] = 1 / df['atr']
            
            df.dropna(inplace=True)
            logger.info("Computed TDM metrics: Trend, Momentum, Conviction, Stability.")
            return df

        except Exception as e:
            logger.error(f"Failed to compute TDM metrics: {e}")
            return None
