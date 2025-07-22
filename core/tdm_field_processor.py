import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class TDMFieldProcessor:
    def __init__(self):
        pass

    def compute_tdm_metrics(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            if df is None or df.empty:
                logger.error("Input DataFrame is None or empty.")
                return None

            df = df.copy()
            
            df['trend'] = df['ema_12'] - df['close'].ewm(span=50, adjust=False).mean()
            df['momentum'] = df['returns'].diff().rolling(window=14).mean()
            df['conviction'] = (df['rsi'] + df['stoch']) / 2
            df['stability'] = 1 / df['atr']
            
            df.dropna(inplace=True)
            logger.info("Computed TDM metrics: Trend, Momentum, Conviction, Stability.")
            return df

        except Exception as e:
            logger.error(f"Failed to compute TDM metrics: {e}")
            return None
