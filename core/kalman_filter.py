import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class KalmanFilter:
    def __init__(self, process_variance: float = 1e-5, measurement_variance: float = 0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.estimate_error = 1.0

    def filter(self, df: pd.DataFrame, column: str = 'close') -> Optional[pd.Series]:
        """Apply Kalman filter to smooth the specified column."""
        try:
            if df is None or df.empty or column not in df.columns:
                logger.error("Invalid DataFrame or column.")
                return None

            smoothed = []
            for measurement in df[column]:
                # Prediction
                prior_estimate = self.estimate
                prior_error = self.estimate_error + self.process_variance
                
                # Update
                kalman_gain = prior_error / (prior_error + self.measurement_variance)
                self.estimate = prior_estimate + kalman_gain * (measurement - prior_estimate)
                self.estimate_error = (1 - kalman_gain) * prior_error
                
                smoothed.append(self.estimate)
            
            logger.info(f"Applied Kalman filter to {column}.")
            return pd.Series(smoothed, index=df.index)

        except Exception as e:
            logger.error(f"Failed to apply Kalman filter: {e}")
            return None
