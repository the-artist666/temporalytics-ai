import numpy as np
import logging

logger = logging.getLogger(__name__)

class KalmanFilter:
    def __init__(self, process_variance: float = 1e-5, measurement_variance: float = 0.1):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = 0.0
        self.error_cov = 1.0
        logger.info("Kalman Filter initialized with process variance %f and measurement variance %f",
                    process_variance, measurement_variance)

    def update(self, measurement: float) -> float:
        # Prediction
        prediction = self.estimate
        prediction_error_cov = self.error_cov + self.process_variance
        
        # Update
        kalman_gain = prediction_error_cov / (prediction_error_cov + self.measurement_variance)
        self.estimate = prediction + kalman_gain * (measurement - prediction)
        self.error_cov = (1 - kalman_gain) * prediction_error_cov
        
        return self.estimate

    def filter_series(self, series: np.ndarray) -> np.ndarray:
        filtered = np.zeros_like(series)
        for i, value in enumerate(series):
            filtered[i] = self.update(value)
        logger.info("Applied Kalman filter to series of length %d", len(series))
        return filtered
