import numpy as np
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class TDMFieldProcessor:
    def __init__(self, time_scale: float = 1.0):
        self.time_scale = time_scale
        logger.info("TDM Field Processor initialized.")

    def get_symbolic_pattern(self, prices: np.ndarray, volatility: np.ndarray) -> str:
        if len(prices) != len(volatility):
            raise ValueError("Prices and volatility arrays must have the same length.")
        
        gradients = np.gradient(prices)
        codes = []
        for i, g in enumerate(gradients):
            threshold = volatility[i]
            if g > threshold:
                codes.append("U")
            elif g > 0:
                codes.append("u")
            elif g < -threshold:
                codes.append("D")
            elif g < 0:
                codes.append("d")
            else:
                codes.append("S")
        return ''.join(codes)

    def calculate_field_metrics(self, pattern: str) -> Dict[str, float]:
        length = len(pattern) + 1e-8
        strong_up = pattern.count("U")
        weak_up = pattern.count("u")
        strong_down = pattern.count("D")
        weak_down = pattern.count("d")
        stable = pattern.count("S")
        momentum = (strong_up + weak_up - strong_down - weak_down) / length
        conviction = (strong_up + strong_down) / (weak_up + weak_down + 1e-8)
        stability = stable / length
        logger.debug(f"Calculated TDM metrics: Momentum={momentum:.3f}, Conviction={conviction:.3f}, Stability={stability:.3f}")
        return {"momentum": momentum, "conviction": conviction, "stability": stability}
