import numpy as np
from pykalman import KalmanFilter

class KalmanFilter:
    def filter(self, df, column):
        kf = KalmanFilter(initial_state_mean=df[column].iloc[0], n_dim_obs=1)
        state_means, _ = kf.filter(df[column].values)
        return state_means
