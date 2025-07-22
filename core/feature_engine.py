import pandas as pd
import numpy as np
import ta

class FeatureEngine:
    def compute_technical_indicators(self, df, coin_data):
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['stoch'] = ta.momentum.stoch(df['high'], df['low'], df['close'], window=14)
        df['atr'] = ta.volatility.atr(df['high'], df['low'], df['close'], window=14)
        df['vwap'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'])
        bb = ta.volatility.BollingerBands(df['close'], window=20)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_middle'] = bb.bollinger_mavg()
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        ichimoku = ta.trend.IchimokuIndicator(df['high'], df['low'])
        df['ichimoku_tenkan'] = ichimoku.ichimoku_conversion_line()
        df['ichimoku_kijun'] = ichimoku.ichimoku_base_line()
        df['ichimoku_senkou_a'] = ichimoku.ichimoku_a()
        df['ichimoku_senkou_b'] = ichimoku.ichimoku_b()
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14)
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(365 * 24)
        df['sharpe'] = (df['close'].pct_change().rolling(window=20).mean() / df['volatility']) * np.sqrt(365 * 24)
        return df

    def compute_tdm_metrics(self, df):
        # Trend: SMA crossover
        df['trend'] = np.where(df['sma_20'] > df['ema_12'], 1, -1)
        # Momentum: Rate of change
        df['momentum'] = df['close'].pct_change(periods=5)
        # Conviction: RSI-based strength
        df['conviction'] = (df['rsi'] - 50) / 50
        # Stability: Inverse of volatility
        df['stability'] = 1 / (1 + df['volatility'])
        return df
