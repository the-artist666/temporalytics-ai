import requests
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)

class CoinGeckoFetcher:
    def __init__(self, api_key):
        self.base_url = "https://pro-api.coingecko.com/api/v3"
        self.api_key = api_key
        if not self.api_key:
            logger.error("CoinGecko API key is missing")
            raise ValueError("CoinGecko API key is required")
        self.headers = {"x-cg-pro-api-key": self.api_key}

    def fetch_realtime_price(self, coin_id, vs_currency):
        try:
            # Try header-based authentication first
            response = requests.get(
                f"{self.base_url}/simple/price?ids={coin_id}&vs_currencies={vs_currency}",
                headers=self.headers
            )
            if response.status_code == 400 or "error" in response.json():
                logger.warning(f"Header auth failed for {coin_id}, trying query parameter")
                # Fallback to query parameter
                response = requests.get(
                    f"{self.base_url}/simple/price?ids={coin_id}&vs_currencies={vs_currency}&api_key={self.api_key}"
                )
            response.raise_for_status()
            data = response.json()
            if 'error' in data:
                logger.error(f"API error: {data['error']}")
                return None, "N/A"
            return data[coin_id][vs_currency], "CoinGecko Pro"
        except Exception as e:
            logger.error(f"Error fetching price for {coin_id}: {e}")
            return None, "N/A"

    def fetch_historical_data(self, coin_id, vs_currency, days=90, interval="daily"):
        try:
            # Use daily interval for 90 days to avoid API limitations
            response = requests.get(
                f"{self.base_url}/coins/{coin_id}/market_chart?vs_currency={vs_currency}&days={days}&interval={interval}",
                headers=self.headers
            )
            if response.status_code == 400 or "error" in response.json():
                logger.warning(f"Header auth failed for {coin_id} historical data, trying query parameter")
                response = requests.get(
                    f"{self.base_url}/coins/{coin_id}/market_chart?vs_currency={vs_currency}&days={days}&interval={interval}&api_key={self.api_key}"
                )
            response.raise_for_status()
            data = response.json()
            if 'error' in data:
                logger.error(f"API error: {data['error']}")
                raise Exception(f"API error: {data['error']}")
            prices = data['prices']
            volumes = data['total_volumes']
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['volume'] = [v[1] for v in volumes]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['high'] = df['close'] * 1.01  # Mock high/low for candlestick
            df['low'] = df['close'] * 0.99
            df['open'] = df['close'].shift(1).fillna(df['close'])
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching historical data for {coin_id}: {e}")
            raise
