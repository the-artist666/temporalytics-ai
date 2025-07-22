import requests
import pandas as pd
import logging
import time

logger = logging.getLogger(__name__)

class CoinGeckoFetcher:
    def __init__(self, api_key):
        self.base_url = "https://pro-api.coingecko.com/api/v3"
        self.free_url = "https://api.coingecko.com/api/v3"
        self.api_key = api_key
        if not self.api_key:
            logger.error("CoinGecko API key is missing")
            raise ValueError("CoinGecko API key is required")
        self.headers = {"x-cg-pro-api-key": self.api_key}

    def test_api_key(self):
        try:
            response = requests.get(f"{self.base_url}/ping", headers=self.headers)
            response.raise_for_status()
            logger.info("API key test successful: %s", response.json())
            return True
        except requests.exceptions.HTTPError as e:
            logger.error("API key test failed: %s", e.response.text)
            return False

    def fetch_realtime_price(self, coin_id, vs_currency):
        try:
            response = requests.get(
                f"{self.base_url}/simple/price?ids={coin_id}&vs_currencies={vs_currency}",
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            if 'error' in data:
                logger.error("API error: %s", data['error'])
                return None, "N/A"
            return data[coin_id][vs_currency], "CoinGecko Pro"
        except requests.exceptions.HTTPError as e:
            logger.warning("Pro API failed for %s: %s, trying free API", coin_id, e.response.text)
            try:
                response = requests.get(
                    f"{self.free_url}/simple/price?ids={coin_id}&vs_currencies={vs_currency}&api_key={self.api_key}"
                )
                response.raise_for_status()
                data = response.json()
                return data[coin_id][vs_currency], "CoinGecko Free"
            except Exception as e2:
                logger.error("Free API also failed for %s: %s", coin_id, e2)
                return None, "N/A"

    def fetch_historical_data(self, coin_id, vs_currency, days=90, interval="daily"):
        try:
            response = requests.get(
                f"{self.base_url}/coins/{coin_id}/market_chart?vs_currency={vs_currency}&days={days}&interval={interval}",
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            if 'error' in data:
                logger.error("API error: %s", data['error'])
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
        except requests.exceptions.HTTPError as e:
            logger.warning("Pro API failed for %s: %s, trying free API with 30-day hourly data", coin_id, e.response.text)
            try:
                response = requests.get(
                    f"{self.free_url}/coins/{coin_id}/market_chart?vs_currency={vs_currency}&days=30&interval=hourly&api_key={self.api_key}"
                )
                response.raise_for_status()
                data = response.json()
                prices = data['prices']
                volumes = data['total_volumes']
                df = pd.DataFrame(prices, columns=['timestamp', 'close'])
                df['volume'] = [v[1] for v in volumes]
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df['high'] = df['close'] * 1.01
                df['low'] = df['close'] * 0.99
                df['open'] = df['close'].shift(1).fillna(df['close'])
                df.set_index('timestamp', inplace=True)
                return df
            except Exception as e2:
                logger.error("Free API also failed for %s: %s", coin_id, e2)
                raise
