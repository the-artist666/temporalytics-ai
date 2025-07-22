import requests
import pandas as pd
import time

class CoinGeckoFetcher:
    def __init__(self, api_key):
        self.base_url = "https://pro-api.coingecko.com/api/v3"
        self.api_key = api_key
        self.headers = {"x-cg-pro-api-key": self.api_key}

    def fetch_realtime_price(self, coin_id, vs_currency):
        try:
            response = requests.get(
                f"{self.base_url}/simple/price?ids={coin_id}&vs_currencies={vs_currency}",
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            return data[coin_id][vs_currency], "CoinGecko Pro"
        except Exception as e:
            print(f"Error fetching price for {coin_id}: {e}")
            return None, "N/A"

    def fetch_historical_data(self, coin_id, vs_currency, days=90, interval="hourly"):
        try:
            response = requests.get(
                f"{self.base_url}/coins/{coin_id}/market_chart?vs_currency={vs_currency}&days={days}&interval={interval}",
                headers=self.headers
            )
            response.raise_for_status()
            data = response.json()
            prices = data['prices']
            volumes = data['total_volumes']
            df = pd.DataFrame(prices, columns=['timestamp', 'close'])
            df['volume'] = [v[1] for v in volumes]
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['high'] = df['close'] * 1.01  # Mock high/low for candlestick
            df['low'] = df['close'] * 0.99
            df.set_index('timestamp', inplace=True)
            return df
        except Exception as e:
            print(f"Error fetching historical data for {coin_id}: {e}")
            raise
