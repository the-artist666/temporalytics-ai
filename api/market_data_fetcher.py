import logging
import requests
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

class MarketDataFetcher:
    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def fetch_historical_data(
        self,
        coin_id: str = "bitcoin",
        vs_currency: str = "usd",
        days: int = 90
    ) -> Optional[pd.DataFrame]:
        endpoint = f"/coins/{coin_id}/market_chart"
        params = {
            "vs_currency": vs_currency,
            "days": str(days),
            "interval": "hourly"
        }
        url = self.BASE_URL + endpoint

        try:
            logger.info(f"Fetching historical data for {coin_id} over {days} days...")
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            
            if 'prices' not in data or not data['prices']:
                logger.warning(f"No price data found for {coin_id}.")
                return None
                
            df = pd.DataFrame(data['prices'], columns=['timestamp', 'close'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            logger.info(f"Fetched {len(df)} data points for {coin_id}.")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse API response: {e}")
            return None
