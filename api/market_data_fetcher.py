import logging
import requests
import pandas as pd
from typing import Optional, Dict
import json

logger = logging.getLogger(__name__)

class MarketDataFetcher:
    COINRANKING_BASE_URL = "https://api.coinranking.com/v2"
    FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
    ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co"
    XAI_BASE_URL = "https://api.x.ai/v1"
    
    API_KEYS = [
        {"provider": "Coinranking", "key": "coinrankingef8b9719f8d31dce795dbbc1b3fc7c3a229827892a81ac29", "calls": 0, "limit": 5000},
        {"provider": "Finnhub", "key": "d1vdjspr01qqgeekurq0d1vdjspr01qqgeekurqg", "calls": 0, "limit": 60 * 60 * 24},
        {"provider": "Alpha Vantage", "key": "6LYYEXSTKJTI3SJR", "calls": 0, "limit": 5 * 60 * 24},
        {"provider": "xAI", "key": "xai-aMD7Q52h0LUgEzgTTqjec5zmtZOhFF6fj1jkwNKZ2dIeNR8FRkRyFC0fBxGjNz823vjwR9Ty8QrwWaPz", "calls": 0, "limit": 10000}
    ]
    
    COIN_IDS = {
        "bitcoin": "Qwsogvtv82FCd",
        "ethereum": "razxDUgYGNAdQ",
        "solana": "zNZHO_Sjf",
        "binancecoin": "WcwrkfNI4FUAe",
        "xrp": "-l8Mn2pVlRs-p",
        "cardano": "qzawljRxB5bYu",
        "dogecoin": "a91GCGd_u96cF",
        "polkadot": "f0_83tArud7OD",
        "chainlink": "VLqpJwogdhH4b",
        "polygon": "9K7H3TJv-3m2f"
    }

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.current_api_index = 0

    def _switch_api(self):
        for _ in range(len(self.API_KEYS)):
            self.current_api_index = (self.current_api_index + 1) % len(self.API_KEYS)
            current_api = self.API_KEYS[self.current_api_index]
            if current_api["calls"] < current_api["limit"]:
                logger.info(f"Using {current_api['provider']} API")
                return True
        logger.warning("All APIs have reached their call limits.")
        return False

    def fetch_historical_data(
        self,
        coin_id: str = "bitcoin",
        vs_currency: str = "usd",
        days: int = 90
    ) -> Optional[pd.DataFrame]:
        if not self._switch_api():
            return None

        current_api = self.API_KEYS[self.current_api_index]
        provider = current_api["provider"]
        api_key = current_api["key"]
        current_api["calls"] += 1

        try:
            if provider == "Coinranking":
                return self._fetch_coinranking_historical(coin_id, vs_currency, days, api_key)
            elif provider == "Finnhub":
                return self._fetch_finnhub_historical(coin_id, vs_currency, days, api_key)
            elif provider == "Alpha Vantage":
                return self._fetch_alpha_vantage_historical(coin_id, vs_currency, days, api_key)
            elif provider == "xAI":
                return self._fetch_xai_historical(coin_id, vs_currency, days, api_key)
        except Exception as e:
            logger.error(f"Failed to fetch data from {provider}: {e}")
            return None

    def _fetch_coinranking_historical(self, coin_id: str, vs_currency: str, days: int, api_key: str) -> Optional[pd.DataFrame]:
        coin_uuid = self.COIN_IDS.get(coin_id.lower(), "Qwsogvtv82FCd")
        endpoint = f"/coin/{coin_uuid}/history"
        params = {"timePeriod": f"{days}d", "x-access-token": api_key}
        url = self.COINRANKING_BASE_URL + endpoint

        try:
            logger.info(f"Fetching historical data from Coinranking for {coin_id}...")
            response = requests.get(url, headers={"x-access-token": api_key}, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data["status"] != "success" or not data["data"]["history"]:
                logger.warning(f"No historical data from Coinranking for {coin_id}.")
                return None
            prices = data["data"]["history"]
            df = pd.DataFrame(prices)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["close"] = df["price"].astype(float)
            df["volume"] = df.get("volume", 0).astype(float)
            df.set_index("timestamp", inplace=True)
            df = df[["close", "volume"]]
            logger.info(f"Fetched {len(df)} data points from Coinranking.")
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Coinranking request failed: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse Coinranking response: {e}")
            return None

    def _fetch_finnhub_historical(self, coin_id: str, vs_currency: str, days: int, api_key: str) -> Optional[pd.DataFrame]:
        symbol = f"BINANCE:{coin_id.upper()}USDT" if vs_currency == "usd" else f"BINANCE:{coin_id.upper()}{vs_currency.upper()}"
        to_time = int(pd.Timestamp.now().timestamp())
        from_time = int((pd.Timestamp.now() - pd.Timedelta(days=days)).timestamp())
        endpoint = f"/crypto/candle"
        params = {
            "symbol": symbol,
            "resolution": "60",
            "from": from_time,
            "to": to_time,
            "token": api_key
        }
        url = self.FINNHUB_BASE_URL + endpoint

        try:
            logger.info(f"Fetching historical data from Finnhub for {symbol}...")
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data.get("s") == "no_data" or not data.get("t"):
                logger.warning(f"No historical data from Finnhub for {symbol}.")
                return None
            df = pd.DataFrame({
                "timestamp": data["t"],
                "close": data["c"],
                "volume": data.get("v", [0] * len(data["t"]))
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            df.set_index("timestamp", inplace=True)
            df["close"] = df["close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            logger.info(f"Fetched {len(df)} data points from Finnhub.")
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Finnhub request failed: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse Finnhub response: {e}")
            return None

    def _fetch_alpha_vantage_historical(self, coin_id: str, vs_currency: str, days: int, api_key: str) -> Optional[pd.DataFrame]:
        symbol = f"{coin_id.upper()}/{vs_currency.upper()}"
        endpoint = "/query"
        params = {
            "function": "CRYPTO_INTRADAY",
            "symbol": coin_id.upper(),
            "market": vs_currency.upper(),
            "interval": "60min",
            "outputsize": "full",
            "apikey": api_key
        }
        url = self.ALPHA_VANTAGE_BASE_URL + endpoint

        try:
            logger.info(f"Fetching historical data from Alpha Vantage for {symbol}...")
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if "Time Series Crypto (60min)" not in data:
                logger.warning(f"No historical data from Alpha Vantage for {symbol}.")
                return None
            time_series = data["Time Series Crypto (60min)"]
            df = pd.DataFrame.from_dict(time_series, orient="index")
            df["timestamp"] = pd.to_datetime(df.index)
            df["close"] = df["4. close"].astype(float)
            df["volume"] = df.get("5. volume", 0).astype(float)
            df.set_index("timestamp", inplace=True)
            df = df[["close", "volume"]]
            df = df.sort_index()
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days)
            df = df[df.index >= cutoff_date]
            logger.info(f"Fetched {len(df)} data points from Alpha Vantage.")
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse Alpha Vantage response: {e}")
            return None

    def _fetch_xai_historical(self, coin_id: str, vs_currency: str, days: int, api_key: str) -> Optional[pd.DataFrame]:
        endpoint = "/chat/completions"
        url = self.XAI_BASE_URL + endpoint
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "Provide historical price data in JSON format."
                },
                {
                    "role": "user",
                    "content": f"Fetch historical price data for {coin_id} in {vs_currency} for the past {days} days, hourly interval, in JSON format with timestamp (Unix ms), close price, and volume."
                }
            ],
            "model": "grok-3-latest",
            "stream": False,
            "temperature": 0
        }

        try:
            logger.info(f"Fetching historical data from xAI for {coin_id}...")
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if "choices" not in data or not data["choices"]:
                logger.warning(f"No data from xAI for {coin_id}.")
                return None
            content = data["choices"][0]["message"]["content"]
            prices = json.loads(content)
            if not isinstance(prices, list) or not prices:
                logger.warning(f"Invalid historical data format from xAI for {coin_id}.")
                return None
            df = pd.DataFrame(prices)
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df["close"] = df["close"].astype(float)
            df["volume"] = df.get("volume", 0).astype(float)
            logger.info(f"Fetched {len(df)} data points from xAI.")
            return df
        except requests.exceptions.RequestException as e:
            logger.error(f"xAI request failed: {e}")
            return None
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse xAI response: {e}")
            return None

    def fetch_sentiment(self, coin_id: str) -> Optional[Dict]:
        if not self._switch_api():
            return None
        current_api = self.API_KEYS[self.current_api_index]
        if current_api["provider"] != "xAI":
            logger.info("Sentiment analysis only available via xAI API.")
            return None
        api_key = current_api["key"]
        current_api["calls"] += 1

        endpoint = "/chat/completions"
        url = self.XAI_BASE_URL + endpoint
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "Provide a sentiment analysis for a cryptocurrency with confidence score and a brief news snippet."
                },
                {
                    "role": "user",
                    "content": f"Analyze the market sentiment for {coin_id}. Return JSON with 'sentiment' ('bullish', 'bearish', 'neutral'), 'confidence' (0-1), and 'news_snippet' (brief text)."
                }
            ],
            "model": "grok-3-latest",
            "stream": False,
            "temperature": 0.5
        }

        try:
            logger.info(f"Fetching sentiment for {coin_id} from xAI...")
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if "choices" not in data or not data["choices"]:
                logger.warning(f"No sentiment data from xAI for {coin_id}.")
                return None
            content = json.loads(data["choices"][0]["message"]["content"])
            logger.info(f"Sentiment for {coin_id}: {content['sentiment']}")
            return content
        except requests.exceptions.RequestException as e:
            logger.error(f"xAI sentiment request failed: {e}")
            return None
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse xAI sentiment response: {e}")
            return None
