import logging
import requests
from typing import Optional, Tuple
import json

logger = logging.getLogger(__name__)

class RealtimePriceFetcher:
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
        " bitcoin": "Qwsogvtv82FCd",
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

    def fetch_realtime_price(
        self,
        coin_id: str = "bitcoin",
        vs_currency: str = "usd"
    ) -> Optional[Tuple[float, str]]:
        if not self._switch_api():
            return None

        current_api = self.API_KEYS[self.current_api_index]
        provider = current_api["provider"]
        api_key = current_api["key"]
        current_api["calls"] += 1

        try:
            if provider == "Coinranking":
                price = self._fetch_coinranking_price(coin_id, vs_currency, api_key)
            elif provider == "Finnhub":
                price = self._fetch_finnhub_price(coin_id, vs_currency, api_key)
            elif provider == "Alpha Vantage":
                price = self._fetch_alpha_vantage_price(coin_id, vs_currency, api_key)
            elif provider == "xAI":
                price = self._fetch_xai_price(coin_id, vs_currency, api_key)
            if price is not None:
                return price, provider
            return None
        except Exception as e:
            logger.error(f"Failed to fetch price from {provider}: {e}")
            return None

    def _fetch_coinranking_price(self, coin_id: str, vs_currency: str, api_key: str) -> Optional[float]:
        coin_uuid = self.COIN_IDS.get(coin_id.lower(), "Qwsogvtv82FCd")
        endpoint = f"/coin/{coin_uuid}/price"
        params = {"x-access-token": api_key}
        url = self.COINRANKING_BASE_URL + endpoint

        try:
            logger.info(f"Fetching real-time price from Coinranking for {coin_id}...")
            response = requests.get(url, headers={"x-access-token": api_key}, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data["status"] != "success" or not data["data"]["price"]:
                logger.warning(f"No price data from Coinranking for {coin_id}.")
                return None
            price = float(data["data"]["price"])
            logger.info(f"Fetched real-time price from Coinranking: {price}")
            return price
        except requests.exceptions.RequestException as e:
            logger.error(f"Coinranking request failed: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse Coinranking response: {e}")
            return None

    def _fetch_finnhub_price(self, coin_id: str, vs_currency: str, api_key: str) -> Optional[float]:
        symbol = f"BINANCE:{coin_id.upper()}USDT" if vs_currency == "usd" else f"BINANCE:{coin_id.upper()}{vs_currency.upper()}"
        endpoint = "/quote"
        params = {"symbol": symbol, "token": api_key}
        url = self.FINNHUB_BASE_URL + endpoint

        try:
            logger.info(f"Fetching real-time price from Finnhub for {symbol}...")
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if not data.get("c"):
                logger.warning(f"No price data from Finnhub for {symbol}.")
                return None
            price = float(data["c"])
            logger.info(f"Fetched real-time price from Finnhub: {price}")
            return price
        except requests.exceptions.RequestException as e:
            logger.error(f"Finnhub request failed: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse Finnhub response: {e}")
            return None

    def _fetch_alpha_vantage_price(self, coin_id: str, vs_currency: str, api_key: str) -> Optional[float]:
        symbol = f"{coin_id.upper()}/{vs_currency.upper()}"
        endpoint = "/query"
        params = {
            "function": "CURRENCY_EXCHANGE_RATE",
            "from_currency": coin_id.upper(),
            "to_currency": vs_currency.upper(),
            "apikey": api_key
        }
        url = self.ALPHA_VANTAGE_BASE_URL + endpoint

        try:
            logger.info(f"Fetching real-time price from Alpha Vantage for {symbol}...")
            response = requests.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if "Realtime Currency Exchange Rate" not in data:
                logger.warning(f"No price data from Alpha Vantage for {symbol}.")
                return None
            price = float(data["Realtime Currency Exchange Rate"]["5. Exchange Rate"])
            logger.info(f"Fetched real-time price from Alpha Vantage: {price}")
            return price
        except requests.exceptions.RequestException as e:
            logger.error(f"Alpha Vantage request failed: {e}")
            return None
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse Alpha Vantage response: {e}")
            return None

    def _fetch_xai_price(self, coin_id: str, vs_currency: str, api_key: str) -> Optional[float]:
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
                    "content": "Provide the current price of a cryptocurrency."
                },
                {
                    "role": "user",
                    "content": f"What is the current price of {coin_id} in {vs_currency}?"
                }
            ],
            "model": "grok-3-latest",
            "stream": False,
            "temperature": 0
        }

        try:
            logger.info(f"Fetching real-time price from xAI for {coin_id}...")
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if "choices" not in data or not data["choices"]:
                logger.warning(f"No price data from xAI for {coin_id}.")
                return None
            content = data["choices"][0]["message"]["content"]
            price_str = content.split("$")[-1].split()[0]
            price = float(price_str.replace(",", ""))
            logger.info(f"Fetched real-time price from xAI: {price}")
            return price
        except requests.exceptions.RequestException as e:
            logger.error(f"xAI request failed: {e}")
            return None
        except (KeyError, ValueError, IndexError) as e:
            logger.error(f"Failed to parse xAI response: {e}")
            return None
