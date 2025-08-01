"""Wrapper for third-party APIs such as CoinMetrics and NewsAPI."""
import requests
import logging
from datetime import datetime, timedelta
from config import API_CONFIG

class ExternalAPIs:
    """Handles connections to third-party APIs like CoinMetrics and NewsAPI."""

    def __init__(self):
        self.coinmetrics_api_key = API_CONFIG["coinmetrics_api_key"]
        self.news_api_key = API_CONFIG["news_api_key"]
        self.coinmetrics_base_url = "https://community-api.coinmetrics.io/v4"
        self.news_base_url = "https://newsapi.org/v2/everything"

    def get_on_chain_data(self, asset: str):
        """Fetch key on-chain metrics from the CoinMetrics Community API."""
        metrics_map = {
            "active_addresses": "AdrActCnt",
            "transaction_count": "TxTfrCnt",
        }

        params = {
            "assets": asset.lower(),
            "metrics": ",".join(metrics_map.values()),
            "frequency": "1d",
            "page_size": 1,
        }

        if self.coinmetrics_api_key:
            params["api_key"] = self.coinmetrics_api_key

        try:
            response = requests.get(
                f"{self.coinmetrics_base_url}/timeseries/asset-metrics", params=params
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            latest = data[-1] if data else {}
            on_chain_features = {
                name: float(latest.get(metric, 0)) for name, metric in metrics_map.items()
            }
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Error fetching on-chain data from CoinMetrics: {e}")
            on_chain_features = {"active_addresses": 0, "transaction_count": 0}

        logging.info(f"🔗 Fetched on-chain data: {on_chain_features}")
        return on_chain_features

    def get_news_headlines(self, query: str):
        """Fetch recent news headlines for a query from NewsAPI."""
        if not self.news_api_key or "YOUR" in self.news_api_key:
            logging.warning("⚠️ NewsAPI key not set. Skipping sentiment analysis.")
            return []
        try:
            params = {
                "q": query,
                "apiKey": self.news_api_key,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 20,
            }
            response = requests.get(self.news_base_url, params=params)
            response.raise_for_status()
            articles = response.json().get("articles", [])
            headlines = [article["title"] for article in articles]
            logging.info(f"📰 Fetched {len(headlines)} headlines for '{query}'.")
            return headlines
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Error fetching news from NewsAPI: {e}")
            return []
