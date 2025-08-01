import logging
from datetime import datetime, timedelta
import requests
from config import API_CONFIG


class ExternalAPIs:
    """Handles connections to third-party APIs like Glassnode and NewsAPI."""

    def __init__(self):
        self.glassnode_api_key = API_CONFIG["glassnode_api_key"]
        self.news_api_key = API_CONFIG["news_api_key"]
        self.glassnode_base_url = "https://api.glassnode.com/v1/metrics"
        self.news_base_url = "https://newsapi.org/v2/everything"

    def get_on_chain_data(self, asset):
        """Fetch key on-chain metrics from Glassnode."""
        if not self.glassnode_api_key or "your" in self.glassnode_api_key.lower():
            logging.warning("⚠️ Glassnode API key not set. Returning neutral on-chain data.")
            return {'mvrv_z_score': 0, 'sopr': 1.0, 'nupl': 0.5}

        metrics = {
            "mvrv_z_score": "/market/mvrv_z_score",
            "sopr": "/indicators/sopr",
            "nupl": "/indicators/net_unrealized_profit_loss"
        }

        on_chain_features = {}
        for name, endpoint in metrics.items():
            try:
                params = {'a': asset, 'api_key': self.glassnode_api_key, 'i': '24h'}
                response = requests.get(f"{self.glassnode_base_url}{endpoint}", params=params)
                response.raise_for_status()
                data = response.json()
                if data:
                    on_chain_features[name] = data[-1]['v']
                else:
                    on_chain_features[name] = 0
            except requests.exceptions.RequestException as e:
                logging.error(f"❌ Error fetching {name} from Glassnode: {e}")
                on_chain_features[name] = 0 if name == 'mvrv_z_score' else (1.0 if name == 'sopr' else 0.5)

        logging.info(f"🔗 Fetched on-chain data: {on_chain_features}")
        return on_chain_features

    def get_news_headlines(self, query):
        """Fetch recent news headlines for a given query from NewsAPI."""
        if not self.news_api_key or "your" in self.news_api_key.lower():
            logging.warning("⚠️ NewsAPI key not set. Skipping sentiment analysis.")
            return []

        try:
            params = {
                'q': query,
                'apiKey': self.news_api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20
            }
            response = requests.get(self.news_base_url, params=params)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            headlines = [article['title'] for article in articles]
            logging.info(f"📰 Fetched {len(headlines)} headlines for '{query}'.")
            return headlines
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Error fetching news from NewsAPI: {e}")
            return []
