import logging
import numpy as np
import pandas as pd
import pandas_ta as ta
try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency
    pipeline = None
from statsmodels.tsa.arima.model import ARIMA

from .external_apis import ExternalAPIs
from config import SENTIMENT_MODEL_NAME, TRADING_CONFIG


class FeatureGenerator:
    """Generate predictive features from market and external data."""

    def __init__(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise ValueError("A valid DataFrame must be provided.")
        self.df = df.copy()
        self.external_api = ExternalAPIs()
        if pipeline:
            try:
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis", model=SENTIMENT_MODEL_NAME
                )
            except Exception as e:  # handle missing backends like torch
                logging.warning(
                    "Sentiment analysis pipeline unavailable: %s", str(e)
                )
                self.sentiment_pipeline = None
        else:
            self.sentiment_pipeline = None
            logging.warning(
                "Sentiment analysis disabled; transformers or backend not available."
            )

    def add_technical_indicators(self):
        self.df.ta.ema(length=50, append=True)
        self.df.ta.ema(length=200, append=True)
        self.df.ta.rsi(append=True)
        self.df.ta.macd(append=True)
        self.df.ta.bbands(append=True)
        self.df.ta.atr(append=True)
        logging.info("🔧 Added Technical Analysis indicators.")
        return self

    def add_onchain_data(self):
        on_chain_data = self.external_api.get_on_chain_data(TRADING_CONFIG["asset"])
        for key, value in on_chain_data.items():
            self.df[key] = value
        return self

    def add_sentiment_data(self):
        if self.sentiment_pipeline is None:
            self.df["sentiment_score"] = 0
            return self

        headlines = self.external_api.get_news_headlines(TRADING_CONFIG["asset"])
        if not headlines:
            self.df["sentiment_score"] = 0
            return self

        sentiments = self.sentiment_pipeline(headlines)
        score_map = {"negative": -1, "neutral": 0, "positive": 1}
        scores = [score_map.get(s["label"].lower(), 0) * s["score"] for s in sentiments]
        avg_sentiment = np.mean(scores) if scores else 0
        self.df["sentiment_score"] = avg_sentiment
        logging.info(f"❤️ Calculated average sentiment score: {avg_sentiment:.4f}")
        return self

    def add_arima_forecast(self):
        close_prices = self.df['Close'].astype(float).values
        try:
            model = ARIMA(close_prices, order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]
            self.df['arima_forecast'] = forecast
            logging.info(f"🧠 ARIMA forecast for next period: {forecast:.2f}")
        except Exception as e:
            logging.error(f"❌ ARIMA model failed: {e}. Using last close price as forecast.")
            self.df['arima_forecast'] = close_prices[-1]
        return self

    def generate_features(self):
        self.add_technical_indicators()
        self.add_onchain_data()
        self.add_sentiment_data()
        self.add_arima_forecast()
        self.df.dropna(inplace=True)
        logging.info("✅ Feature generation complete.")
        return self.df
