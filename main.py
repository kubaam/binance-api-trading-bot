"""Single-file Crypto Trading Bot Example.

This script merges the functionality of the modular bot
into one file for simplicity. API credentials are loaded
from a `.env` file so secrets remain outside of version
control.

The bot performs these steps:
1. Fetch historical klines from Binance.
2. Generate technical, on-chain, sentiment and forecast features.
3. Combine signals into a final BUY/SELL/HOLD decision.
4. Size and execute a trade using OCO orders on Binance Testnet.

This code is for educational purposes only. Use at your own risk.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pandas_ta as ta
import requests
from statsmodels.tsa.arima.model import ARIMA
from transformers import pipeline
from binance.client import Client
from binance.enums import (
    SIDE_SELL,
    TIME_IN_FORCE_GTC,
    ORDER_TYPE_MARKET,
)
from binance.exceptions import BinanceAPIException, BinanceOrderException
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load environment variables from .env
load_dotenv(Path(__file__).resolve().parent / ".env")

API_CONFIG: Dict[str, str] = {
    "use_testnet": True,
    "binance_api_key": os.getenv("BINANCE_KEY", ""),
    "binance_api_secret": os.getenv("BINANCE_SECRET", ""),
    "coinmetrics_api_key": os.getenv("COINMETRICS_API_KEY", ""),
    "news_api_key": os.getenv("NEWS_API_KEY", ""),
}

TRADING_CONFIG = {
    "trade_symbol": "BTCUSDT",
    "asset": "BTC",
    "quote_asset": "USDT",
    "timeframe": "1h",
    "klines_limit": 500,
}

RISK_CONFIG = {
    "risk_per_trade_percent": 1.0,
    "atr_period_for_sl": 14,
    "atr_multiplier_for_sl": 2.0,
    "min_risk_reward_ratio": 2.0,
}

SIGNAL_WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.20,
    "on_chain": 0.25,
    "sentiment": 0.15,
    "forecast": 0.10,
}

SENTIMENT_MODEL_NAME = "ProsusAI/finbert"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# Binance API Wrapper
# ---------------------------------------------------------------------------

class BinanceHandler:
    """Simplified Binance API helper."""

    def __init__(self) -> None:
        if not API_CONFIG["binance_api_key"]:
            raise ValueError("Binance API key missing. Set BINANCE_KEY in .env")
        self.client = Client(
            API_CONFIG["binance_api_key"],
            API_CONFIG["binance_api_secret"],
            testnet=API_CONFIG["use_testnet"],
        )
        self.client.ping()
        logging.info("Connected to Binance API")

    def get_historical_data(self, symbol: str, timeframe: str, hours: int) -> pd.DataFrame:
        klines = self.client.get_historical_klines(symbol, timeframe, f"{hours} hours ago UTC")
        df = pd.DataFrame(
            klines,
            columns=[
                "Open Time",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Close Time",
                "Quote Asset Volume",
                "Number of Trades",
                "Taker Buy Base",
                "Taker Buy Quote",
                "Ignore",
            ],
        )
        for col in ["Open", "High", "Low", "Close", "Volume", "Quote Asset Volume"]:
            df[col] = pd.to_numeric(df[col])
        df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
        df.set_index("Open Time", inplace=True)
        return df

    def get_account_balance(self, asset: str) -> float:
        bal = self.client.get_asset_balance(asset=asset)
        return float(bal["free"])

    def get_current_price(self, symbol: str) -> float:
        return float(self.client.get_symbol_ticker(symbol=symbol)["price"])

    def place_oco(self, symbol: str, quantity: float, price: float, stop: float, stop_limit: float):
        try:
            return self.client.create_oco_order(
                symbol=symbol,
                side=SIDE_SELL,
                quantity=quantity,
                price=price,
                stopPrice=stop,
                stopLimitPrice=stop_limit,
                stopLimitTimeInForce=TIME_IN_FORCE_GTC,
            )
        except (BinanceAPIException, BinanceOrderException) as e:
            logging.error(f"OCO order failed: {e}; submitting market order instead")
            return self.client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
            )

# ---------------------------------------------------------------------------
# External APIs
# ---------------------------------------------------------------------------

class ExternalAPIs:
    """Fetch on-chain metrics and news headlines."""

    def __init__(self) -> None:
        self.coinmetrics_base = "https://community-api.coinmetrics.io/v4"
        self.news_base = "https://newsapi.org/v2/everything"

    def get_on_chain_data(self, asset: str) -> Dict[str, float]:
        params = {
            "assets": asset.lower(),
            "metrics": "AdrActCnt,TxTfrCnt",
            "frequency": "1d",
            "page_size": 1,
        }
        if API_CONFIG["coinmetrics_api_key"]:
            params["api_key"] = API_CONFIG["coinmetrics_api_key"]
        try:
            r = requests.get(f"{self.coinmetrics_base}/timeseries/asset-metrics", params=params)
            r.raise_for_status()
            data = r.json().get("data", [])
            latest = data[-1] if data else {}
            return {
                "active_addresses": float(latest.get("AdrActCnt", 0)),
                "transaction_count": float(latest.get("TxTfrCnt", 0)),
            }
        except requests.RequestException as e:
            logging.error(f"On-chain data error: {e}")
            return {"active_addresses": 0.0, "transaction_count": 0.0}

    def get_news_headlines(self, query: str):
        key = API_CONFIG["news_api_key"]
        if not key:
            logging.warning("NewsAPI key missing; skipping sentiment")
            return []
        params = {"q": query, "apiKey": key, "language": "en", "sortBy": "publishedAt", "pageSize": 20}
        try:
            r = requests.get(self.news_base, params=params)
            r.raise_for_status()
            return [a["title"] for a in r.json().get("articles", [])]
        except requests.RequestException as e:
            logging.error(f"NewsAPI error: {e}")
            return []

# ---------------------------------------------------------------------------
# Feature Generation
# ---------------------------------------------------------------------------

class FeatureGenerator:
    """Create features from price data and external sources."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.copy()
        self.apis = ExternalAPIs()
        self.sentiment = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)

    def add_ta(self):
        self.df.ta.ema(length=50, append=True)
        self.df.ta.ema(length=200, append=True)
        self.df.ta.rsi(append=True)
        self.df.ta.macd(append=True)
        self.df.ta.bbands(append=True)
        self.df.ta.atr(append=True)
        return self

    def add_on_chain(self):
        data = self.apis.get_on_chain_data(TRADING_CONFIG["asset"])
        for k, v in data.items():
            self.df[k] = v
        return self

    def add_sentiment(self):
        headlines = self.apis.get_news_headlines(TRADING_CONFIG["asset"])
        if not headlines:
            self.df["sentiment_score"] = 0.0
            return self
        results = self.sentiment(headlines)
        score_map = {"negative": -1, "neutral": 0, "positive": 1}
        scores = [score_map[r["label"]] * r["score"] for r in results]
        self.df["sentiment_score"] = np.mean(scores) if scores else 0.0
        return self

    def add_forecast(self):
        try:
            model = ARIMA(self.df["Close"].values, order=(5, 1, 0))
            fit = model.fit()
            forecast = fit.forecast(steps=1)
            forecast_value = float(forecast[0])
            self.df["arima_forecast"] = forecast_value
            logging.info(f"ARIMA forecast: {forecast_value:.2f}")
        except Exception as e:
            logging.error(f"ARIMA failed: {e}; using last close")
            self.df["arima_forecast"] = self.df["Close"].iloc[-1]
        return self

    def generate(self) -> pd.DataFrame:
        return self.add_ta().add_on_chain().add_sentiment().add_forecast().df.dropna()

# ---------------------------------------------------------------------------
# Signal Combination
# ---------------------------------------------------------------------------

class SignalCombiner:
    """Combine indicators into a final trading signal."""

    def __init__(self, df: pd.DataFrame) -> None:
        self.data = df.iloc[-1]

    def _trend(self) -> int:
        short = self.data["EMA_50"]
        long = self.data["EMA_200"]
        return 1 if short > long else -1 if short < long else 0

    def _momentum(self) -> int:
        rsi = self.data["RSI_14"]
        return -1 if rsi > 70 else 1 if rsi < 30 else 0

    def _on_chain(self) -> float:
        score = 0.0
        if self.data["active_addresses"] > 700000:
            score += 0.5
        elif self.data["active_addresses"] < 300000:
            score -= 0.5
        if self.data["transaction_count"] > 1000000:
            score += 0.5
        elif self.data["transaction_count"] < 500000:
            score -= 0.5
        return max(min(score, 1), -1)

    def _sentiment(self) -> float:
        return self.data["sentiment_score"]

    def _forecast(self) -> int:
        price = self.data["Close"]
        forecast = self.data["arima_forecast"]
        return 1 if forecast > price * 1.005 else -1 if forecast < price * 0.995 else 0

    def final_signal(self) -> str:
        parts = {
            "trend": self._trend(),
            "momentum": self._momentum(),
            "on_chain": self._on_chain(),
            "sentiment": self._sentiment(),
            "forecast": self._forecast(),
        }
        logging.info(f"Signal components: {parts}")
        score = sum(parts[name] * SIGNAL_WEIGHTS[name] for name in SIGNAL_WEIGHTS)
        logging.info(f"Weighted score: {score:.4f}")
        if score >= 0.4:
            return "BUY"
        if score <= -0.4:
            return "SELL"
        return "HOLD"

# ---------------------------------------------------------------------------
# Position Sizing and Trade Execution
# ---------------------------------------------------------------------------

class PositionSizer:
    """ATR-based position sizing."""

    def __init__(self, balance: float, price: float, atr: float) -> None:
        self.balance = balance
        self.price = price
        self.atr = atr

    def size(self) -> tuple[float, float, float]:
        if self.price <= 0 or self.atr <= 0 or self.balance <= 0:
            return 0.0, 0.0, 0.0
        risk_amount = self.balance * (RISK_CONFIG["risk_per_trade_percent"] / 100)
        sl_dist = self.atr * RISK_CONFIG["atr_multiplier_for_sl"]
        sl_price = self.price - sl_dist
        tp_price = self.price + sl_dist * RISK_CONFIG["min_risk_reward_ratio"]
        qty = min(risk_amount / sl_dist, self.balance / self.price)
        return qty, sl_price, tp_price

class TradeExecutor:
    """Place orders using BinanceHandler."""

    def __init__(self, handler: BinanceHandler) -> None:
        self.handler = handler
        self.in_position = False

    def execute(self, signal: str, symbol: str, quote_asset: str, features: pd.DataFrame):
        if self.in_position:
            logging.info("Already in position; skipping")
            return
        if signal not in {"BUY", "SELL"}:
            logging.info("Signal HOLD; no trade")
            return
        price = self.handler.get_current_price(symbol)
        if price is None:
            return
        atr = features.iloc[-1]["ATR_14"]
        balance = self.handler.get_account_balance(quote_asset)
        if balance <= 10:
            logging.warning("Insufficient balance")
            return
        qty, sl, tp = PositionSizer(balance, price, atr).size()
        qty = round(qty, 5)
        if qty <= 0:
            logging.info("Position size too small")
            return
        self.handler.place_oco(symbol, qty, tp, sl, sl * 0.998)
        self.in_position = True

# ---------------------------------------------------------------------------
# Main Bot Loop
# ---------------------------------------------------------------------------

def run_bot():
    symbol = TRADING_CONFIG["trade_symbol"]
    timeframe = TRADING_CONFIG["timeframe"]
    hours = TRADING_CONFIG["klines_limit"]
    quote_asset = TRADING_CONFIG["quote_asset"]

    handler = BinanceHandler()
    executor = TradeExecutor(handler)

    while True:
        logging.info("Fetching data and generating signal...")
        df = handler.get_historical_data(symbol, timeframe, hours)
        features = FeatureGenerator(df).generate()
        signal = SignalCombiner(features).final_signal()
        logging.info(f"Final signal: {signal}")
        executor.execute(signal, symbol, quote_asset, features)
        tf_map = {"m": 60, "h": 3600, "d": 86400}
        sleep_s = int(timeframe[:-1]) * tf_map[timeframe[-1]]
        logging.info("Sleeping...")
        time.sleep(sleep_s)

if __name__ == "__main__":
    run_bot()
