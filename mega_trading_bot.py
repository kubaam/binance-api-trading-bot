"""Mega All-in-One Crypto Trading Bot (Educational).

This script combines the main features from the modular trading
bot into a single file. It fetches market data, augments it with
technical indicators, basic on-chain metrics, and news sentiment,
generates a trading signal, sizes a position, and executes the
trade using Binance Testnet.

⚠️  IMPORTANT: This code is for educational purposes only.
   Algorithmic trading is risky, and there are no guarantees of
   profit. Use at your own risk and always test on the Binance
   Spot Testnet before considering real funds.
"""

import logging
import time
import numpy as np
import pandas as pd
import pandas_ta as ta
from transformers import pipeline
from statsmodels.tsa.arima.model import ARIMA
import requests
from binance.client import Client
from binance.enums import SIDE_SELL, TIME_IN_FORCE_GTC, ORDER_TYPE_MARKET
from binance.exceptions import BinanceAPIException, BinanceOrderException

from config import API_CONFIG, TRADING_CONFIG, RISK_CONFIG, SIGNAL_WEIGHTS, SENTIMENT_MODEL_NAME

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BinanceHandler:
    """Simplified Binance API wrapper."""

    def __init__(self):
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
                "Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time",
                "Quote Asset Volume", "Number of Trades", "Taker Buy Base", "Taker Buy Quote", "Ignore",
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
            logging.error(f"OCO order failed: {e}; falling back to market order")
            return self.client.create_order(
                symbol=symbol,
                side=SIDE_SELL,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
            )


class ExternalAPIs:
    """Fetch on-chain metrics and news headlines."""

    def __init__(self):
        self.coinmetrics_base = "https://community-api.coinmetrics.io/v4"
        self.news_base = "https://newsapi.org/v2/everything"

    def get_on_chain_data(self, asset: str) -> dict:
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
        if not key or "YOUR" in key:
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


class FeatureGenerator:
    """Create features from price data and external sources."""

    def __init__(self, df: pd.DataFrame):
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
            self.df["arima_forecast"] = forecast
        except Exception as e:
            logging.error(f"ARIMA failed: {e}; using last close")
            self.df["arima_forecast"] = self.df["Close"].iloc[-1]
        return self

    def generate(self) -> pd.DataFrame:
        return self.add_ta().add_on_chain().add_sentiment().add_forecast().df.dropna()


class SignalCombiner:
    """Generate a buy/sell/hold signal from features."""

    def __init__(self, df: pd.DataFrame):
        self.data = df.iloc[-1]

    def _trend(self):
        short = self.data["EMA_50"]
        long = self.data["EMA_200"]
        return 1 if short > long else -1 if short < long else 0

    def _momentum(self):
        rsi = self.data["RSI_14"]
        return -1 if rsi > 70 else 1 if rsi < 30 else 0

    def _on_chain(self):
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

    def _sentiment(self):
        return self.data["sentiment_score"]

    def _forecast(self):
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


class PositionSizer:
    """Simple ATR-based position sizing."""

    def __init__(self, balance: float, price: float, atr: float):
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
    """Place orders using the BinanceHandler."""

    def __init__(self, handler: BinanceHandler):
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
