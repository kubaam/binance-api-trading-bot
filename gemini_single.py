"""Unified Gemini Advanced Crypto Trader
This file bundles all core functionality in a single script.
"""

import logging
import time
import os
import pandas as pd
import pandas_ta as ta
import numpy as np
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException
from transformers import pipeline
from statsmodels.tsa.arima.model import ARIMA
import requests
from dotenv import load_dotenv

load_dotenv()

API_CONFIG = {
    "use_testnet": True,
    "binance_api_key": os.getenv("BINANCE_KEY", "your_binance_key"),
    "binance_api_secret": os.getenv("BINANCE_SECRET", "your_binance_secret"),
    "glassnode_api_key": os.getenv("GLASSNODE_API_KEY", "your_glassnode_api_key"),
    "news_api_key": os.getenv("NEWS_API_KEY", "your_newsapi_org_key"),
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

class BinanceHandler:
    def __init__(self):
        self.use_testnet = API_CONFIG["use_testnet"]
        self.client = Client(
            API_CONFIG["binance_api_key"],
            API_CONFIG["binance_api_secret"],
            testnet=self.use_testnet,
        )
        self.client.ping()
        logging.info("✅ Initialized Binance client successfully.")

    def get_historical_data(self, symbol, timeframe, limit):
        try:
            klines = self.client.get_historical_klines(symbol, timeframe, f"{limit} hours ago UTC")
            columns = [
                "Open Time", "Open", "High", "Low", "Close", "Volume",
                "Close Time", "Quote Asset Volume", "Number of Trades",
                "Taker Buy Base", "Taker Buy Quote", "Ignore",
            ]
            df = pd.DataFrame(klines, columns=columns)
            numeric = ["Open", "High", "Low", "Close", "Volume", "Quote Asset Volume", "Taker Buy Base", "Taker Buy Quote"]
            df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
            df["Open Time"] = pd.to_datetime(df["Open Time"], unit="ms")
            df.set_index("Open Time", inplace=True)
            logging.info(f"📈 Successfully fetched {len(df)} klines for {symbol}.")
            return df
        except BinanceAPIException as e:
            logging.error(f"❌ Error fetching historical data: {e}")
            return None

    def get_account_balance(self, asset):
        try:
            balance = self.client.get_asset_balance(asset=asset)
            return float(balance["free"])
        except BinanceAPIException as e:
            logging.error(f"❌ Error fetching account balance for {asset}: {e}")
            return 0.0

    def get_current_price(self, symbol):
        try:
            return float(self.client.get_symbol_ticker(symbol=symbol)["price"])
        except BinanceAPIException as e:
            logging.error(f"❌ Could not get current price for {symbol}: {e}")
            return None

    def place_oco_order(self, symbol, side, quantity, price, stop_price, stop_limit_price):
        try:
            logging.info(f"➡️  Placing OCO {side} order for {quantity} of {symbol}...")
            order = self.client.create_oco_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,
                stopPrice=stop_price,
                stopLimitPrice=stop_limit_price,
                stopLimitTimeInForce=TIME_IN_FORCE_GTC,
            )
            logging.info(f"✅ OCO Order placed successfully: {order}")
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            logging.error(f"❌ Error placing OCO order: {e}")
            logging.warning("OCO order failed. Placing a MARKET order instead.")
            return self.place_market_order(symbol, side, quantity)

    def place_market_order(self, symbol, side, quantity):
        try:
            logging.info(f"➡️  Placing MARKET {side} order for {quantity} of {symbol}...")
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity,
            )
            logging.info(f"✅ MARKET Order placed successfully: {order}")
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            logging.error(f"❌ Error placing MARKET order: {e}")
            return None

class ExternalAPIs:
    def __init__(self):
        self.glassnode_api_key = API_CONFIG["glassnode_api_key"]
        self.news_api_key = API_CONFIG["news_api_key"]
        self.glassnode_base_url = "https://api.glassnode.com/v1/metrics"
        self.news_base_url = "https://newsapi.org/v2/everything"

    def get_on_chain_data(self, asset):
        if not self.glassnode_api_key or "your" in self.glassnode_api_key.lower():
            logging.warning("⚠️ Glassnode API key not set. Returning neutral on-chain data.")
            return {"mvrv_z_score": 0, "sopr": 1.0, "nupl": 0.5}
        metrics = {
            "mvrv_z_score": "/market/mvrv_z_score",
            "sopr": "/indicators/sopr",
            "nupl": "/indicators/net_unrealized_profit_loss",
        }
        result = {}
        for name, endpoint in metrics.items():
            try:
                params = {"a": asset, "api_key": self.glassnode_api_key, "i": "24h"}
                r = requests.get(f"{self.glassnode_base_url}{endpoint}", params=params)
                r.raise_for_status()
                data = r.json()
                result[name] = data[-1]["v"] if data else 0
            except requests.exceptions.RequestException as e:
                logging.error(f"❌ Error fetching {name} from Glassnode: {e}")
                result[name] = 0 if name == "mvrv_z_score" else (1.0 if name == "sopr" else 0.5)
        logging.info(f"🔗 Fetched on-chain data: {result}")
        return result

    def get_news_headlines(self, query):
        if not self.news_api_key or "your" in self.news_api_key.lower():
            logging.warning("⚠️ NewsAPI key not set. Skipping sentiment analysis.")
            return []
        try:
            params = {"q": query, "apiKey": self.news_api_key, "language": "en", "sortBy": "publishedAt", "pageSize": 20}
            r = requests.get(self.news_base_url, params=params)
            r.raise_for_status()
            articles = r.json().get("articles", [])
            headlines = [a["title"] for a in articles]
            logging.info(f"📰 Fetched {len(headlines)} headlines for '{query}'.")
            return headlines
        except requests.exceptions.RequestException as e:
            logging.error(f"❌ Error fetching news from NewsAPI: {e}")
            return []

class FeatureGenerator:
    def __init__(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("A valid DataFrame must be provided.")
        self.df = df.copy()
        self.external_api = ExternalAPIs()
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)

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
        oc = self.external_api.get_on_chain_data(TRADING_CONFIG["asset"])
        for k, v in oc.items():
            self.df[k] = v
        return self

    def add_sentiment_data(self):
        headlines = self.external_api.get_news_headlines(TRADING_CONFIG["asset"])
        if not headlines:
            self.df["sentiment_score"] = 0
            return self
        sentiments = self.sentiment_pipeline(headlines)
        score_map = {"negative": -1, "neutral": 0, "positive": 1}
        scores = [score_map.get(s["label"].lower(), 0) * s["score"] for s in sentiments]
        self.df["sentiment_score"] = float(np.mean(scores)) if scores else 0
        logging.info(f"❤️ Calculated average sentiment score: {self.df['sentiment_score'].iloc[-1]:.4f}")
        return self

    def add_arima_forecast(self):
        close_prices = self.df["Close"].astype(float).values
        try:
            model = ARIMA(close_prices, order=(5, 1, 0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]
            self.df["arima_forecast"] = forecast
            logging.info(f"🧠 ARIMA forecast for next period: {forecast:.2f}")
        except Exception as e:
            logging.error(f"❌ ARIMA model failed: {e}. Using last close price as forecast.")
            self.df["arima_forecast"] = close_prices[-1]
        return self

    def generate_features(self):
        self.add_technical_indicators()
        self.add_onchain_data()
        self.add_sentiment_data()
        self.add_arima_forecast()
        self.df.dropna(inplace=True)
        logging.info("✅ Feature generation complete.")
        return self.df

class SignalCombiner:
    def __init__(self, data: pd.DataFrame):
        self.data = data.iloc[-1]
        self.weights = SIGNAL_WEIGHTS

    def _get_trend_signal(self):
        ema_short = self.data["EMA_50"]
        ema_long = self.data["EMA_200"]
        if ema_short > ema_long:
            return 1
        elif ema_short < ema_long:
            return -1
        return 0

    def _get_momentum_signal(self):
        rsi = self.data["RSI_14"]
        if rsi > 70:
            return -1
        elif rsi < 30:
            return 1
        return 0

    def _get_on_chain_signal(self):
        score = 0
        mvrv_z = self.data["mvrv_z_score"]
        sopr = self.data["sopr"]
        if mvrv_z > 7.0:
            score -= 1
        elif mvrv_z < 0.1:
            score += 1
        if sopr < 1.0:
            score += 0.5
        return max(min(score, 1), -1)

    def _get_sentiment_signal(self):
        return self.data["sentiment_score"]

    def _get_forecast_signal(self):
        current_price = self.data["Close"]
        forecast_price = self.data["arima_forecast"]
        if forecast_price > current_price * 1.005:
            return 1
        elif forecast_price < current_price * 0.995:
            return -1
        return 0

    def generate_final_signal(self):
        signals = {
            "trend": self._get_trend_signal(),
            "momentum": self._get_momentum_signal(),
            "on_chain": self._get_on_chain_signal(),
            "sentiment": self._get_sentiment_signal(),
            "forecast": self._get_forecast_signal(),
        }
        logging.info(f"📊 Individual Signal Scores: {signals}")
        final_score = sum(signals[name] * w for name, w in self.weights.items())
        logging.info(f"⚖️ Final Weighted Score: {final_score:.4f}")
        if final_score >= 0.4:
            return "BUY"
        elif final_score <= -0.4:
            return "SELL"
        return "HOLD"

class PositionSizer:
    def __init__(self, balance, current_price, atr):
        self.balance = balance
        self.current_price = current_price
        self.atr = atr
        self.risk_percent = RISK_CONFIG["risk_per_trade_percent"]
        self.atr_multiplier = RISK_CONFIG["atr_multiplier_for_sl"]

    def calculate_position_size(self):
        if self.current_price <= 0 or self.atr <= 0 or self.balance <= 0:
            logging.warning("⚠️ Cannot calculate position size due to invalid price, ATR, or balance.")
            return 0.0, 0.0, 0.0
        risk_amount_usd = self.balance * (self.risk_percent / 100)
        stop_loss_distance = self.atr * self.atr_multiplier
        stop_loss_price = self.current_price - stop_loss_distance
        take_profit_price = self.current_price + (stop_loss_distance * RISK_CONFIG["min_risk_reward_ratio"])
        if stop_loss_distance == 0:
            return 0.0, 0.0, 0.0
        position_size = risk_amount_usd / stop_loss_distance
        max_position = self.balance / self.current_price
        final_size = min(position_size, max_position)
        logging.info(
            f"💰 Position Sizing: Risk Amount=${risk_amount_usd:.2f}, SL Distance=${stop_loss_distance:.2f}, Size={final_size:.6f}"
        )
        return final_size, stop_loss_price, take_profit_price

class TradeExecutor:
    def __init__(self, binance_handler: BinanceHandler):
        self.binance_handler = binance_handler
        self.in_position = False

    def execute_trade(self, signal, symbol, quote_asset, features_df):
        if self.in_position:
            logging.info("🧘 Already in a position. Holding.")
            return
        if signal in ("BUY", "SELL"):
            current_price = self.binance_handler.get_current_price(symbol)
            if not current_price:
                return
            atr = features_df.iloc[-1]["ATRr_14"] if "ATRr_14" in features_df.columns else None
            if atr is None:
                logging.warning("⚠️ ATR value not found. Cannot size position.")
                return
            balance = self.binance_handler.get_account_balance(quote_asset)
            if balance <= 10:
                logging.warning(f"⚠️ {quote_asset} balance ({balance}) too low to trade.")
                return
            sizer = PositionSizer(balance=balance, current_price=current_price, atr=atr)
            quantity, sl_price, tp_price = sizer.calculate_position_size()
            quantity = round(quantity, 5)
            sl_price = round(sl_price, 2)
            tp_price = round(tp_price, 2)
            if quantity <= 0:
                logging.info("ℹ️ Position size is too small to execute a trade.")
                return
            if signal == "BUY":
                self.binance_handler.place_oco_order(
                    symbol=symbol,
                    side=SIDE_SELL,
                    quantity=quantity,
                    price=tp_price,
                    stopPrice=sl_price,
                    stopLimitPrice=sl_price * 0.998,
                )
                self.in_position = True
            elif signal == "SELL":
                logging.warning("⚠️ SELL signal received, but shorting is not implemented for Spot market in this example.")
        else:
            logging.info("🧘 Signal is HOLD. No action taken.")


def run_trading_bot():
    symbol = TRADING_CONFIG["trade_symbol"]
    quote_asset = TRADING_CONFIG["quote_asset"]
    timeframe = TRADING_CONFIG["timeframe"]
    klines_limit = TRADING_CONFIG["klines_limit"]
    try:
        binance_handler = BinanceHandler()
        trade_executor = TradeExecutor(binance_handler)
    except Exception as e:
        logging.critical(f"🔥 Failed to initialize critical components: {e}")
        return
    logging.info("🚀 Starting Gemini Advanced Trading Bot...")
    logging.info(f"Trading Pair: {symbol} | Timeframe: {timeframe}")
    logging.info("---")
    while True:
        try:
            logging.info("\n --- Checking for new signals ---")
            df = binance_handler.get_historical_data(symbol, timeframe, klines_limit)
            if df is None or df.empty:
                raise ValueError("Failed to fetch market data.")
            features_df = FeatureGenerator(df).generate_features()
            final_signal = SignalCombiner(features_df).generate_final_signal()
            logging.info(f"🎯 Final Signal: {final_signal}")
            trade_executor.execute_trade(final_signal, symbol, quote_asset, features_df)
            logging.info("--- Sleeping until the next candle... ---")
            timeframe_seconds = {"m": 60, "h": 3600, "d": 86400}
            sleep_duration = int(timeframe[:-1]) * timeframe_seconds[timeframe[-1]]
            time.sleep(sleep_duration)
        except KeyboardInterrupt:
            logging.info("\n🛑 Bot stopped by user.")
            break
        except Exception as e:
            logging.error(f"🔥 An unexpected error occurred in the main loop: {e}", exc_info=True)
            logging.info("--- Retrying in 60 seconds... ---")
            time.sleep(60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_trading_bot()
