# config.py

# --- API Configuration ---
# IMPORTANT: Use the TESTNET for development and testing.
import os
from dotenv import load_dotenv

load_dotenv()

API_CONFIG = {
    "use_testnet": True,
    "binance_api_key": os.getenv("BINANCE_KEY", "your_binance_key"),
    "binance_api_secret": os.getenv("BINANCE_SECRET", "your_binance_secret"),
    "glassnode_api_key": os.getenv("GLASSNODE_API_KEY", "your_glassnode_api_key"),
    "news_api_key": os.getenv("NEWS_API_KEY", "your_newsapi_org_key"),
}

# --- Trading Parameters ---
TRADING_CONFIG = {
    "trade_symbol": "BTCUSDT",
    "asset": "BTC",
    "quote_asset": "USDT",
    "timeframe": "1h",
    "klines_limit": 500,
}

# --- Risk Management ---
RISK_CONFIG = {
    "risk_per_trade_percent": 1.0,
    "atr_period_for_sl": 14,
    "atr_multiplier_for_sl": 2.0,
    "min_risk_reward_ratio": 2.0
}

# --- Signal Generation Weights ---
SIGNAL_WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.20,
    "on_chain": 0.25,
    "sentiment": 0.15,
    "forecast": 0.10
}

# --- Model Configuration ---
SENTIMENT_MODEL_NAME = "ProsusAI/finbert"
