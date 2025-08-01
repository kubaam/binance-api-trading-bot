"""Configuration settings for the Gemini Advanced Crypto Trader."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load variables from a .env file if present
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# --- API Configuration ---
# IMPORTANT: Use the TESTNET for development and testing.
# Get your testnet keys from https://testnet.binance.vision/
API_CONFIG = {
    "use_testnet": True,
    "binance_api_key": os.environ.get("BINANCE_KEY", "YOUR_BINANCE_TESTNET_API_KEY"),
    "binance_api_secret": os.environ.get("BINANCE_SECRET", "YOUR_BINANCE_TESTNET_API_SECRET"),
    "glassnode_api_key": os.environ.get("GLASSNODE_API_KEY", "YOUR_GLASSNODE_API_KEY"),
    "news_api_key": os.environ.get("NEWS_API_KEY", "YOUR_NEWSAPI_ORG_KEY"),
}

# Expose additional environment-based tokens if needed elsewhere
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# --- Trading Parameters ---
TRADING_CONFIG = {
    "trade_symbol": "BTCUSDT",
    "asset": "BTC",
    "quote_asset": "USDT",
    "timeframe": "1h",       # e.g., '1m', '5m', '15m', '1h', '4h', '1d'
    "klines_limit": 500,     # Number of historical candles to fetch for analysis
}

# --- Risk Management ---
RISK_CONFIG = {
    "risk_per_trade_percent": 1.0,  # Max % of total portfolio to risk on a single trade
    "atr_period_for_sl": 14,        # ATR period for calculating Stop Loss
    "atr_multiplier_for_sl": 2.0,   # ATR multiplier for Stop Loss distance
    "min_risk_reward_ratio": 2.0    # Minimum Take Profit distance as a multiple of SL distance
}

# --- Signal Generation Weights ---
# These weights determine the influence of each analysis type on the final signal.
# They should sum to 1.0.
SIGNAL_WEIGHTS = {
    "trend": 0.30,
    "momentum": 0.20,
    "on_chain": 0.25,
    "sentiment": 0.15,
    "forecast": 0.10,
}

# --- Model Configuration ---
SENTIMENT_MODEL_NAME = "ProsusAI/finbert"  # Hugging Face model for financial sentiment
