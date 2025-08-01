"""Binance API handler for fetching data and placing orders."""
import logging
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException, BinanceOrderException
import pandas as pd
from config import API_CONFIG

class BinanceHandler:
    """Handles all interactions with the Binance API."""

    def __init__(self):
        self.use_testnet = API_CONFIG["use_testnet"]
        self.api_key = API_CONFIG["binance_api_key"]
        self.api_secret = API_CONFIG["binance_api_secret"]

        if not self.api_key or "YOUR" in self.api_key:
            raise ValueError("Binance API key is not set in config.py")

        try:
            self.client = Client(self.api_key, self.api_secret, testnet=self.use_testnet)
            self.client.ping()
            logging.info("✅ Initialized Binance client successfully.")
        except BinanceAPIException as e:
            logging.error(f"❌ Binance API Error: {e}")
            raise

    def get_historical_data(self, symbol: str, timeframe: str, limit: int):
        """Fetch historical klines and return a pandas DataFrame."""
        try:
            klines = self.client.get_historical_klines(symbol, timeframe, f"{limit} hours ago UTC")
            df = pd.DataFrame(klines, columns=[
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time',
                'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base',
                'Taker Buy Quote', 'Ignore'
            ])
            for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume']:
                df[col] = pd.to_numeric(df[col])
            df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
            df.set_index('Open Time', inplace=True)
            logging.info(f"📈 Successfully fetched {len(df)} klines for {symbol}.")
            return df
        except BinanceAPIException as e:
            logging.error(f"❌ Error fetching historical data: {e}")
            return None

    def get_account_balance(self, asset: str) -> float:
        """Retrieve free balance of the given asset."""
        try:
            balance = self.client.get_asset_balance(asset=asset)
            return float(balance['free'])
        except BinanceAPIException as e:
            logging.error(f"❌ Error fetching account balance for {asset}: {e}")
            return 0.0

    def get_current_price(self, symbol: str):
        """Get the latest ticker price."""
        try:
            return float(self.client.get_symbol_ticker(symbol=symbol)['price'])
        except BinanceAPIException as e:
            logging.error(f"❌ Could not get current price for {symbol}: {e}")
            return None

    def place_oco_order(self, symbol: str, side: str, quantity: float, price: float, stop_price: float, stop_limit_price: float):
        """Place an OCO order."""
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
            logging.warning("OCO order failed. Placing a simple MARKET order instead.")
            return self.place_market_order(symbol, side, quantity)

    def place_market_order(self, symbol: str, side: str, quantity: float):
        """Place a MARKET order."""
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
