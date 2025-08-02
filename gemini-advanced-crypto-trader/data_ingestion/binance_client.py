# data_ingestion/binance_client.py
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

    def get_historical_data(self, symbol, timeframe, limit):
        """Fetches historical kline data and returns it as a pandas DataFrame."""
        try:
            klines = self.client.get_historical_klines(symbol, timeframe, f"{limit} hours ago UTC")
            columns = [
                'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close Time', 'Quote Asset Volume', 'Number of Trades',
                'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
            ]
            df = pd.DataFrame(klines, columns=columns)
            numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume']
            df[numeric_cols] = df[numeric_cols].astype(float)
            df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
            df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')
            df.set_index('Open Time', inplace=True)
            logging.info(f"📈 Successfully fetched {len(df)} klines for {symbol}.")
            return df
        except BinanceAPIException as e:
            logging.error(f"❌ Error fetching historical data: {e}")
            return None

    def get_account_balance(self, asset):
        """Retrieves the free balance of a specific asset."""
        try:
            balance = self.client.get_asset_balance(asset=asset)
            return float(balance['free'])
        except BinanceAPIException as e:
            logging.error(f"❌ Error fetching account balance for {asset}: {e}")
            return 0.0

    def get_current_price(self, symbol):
        """Gets the most recent price for a symbol."""
        try:
            return float(self.client.get_symbol_ticker(symbol=symbol)['price'])
        except BinanceAPIException as e:
            logging.error(f"❌ Could not get current price for {symbol}: {e}")
            return None

    def place_oco_order(self, symbol, side, quantity, price, stop_price, stop_limit_price):
        """Places an OCO (One-Cancels-the-Other) order."""
        try:
            logging.info(f"➡️  Placing OCO {side} order for {quantity} of {symbol}...")
            order = self.client.create_oco_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price,  # This is the Take Profit price
                stopPrice=stop_price,  # This is the Stop Loss trigger price
                stopLimitPrice=stop_limit_price,  # Price at which the Stop Loss Limit order will be placed
                stopLimitTimeInForce=TIME_IN_FORCE_GTC
            )
            logging.info(f"✅ OCO Order placed successfully: {order}")
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            logging.error(f"❌ Error placing OCO order: {e}")
            logging.warning("OCO order failed. Placing a simple MARKET order instead.")
            return self.place_market_order(symbol, side, quantity)

    def place_market_order(self, symbol, side, quantity):
        """Places a simple MARKET order."""
        try:
            logging.info(f"➡️  Placing MARKET {side} order for {quantity} of {symbol}...")
            order = self.client.create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=quantity
            )
            logging.info(f"✅ MARKET Order placed successfully: {order}")
            return order
        except (BinanceAPIException, BinanceOrderException) as e:
            logging.error(f"❌ Error placing MARKET order: {e}")
            return None
