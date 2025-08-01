import logging
from binance.enums import *

from risk_management.position_sizer import PositionSizer


class TradeExecutor:
    """Handle logic for executing trades."""

    def __init__(self, binance_handler):
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

            atr = features_df.iloc[-1]['ATRr_14'] if 'ATRr_14' in features_df.columns else None
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
                    stopLimitPrice=sl_price * 0.998
                )
                self.in_position = True
            elif signal == "SELL":
                logging.warning(
                    "⚠️ SELL signal received, but shorting is not implemented for Spot market in this example."
                )
        else:
            logging.info("🧘 Signal is HOLD. No action taken.")
