# risk_management/position_sizer.py
import logging
from config import RISK_CONFIG

class PositionSizer:
    """Calculates the appropriate position size based on risk parameters."""

    def __init__(self, balance, current_price, atr):
        self.balance = balance
        self.current_price = current_price
        self.atr = atr
        self.risk_percent = RISK_CONFIG["risk_per_trade_percent"]
        self.atr_multiplier = RISK_CONFIG["atr_multiplier_for_sl"]

    def calculate_position_size(self):
        """Returns the quantity of the base asset to trade."""
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
        max_position_from_balance = self.balance / self.current_price
        final_position_size = min(position_size, max_position_from_balance)
        logging.info(
            f"💰 Position Sizing: Risk Amount=${risk_amount_usd:.2f}, SL Distance=${stop_loss_distance:.2f}, Size={final_position_size:.6f}"
        )
        return final_position_size, stop_loss_price, take_profit_price
