"""Combine various indicator signals into a final BUY/SELL/HOLD decision."""
import logging
from config import SIGNAL_WEIGHTS

class SignalCombiner:
    """Combines multiple signals using weighted scores."""

    def __init__(self, data):
        self.data = data.iloc[-1]
        self.weights = SIGNAL_WEIGHTS

    def _get_trend_signal(self):
        ema_short = self.data['EMA_50']
        ema_long = self.data['EMA_200']
        if ema_short > ema_long:
            return 1
        elif ema_short < ema_long:
            return -1
        return 0

    def _get_momentum_signal(self):
        rsi = self.data['RSI_14']
        if rsi > 70:
            return -1
        elif rsi < 30:
            return 1
        return 0

    def _get_on_chain_signal(self):
        score = 0
        active_addresses = self.data['active_addresses']
        tx_count = self.data['transaction_count']
        if active_addresses > 700000:
            score += 0.5
        elif active_addresses < 300000:
            score -= 0.5
        if tx_count > 1000000:
            score += 0.5
        elif tx_count < 500000:
            score -= 0.5
        return max(min(score, 1), -1)

    def _get_sentiment_signal(self):
        return self.data['sentiment_score']

    def _get_forecast_signal(self):
        current_price = self.data['Close']
        forecast_price = self.data['arima_forecast']
        if forecast_price > current_price * 1.005:
            return 1
        elif forecast_price < current_price * 0.995:
            return -1
        return 0

    def generate_final_signal(self):
        signals = {
            'trend': self._get_trend_signal(),
            'momentum': self._get_momentum_signal(),
            'on_chain': self._get_on_chain_signal(),
            'sentiment': self._get_sentiment_signal(),
            'forecast': self._get_forecast_signal(),
        }
        logging.info(f"📊 Individual Signal Scores: {signals}")
        final_score = 0
        for name, weight in self.weights.items():
            final_score += signals[name] * weight
        logging.info(f"⚖️ Final Weighted Score: {final_score:.4f}")
        if final_score >= 0.4:
            return 'BUY'
        elif final_score <= -0.4:
            return 'SELL'
        return 'HOLD'
