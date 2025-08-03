# main.py
import time
import logging
from datetime import datetime
from config import TRADING_CONFIG
from data_ingestion.binance_client import BinanceHandler
from data_ingestion.feature_generator import FeatureGenerator
from modeling.signal_combiner import SignalCombiner
from execution.trade_executor import TradeExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            feature_gen = FeatureGenerator(df)
            features_df = feature_gen.generate_features()
            combiner = SignalCombiner(features_df)
            final_signal = combiner.generate_final_signal()
            logging.info(f"🎯 Final Signal: {final_signal}")
            trade_executor.execute_trade(final_signal, symbol, quote_asset, features_df)
            logging.info("--- Sleeping until the next candle... ---")
            timeframe_seconds = {'m': 60, 'h': 3600, 'd': 86400}
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
    run_trading_bot()
