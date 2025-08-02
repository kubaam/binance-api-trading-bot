# Gemini Advanced Crypto Trader (Full Implementation)
This repository contains the source code for a sophisticated, multi-signal cryptocurrency trading bot designed to operate on Binance. The architecture integrates multiple layers of analysis, including advanced technical indicators, on-chain metrics, news sentiment analysis, and a time-series forecasting model.

⚠️ **IMPORTANT DISCLAIMER** ⚠️

* **Educational Use Only:** This is a proof-of-concept and an educational tool, NOT a plug-and-play, profitable trading system.
* **High Risk:** Algorithmic trading is extremely risky and can lead to significant financial loss. You are solely responsible for any actions taken by this bot.
* **Testnet Default:** The bot is configured by default to use the Binance Spot Testnet. You are solely responsible for any activity if you switch to the live market.
* **No Warranty:** The code is provided "as-is" without any warranty. Use at your own risk.

## Architecture Overview
The bot's architecture is modular, separating concerns into distinct packages:

* `config.py`: Central configuration for API keys, trading pairs, and strategy parameters. You must add your API keys here.
* `main.py`: The main application entry point that runs the trading loop.
* `data_ingestion/`: Handles all data collection and feature generation.
  * `binance_client.py`: Manages the connection to the Binance API.
  * `external_apis.py`: Manages connections to third-party APIs for on-chain data (Glassnode) and news/sentiment (NewsAPI).
  * `feature_generator.py`: Calculates technical indicators, fetches external data, and runs a predictive ARIMA model to create a comprehensive feature set.
* `modeling/`: Responsible for generating the final trading signal.
  * `signal_combiner.py`: Aggregates all features using a weighted scoring system to produce a final Buy/Sell/Hold signal.
* `risk_management/`: Manages position sizing and risk controls.
  * `position_sizer.py`: Calculates trade size based on volatility (ATR) and portfolio risk settings.
* `execution/`: Handles the placement and management of orders.
  * `trade_executor.py`: Interacts with Binance to execute trades using OCO (One-Cancels-the-Other) orders for simultaneous Stop-Loss and Take-Profit placement.

## Setup & Installation
1. **Clone the Repository**
   ```bash
   git clone <repository_url>
   cd gemini-advanced-crypto-trader
   ```
2. **Install Dependencies**
   It is highly recommended to use a Python virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```
3. **Configure the Bot**
   * **Get API Keys:**
     * Binance: Go to [https://testnet.binance.vision/](https://testnet.binance.vision/) to create free testnet API keys.
     * Glassnode: Sign up for a Glassnode account to get an API key for on-chain data.
     * NewsAPI: Sign up at [newsapi.org](https://newsapi.org) for a free developer API key to fetch news articles for sentiment analysis.
   * **Edit `config.py`:**
     * Enter your API keys in the respective sections.
     * Adjust trading parameters like `TRADE_SYMBOL` and `TIMEFRAME` as needed.

## How to Run
Ensure your virtual environment is activated and you have configured `config.py`.

```bash
python main.py
```

The bot will start, fetch initial data, generate features, and then enter a loop to check for trading opportunities at the interval defined by the timeframe in the configuration.
