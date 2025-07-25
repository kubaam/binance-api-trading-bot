# Trading Bot

This repository contains a single-file Python trading bot implementation.

## Requirements

Install Python 3.8 or later. Required packages are listed in `requirements.txt`.
You can install them using:

```bash
pip install -r requirements.txt
```

For optimal performance the bot uses `uvloop` as the event loop when available.
`uvloop` does not currently support Windows, so the bot falls back to the default
asyncio event loop on that platform.

## Configuration

Create a `.env` file in the repository root with your API keys:

```bash
BINANCE_KEY="your_binance_key"
BINANCE_SECRET="your_binance_secret"
OPENAI_API_KEY="your_openai_key"
DISCORD_WEBHOOK_URL="your_discord_webhook"
```

A sample file is provided as `.env.sample`.
All other settings are loaded from `config.json` and changes to that file are
applied on the fly. Keep your API keys only in `.env`; they are never written
back to `config.json`.

## API Rate Limits

The bot throttles requests to respect Binance's published limits. By default the
settings in `config.json` allow up to 20 API calls per second and 10 orders per
second (the equivalent of 100 orders every 10 seconds). Adjust
`api_requests_per_second` and `order_requests_per_second` if stricter limits are
required for your account.

## Running

Execute the bot with:

```bash
python trading_bot_unified.py
```

On Windows you can also run `start.bat`.

## Dynamic Risk Management

The bot now includes an optional self-adjusting risk mode. When enabled (the
default), the position size risk percentage increases after a series of winning
trades and decreases after losses, bounded by the limits defined in
`RiskSettings`.

## Trailing Stops

Open positions automatically use a trailing stop that moves up as the price
increases. The trailing distance is equal to the initial risk amount,
allowing profits to be locked in while letting winners run.

## Moving Average Crossover

An additional strategy uses a fast/slow simple moving average crossover
(`sma_fast` and `sma_slow` in `config.json`) to help identify strong trends.
Signals are only generated when the fast average crosses the slow average.

## Automatic Symbol Refresh

Every 10 minutes the bot fetches the top trading pairs by volume and
restarts its market data streams if the list of symbols has changed. This
keeps the strategy focused on the most liquid markets without manual
intervention.

## Market Regime Detection

The bot trains a Hidden Markov Model (HMM) to categorize market regimes.
Training now uses historical data from several tickers by default
(`BTCUSDC`, `ETHUSDC` and `BNBUSDC`) for a more robust model. You can
override this list with the `REGIME_TRAINING_TICKERS` environment
variable.

## Troubleshooting

If the bot exits with a `TimeoutError` during startup, it usually means the
Binance API could not be reached. Verify that your network connection allows
outbound HTTPS requests to `api.binance.com` and try again. Some environments may
require a proxy or VPN to access the API.

