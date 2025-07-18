# Trading Bot

This repository contains a single-file Python trading bot implementation.

## Requirements

Install Python 3.8 or later. Required packages are listed in `requirements.txt`.
You can install them using:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the repository root with your API keys:

```bash
BINANCE_KEY="your_binance_key"
BINANCE_SECRET="your_binance_secret"
OPENAI_API_KEY="your_openai_key"
DISCORD_WEBHOOK_URL="your_discord_webhook"
```

A sample file is provided as `.env.sample`.

## Running

Execute the bot with:

```bash
python trading_bot_unified.py
```

On Windows you can also run `start.bat`.

