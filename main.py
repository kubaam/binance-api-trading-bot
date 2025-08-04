"""Regime-aware trading bot implementation.

This module integrates configuration management, exchange interaction,
market data handling, strategy execution and risk management into a single
script. It is adapted from a full V8.6 implementation with placeholders
removed so that credentials must be supplied via environment variables or a
``config.json`` file.
"""

import asyncio
import json
import logging
import math
import sys
import os
import re
import pickle
import random
import csv
import signal
import time
import contextlib
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from typing import (
    Dict,
    List,
    Any,
    Tuple,
    Optional,
    Type,
    Union,
    AsyncGenerator,
    Coroutine,
)
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field

import httpx
import aiohttp
import polars as pl
import polars_talib as plta
import numpy as np
import joblib
from binance import AsyncClient, BinanceSocketManager
from binance.enums import (
    KLINE_INTERVAL_15MINUTE,
    KLINE_INTERVAL_1DAY,
    ORDER_TYPE_MARKET,
    SIDE_BUY,
    SIDE_SELL,
)
from binance.exceptions import BinanceAPIException, BinanceRequestException
from openai import AsyncOpenAI
import hashlib
from collections import OrderedDict
from rich.logging import RichHandler
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from hmmlearn.hmm import GaussianHMM
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
import structlog
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from colorama import Fore, Style, init

# =============================================================================
# INITIALIZATION & LOGGING SETUP
# =============================================================================

init(autoreset=True)

if sys.platform != "win32":
    try:
        import uvloop
        uvloop.install()
    except ImportError:
        pass


class SafeFileHandler(logging.FileHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_notified = False

    def emit(self, record):
        try:
            super().emit(record)
            self.error_notified = False
        except OSError as e:
            if not self.error_notified:
                print(
                    f"{Fore.RED}CRITICAL: Logging to {self.baseFilename} failed due to OSError: {e}. "
                    f"Further file logging errors will be suppressed.",
                    file=sys.stderr,
                )
                self.error_notified = True
        except Exception as e:
            if not self.error_notified:
                print(
                    f"{Fore.RED}CRITICAL: An unexpected error occurred in file logger: {e}",
                    file=sys.stderr,
                )
                self.error_notified = True


def setup_logging(log_level: str = "INFO", is_production: bool = False):
    Path("data").mkdir(exist_ok=True)
    REPIPED_LOGGERS = [
        "binance",
        "websockets",
        "httpx",
        "aiohttp",
        "sqlalchemy",
        "yfinance",
    ]
    for logger_name in REPIPED_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    shared_processors: List[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    console_renderer = structlog.dev.ConsoleRenderer(colors=True)
    console_formatter = structlog.stdlib.ProcessorFormatter(processor=console_renderer)

    json_formatter = structlog.stdlib.ProcessorFormatter(
        processor=structlog.processors.JSONRenderer()
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    log_file_path = "data/bot_logs_v8.json"
    file_handler = SafeFileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(json_formatter)

    error_log_file_path = "data/bot_errors_v8.log"
    error_file_handler = SafeFileHandler(
        error_log_file_path, mode="a", encoding="utf-8"
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(json_formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_file_handler)
    root_logger.setLevel(log_level.upper())


log = structlog.get_logger("TradingBotV8.5-Final")

# =============================================================================
# CONFIGURATION & CORE DATA MODELS
# =============================================================================


class ObservableSettings(BaseSettings):
    def __init__(self, **data: Any):
        super().__init__(**data)
        object.__setattr__(self, "_change_callback", None)
        object.__setattr__(self, "_in_update", False)

    def set_change_callback(self, callback):
        object.__setattr__(self, "_change_callback", callback)
        for v in self.__dict__.values():
            if isinstance(v, ObservableSettings):
                v.set_change_callback(callback)

    def begin_update(self):
        object.__setattr__(self, "_in_update", True)
        for v in self.__dict__.values():
            if isinstance(v, ObservableSettings):
                v.begin_update()

    def end_update(self):
        for v in self.__dict__.values():
            if isinstance(v, ObservableSettings):
                v.end_update()
        object.__setattr__(self, "_in_update", False)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        cb = getattr(self, "_change_callback", None)
        if cb and not getattr(self, "_in_update", False):
            cb()


class StrategyType(str, Enum):
    TREND_MACD = "TREND_MACD"
    MEAN_REVERSION_RSI = "MEAN_REVERSION_RSI"
    ICHIMOKU_BREAKOUT = "ICHIMOKU_BREAKOUT"
    SMA_CROSSOVER = "SMA_CROSSOVER"
    COMBINED = "COMBINED"


class ApiSettings(ObservableSettings):
    binance_key: str = ""
    binance_secret: str = ""
    openai_api_key: str = ""
    discord_webhook_url: str = ""
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )


class TradingSettings(ObservableSettings):
    quote_asset: str = "USDT"
    interval: str = "1h"
    top_n_symbols: int = 100
    max_active_positions: int = 50
    initial_cash_balance: float = 1000.0
    initial_trade_amount: float = 100.0
    trading_mode: str = "PAPER"


class StrategySettings(ObservableSettings):
    model_config = SettingsConfigDict(env_prefix="STRATEGY_")
    data_fetch_limit: int = 750
    min_candles_for_analysis: int = 200
    signal_consensus_threshold: int = 1
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    rsi_entry_midline: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    sma_fast: int = 50
    sma_slow: int = 200
    ichimoku_tenkan: int = 9
    ichimoku_kijun: int = 26
    ichimoku_senkou_b: int = 52
    trend_ma_period: int = 100
    volatility_window: int = 20
    breakout_margin_pct: float = 0.00


class RiskSettings(ObservableSettings):
    initial_risk_per_trade_pct: float = 0.015
    max_risk_per_trade_pct: float = 0.025
    min_risk_per_trade_pct: float = 0.01
    risk_reward_ratio: float = 2.0
    atr_period_risk: int = 14
    atr_stop_multiplier: float = 2.0
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.05
    max_drawdown_pct: float = 0.20
    enable_concentration_veto: bool = True
    correlation_threshold: float = 0.80
    correlation_matrix_update_interval_sec: int = 3600
    dynamic_risk: bool = True


class DcaSettings(ObservableSettings):
    max_dca_orders: int = 1
    dca_buy_threshold_pct: float = 0.03
    dca_volume_scale: float = 1.0


class AISettings(ObservableSettings):
    enable_ai_decider: bool = True
    min_ai_confidence_score: float = 0.60


class RegimeSettings(ObservableSettings):
    enabled: bool = True
    model_path: Path = Path("data/market_regime_hmm.joblib")
    training_ticker: str = "BTCUSDT"
    training_tickers: List[str] = Field(
        default_factory=lambda: ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    )
    hmm_training_days: int = 2000
    high_risk_regimes: List[str] = field(default_factory=lambda: ["Bearish Volatile"])
    retraining_interval_sec: int = 86400


class BotSettings(ObservableSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )
    version: str = "8.7-Unified"
    log_level: str = "INFO"
    is_production: bool = Field(default_factory=lambda: not sys.stderr.isatty())
    db_path: Path = Path("data/trades_v8.db")
    correlation_matrix_path: Path = Path("data/correlation_matrix_v8.pkl")
    concurrent_api_calls: int = 10
    api_timeout_seconds: int = 15
    api_max_retries: int = 5
    api_backoff_factor: float = 1.5
    api_requests_per_second: int = 10
    order_requests_per_second: int = 10
    position_manage_interval_sec: int = 15
    symbol_refresh_interval_sec: int = 600
    portfolio_status_interval_sec: int = 60
    poll_interval_seconds: int = 60
    reweight_interval_hours: int = 24
    performance_window_hours: int = 720
    ensemble_execution_threshold: float = 0.7
    api: ApiSettings = ApiSettings()
    trading: TradingSettings = TradingSettings()
    strategy: StrategySettings = StrategySettings()
    risk: RiskSettings = RiskSettings()
    dca: DcaSettings = DcaSettings()
    ai: AISettings = AISettings()
    regime: RegimeSettings = RegimeSettings()


class ConfigManager:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.log = structlog.get_logger(self.__class__.__name__)
        self.config = self._load_or_create()
        self.config.set_change_callback(self.save)
        self.last_mtime = self.path.stat().st_mtime if self.path.exists() else 0
        self._watch_task: Optional[asyncio.Task] = None

    def _load_or_create(self) -> BotSettings:
        if self.path.exists():
            try:
                with self.path.open() as f:
                    data = json.load(f)
                data.pop("api", None)
                return BotSettings.model_validate(data)
            except Exception as e:
                self.log.error("Failed to load config.json, using defaults", error=e)
        cfg = BotSettings()
        try:
            data = cfg.model_dump(mode="json")
            data.pop("api", None)
            with self.path.open("w") as f:
                json.dump(data, f, indent=2)
            self.last_mtime = self.path.stat().st_mtime
        except Exception as e:
            self.log.error("Failed to write default config.json", error=e)
        return cfg

    def save(self, *_):
        try:
            data = self.config.model_dump(mode="json")
            data.pop("api", None)
            with self.path.open("w") as f:
                json.dump(data, f, indent=2)
            self.last_mtime = self.path.stat().st_mtime
        except Exception as e:
            self.log.error("Failed to write config.json", error=e)

    async def _watch_loop(self, interval: int = 1):
        while not SHUTDOWN_EVENT.is_set():
            try:
                if self.path.exists():
                    mtime = self.path.stat().st_mtime
                    if mtime != self.last_mtime:
                        self.log.info("Reloading configuration from file")
                        with self.path.open() as f:
                            data = json.load(f)
                        data.pop("api", None)
                        new_cfg = BotSettings.model_validate(data)
                        self.config.begin_update()
                        try:
                            self._apply_updates(self.config, new_cfg)
                        finally:
                            self.config.end_update()
                        self.last_mtime = mtime
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log.error("Error watching config file", error=e)

    def _apply_updates(self, current: ObservableSettings, new: ObservableSettings):
        for name in current.model_fields:
            cur_val = getattr(current, name)
            new_val = getattr(new, name)
            if isinstance(cur_val, ObservableSettings):
                self._apply_updates(cur_val, new_val)
            else:
                if cur_val != new_val:
                    setattr(current, name, new_val)

    def start_watch(self, interval: int = 1):
        if self._watch_task is None:
            self._watch_task = asyncio.create_task(self._watch_loop(interval))

    async def stop_watch(self):
        if self._watch_task:
            self._watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._watch_task


class Event:
    pass


@dataclass
class MarketEvent(Event):
    symbol: str
    data: pl.DataFrame


@dataclass
class SignalEvent(Event):
    symbol: str
    action: str
    price: float
    details: dict


@dataclass
class OrderEvent(Event):
    symbol: str
    side: str
    quantity: float
    order_type: str = ORDER_TYPE_MARKET
    quote_order_qty: Optional[float] = None


@dataclass
class FillEvent(Event):
    symbol: str
    side: str
    quantity: float
    price: float
    commission: float
    order_id: str
    raw_data: dict
    cummulative_quote_qty: float
    market_regime: Optional[str] = None
    strategy_id: Optional[str] = None


class BotState(Enum):
    AWAITING_SIGNAL = 1
    ORDER_PENDING = 2
    IN_POSITION = 3
    EXIT_PENDING = 4


@dataclass
class Position:
    symbol: str
    entry_price: float
    avg_price: float
    quantity: float
    side: str
    entry_time: datetime
    entry_strategy: List[str]
    stop_loss_price: float
    take_profit_price: float
    trailing_stop_price: Optional[float]
    market_regime_at_entry: str
    dca_count: int = 0


# =============================================================================
# MARKET REGIME DETECTION
# =============================================================================


class MarketRegimeDetector:
    def __init__(self, config: RegimeSettings, exchange_service: "ExchangeService"):
        self.config = config
        self.exchange = exchange_service
        self._log = structlog.get_logger(self.__class__.__name__)
        self.model: Optional[GaussianHMM] = None
        self.regime_names: Dict[int, str] = {}
        self.current_regime: str = "UNKNOWN"
        self.last_prediction_time = None

    async def _train_model(self):
        self._log.info("--- Starting HMM Model Training ---")
        try:
            tickers = self.config.training_tickers or [self.config.training_ticker]
            self._log.info(
                f"Fetching {self.config.hmm_training_days} days of historical data for HMM training.",
                tickers=tickers,
            )

            all_features = []
            for ticker in tickers:
                self._log.info("Fetching data for HMM training", ticker=ticker)

                data = await self.exchange.fetch_klines(
                    symbol=ticker,
                    interval=KLINE_INTERVAL_1DAY,
                    limit=self.config.hmm_training_days,
                )

                if data.is_empty() or data.height < 100:
                    raise IOError(
                        f"Not enough historical data for {ticker} from Binance. Received {data.height} rows."
                    )

                returns = (data["close"] / data["close"].shift(1)) - 1
                log_returns = returns.log1p()
                volatility = log_returns.rolling_std(window_size=21)

                features_df = pl.DataFrame(
                    {"log_returns": log_returns, "volatility": volatility}
                ).drop_nulls()

                feat = features_df.to_numpy()

                if len(feat) < 100:
                    raise IOError(
                        f"Not enough feature data points after calculation for {ticker}. Have {len(feat)} points."
                    )

                self._log.info(f"Prepared {len(feat)} data points for {ticker}.")
                all_features.append(feat)

            features = np.vstack(all_features)

            self._log.info(f"Total concatenated feature points: {len(features)}")
            self._log.info("Training GaussianHMM with 3 components...")
            model = GaussianHMM(
                n_components=3,
                covariance_type="full",
                n_iter=1000,
                random_state=42,
                verbose=False,
            )

            await asyncio.to_thread(model.fit, features)

            self._log.info("HMM model training complete.")

            joblib.dump(model, self.config.model_path)
            self._log.info(
                f"{Fore.GREEN}✅ HMM model saved successfully!",
                path=str(self.config.model_path),
            )

        except Exception as e:
            self._log.critical("HMM model training failed.", error=e, exc_info=True)
            raise

    async def initialize(self):
        if not self.config.enabled:
            self._log.warning("Regime detection is disabled in settings.")
            return

        if not self.config.model_path.exists():
            self._log.warning(
                f"{Fore.YELLOW}HMM model not found. Initiating automatic training...{Style.RESET_ALL}"
            )
            try:
                await self._train_model()
                self._log.info("Automatic HMM training completed successfully.")
            except Exception as e:
                self._log.critical(
                    "Automatic HMM training failed. Regime detection will be disabled.",
                    error=e,
                )
                self.config.enabled = False
                return

        try:
            self.model = joblib.load(self.config.model_path)
            self._log.info(
                "HMM model loaded successfully.", path=str(self.config.model_path)
            )
            self._label_regimes()
        except Exception as e:
            self._log.error("Failed to load HMM model.", error=e, exc_info=True)
            self.config.enabled = False

    def reload_model(self):
        self._log.info("Reloading HMM model...")
        if not self.config.enabled or not self.config.model_path.exists():
            self._log.warning("Cannot reload model, it is disabled or does not exist.")
            return
        try:
            self.model = joblib.load(self.config.model_path)
            self._log.info("HMM model reloaded successfully.")
            self._label_regimes()
        except Exception as e:
            self._log.error("Failed to reload HMM model.", error=e, exc_info=True)
            self.config.enabled = False

    def _label_regimes(self):
        if not self.model:
            return
        sorted_indices_by_vol = np.argsort(self.model.means_[:, 1])
        self.regime_names = {
            int(sorted_indices_by_vol[0]): "Neutral / Ranging",
            int(sorted_indices_by_vol[1]): "Bullish Quiet",
            int(sorted_indices_by_vol[2]): "Bearish Volatile",
        }
        self._log.info(
            "HMM regimes labeled based on volatility:", labels=self.regime_names
        )
        for state_idx in sorted_indices_by_vol:
            state_idx_int = int(state_idx)
            self._log.info(
                f"  - Regime '{self.regime_names.get(state_idx_int, 'Unlabeled')}' (State {state_idx_int})",
                mean_return=f"{self.model.means_[state_idx_int][0]:.5f}",
                mean_vol=f"{self.model.means_[state_idx_int][1]:.5f}",
            )

    async def predict_current_regime(self, recent_features: np.ndarray) -> str:
        if not self.config.enabled or self.model is None:
            return "DISABLED"
        if self.last_prediction_time and (
            datetime.now() - self.last_prediction_time
        ) < timedelta(seconds=30):
            return self.current_regime

        try:
            if recent_features.ndim == 1:
                recent_features = recent_features.reshape(1, -1)
            if recent_features.shape[1] != self.model.n_features:
                self._log.error(
                    "Feature dimension mismatch for HMM prediction.",
                    expected=self.model.n_features,
                    got=recent_features.shape[1],
                )
                return "PREDICTION_ERROR"

            hidden_states = await asyncio.to_thread(self.model.predict, recent_features)
            latest_state_idx = hidden_states[-1]
            self.current_regime = self.regime_names.get(
                int(latest_state_idx), f"UNKNOWN_{latest_state_idx}"
            )
            self.last_prediction_time = datetime.now()
            return self.current_regime
        except Exception as e:
            self._log.error("Failed to predict regime.", error=e)
            return "PREDICTION_ERROR"


# =============================================================================
# CORE SERVICES & MANAGERS
# =============================================================================

class PersistenceService:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.engine = create_async_engine(f"sqlite+aiosqlite:///{self.db_path}")
        self._log = structlog.get_logger(self.__class__.__name__)

    async def initialize(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(self._create_tables)
        self._log.info(f"Database initialized at {self.db_path}")

    def _create_tables(self, sync_conn):
        sync_conn.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS trade_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp DATETIME, symbol TEXT,
                orderId TEXT, side TEXT, price REAL, quantity REAL, commission REAL,
                pnl REAL, reason TEXT,
                market_regime TEXT, strategy_id TEXT, raw_data TEXT
            )
        """
            )
        )
        sync_conn.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS portfolio_history (
                timestamp DATETIME PRIMARY KEY, total_value REAL
            )
        """
            )
        )

    async def log_trade(self, fill_event: FillEvent, pnl=None, reason=None):
        async with self.engine.begin() as conn:
            await conn.execute(
                text(
                    """
                    INSERT INTO trade_logs (timestamp, symbol, orderId, side, price, quantity, commission, pnl, reason, market_regime, strategy_id, raw_data)
                    VALUES (:ts, :s, :i, :S, :p, :l, :n, :pnl, :reason, :regime, :strat, :raw)
                """
                ),
                {
                    "ts": datetime.now(timezone.utc),
                    "s": fill_event.symbol,
                    "i": fill_event.order_id,
                    "S": fill_event.side,
                    "p": fill_event.price,
                    "l": fill_event.quantity,
                    "n": fill_event.commission,
                    "pnl": pnl,
                    "reason": reason,
                    "regime": fill_event.market_regime,
                    "strat": fill_event.strategy_id,
                    "raw": json.dumps(fill_event.raw_data),
                },
            )
        self._log.debug("Trade log saved to DB.", symbol=fill_event.symbol, pnl=pnl)


class Notifier:
    def __init__(self, webhook_url: str, session: aiohttp.ClientSession):
        self._webhook_url = webhook_url
        self._session = session
        self._log = structlog.get_logger(self.__class__.__name__)

    async def send(
        self, title: str, msg: str, color: int = 3447003, is_error: bool = False
    ):
        if not self._webhook_url:
            return
        if self._session.closed:
            self._log.error(
                "Discord notification failed: HTTP session is closed.", title=title
            )
            return

        payload = {
            "embeds": [
                {
                    "title": title,
                    "description": msg,
                    "color": color,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            ]
        }
        try:
            await self._session.post(
                self._webhook_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)
            )
        except Exception as e:
            self._log.error("Discord notification failed", error=str(e), title=title)


class AsyncRateLimiter:
    def __init__(self, rate: int, per: float = 1.0):
        self._rate = rate
        self._per = per
        self._tokens = rate
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._rate, self._tokens + elapsed * (self._rate / self._per))
            if self._tokens < 1:
                sleep_time = (1 - self._tokens) * (self._per / self._rate)
                await asyncio.sleep(sleep_time)
                now = time.monotonic()
                elapsed = now - self._last
                self._tokens = min(self._rate, self._tokens + elapsed * (self._rate / self._per))
            self._tokens -= 1
            self._last = now


class ExchangeService:
    def __init__(self, client: AsyncClient, config: BotSettings):
        self._client = client
        self._config = config
        self._exchange_info: Dict[str, Any] = {}
        self._semaphore = asyncio.Semaphore(config.concurrent_api_calls)
        self._rate_limiter = AsyncRateLimiter(config.api_requests_per_second)
        self._order_limiter = AsyncRateLimiter(config.order_requests_per_second)
        self._log = structlog.get_logger(self.__class__.__name__)

    async def initialize(self):
        try:
            await self._execute_api_call(self._client.ping)
            info = await self._execute_api_call(self._client.get_exchange_info)
            self._exchange_info = {s["symbol"]: s for s in info["symbols"]}
            self._log.info(
                f"{Fore.GREEN}✅ Exchange service initialized",
                symbols_loaded=len(self._exchange_info),
            )
        except Exception as e:
            self._log.critical(
                "FATAL: Exchange service failed to initialize.",
                error=str(e),
                exc_info=True,
            )
            await self._client.close_connection()
            raise

    async def _execute_api_call(self, api_func: Coroutine, *args, **kwargs) -> Any:
        retries = self._config.api_max_retries
        delay = self._config.api_backoff_factor
        for i in range(retries):
            try:
                await self._rate_limiter.acquire()
                async with self._semaphore:
                    return await asyncio.wait_for(
                        api_func(*args, **kwargs),
                        timeout=self._config.api_timeout_seconds,
                    )
            except (
                BinanceAPIException,
                BinanceRequestException,
                httpx.ReadTimeout,
                aiohttp.ClientError,
                asyncio.TimeoutError,
                ConnectionError,
            ) as e:
                if isinstance(e, BinanceAPIException) and e.code in [
                    -1021,
                    -2015,
                    -1121,
                ]:
                    self._log.error(f"Non-retryable API error: {e}")
                    raise
                if i < retries - 1:
                    sleep_time = delay * (2**i) + (random.random() * 0.1)
                    self._log.warning(
                        f"API call failed (attempt {i+1}/{retries}). Retrying in {sleep_time:.2f}s.",
                        error=str(e),
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    self._log.error(
                        f"API call failed after {retries} attempts.",
                        func=api_func.__name__,
                        error=e,
                    )
                    raise
        raise ConnectionError(f"API call failed after {retries} retries.")

    def get_symbol_filter(self, symbol: str, filter_type: str) -> Optional[Dict]:
        filters = self._exchange_info.get(symbol, {}).get("filters", [])
        return next((f for f in filters if f.get("filterType") == filter_type), None)

    def get_lot_size_info(self, symbol: str) -> tuple[int, float]:
        f = self.get_symbol_filter(symbol, "LOT_SIZE")
        step = float(f["stepSize"]) if f and "stepSize" in f else 1e-8
        return (int(round(-math.log10(step))) if step > 0 else 0), step

    def get_min_notional(self, symbol: str) -> float:
        f = self.get_symbol_filter(symbol, "NOTIONAL")
        return float(f.get("minNotional", 5.0)) if f else 5.0

    async def get_top_symbols(self, quote_asset: str, top_n: int) -> List[str]:
        try:
            tickers = await self._execute_api_call(self._client.get_ticker)
            df = pl.from_records(
                tickers, schema={"symbol": pl.String, "quoteVolume": pl.String}
            )
            return (
                df.filter(
                    pl.col("symbol").str.ends_with(quote_asset)
                    & ~pl.col("symbol").str.contains("UP|DOWN|BEAR|BULL|EUR|GBP|AUD")
                )
                .with_columns(
                    pl.col("quoteVolume").cast(pl.Float64, strict=False).fill_null(0)
                )
                .filter(pl.col("quoteVolume") > 2_000_000)
                .sort("quoteVolume", descending=True)
                .head(top_n)["symbol"]
                .to_list()
            )
        except Exception as e:
            self._log.error("Error fetching top symbols", error=str(e))
            return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    async def fetch_klines(
        self, symbol: str, interval: str, limit: int
    ) -> pl.DataFrame:
        try:
            raw = await self._execute_api_call(
                self._client.get_klines, symbol=symbol, interval=interval, limit=limit
            )
            if not raw:
                return pl.DataFrame()
            df = pl.from_records(
                raw,
                schema=[
                    "open_time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_volume",
                    "n_trades",
                    "taker_buy_base_vol",
                    "taker_buy_quote_vol",
                    "ignore",
                ],
            )
            return df.select(
                pl.col("open_time")
                .cast(pl.Int64)
                .mul(1000)
                .cast(pl.Datetime(time_unit="us", time_zone="UTC")),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            )
        except Exception as e:
            self._log.error("Failed to fetch klines.", symbol=symbol, error=e)
            return pl.DataFrame()

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: Optional[float] = None,
        quote_order_qty: Optional[float] = None,
    ) -> Optional[Dict]:
        log_ctx = self._log.bind(
            symbol=symbol,
            side=side,
            requested_qty=quantity,
            requested_quote_qty=quote_order_qty,
        )
        try:
            params = {"symbol": symbol, "side": side.upper(), "type": "MARKET"}
            if quote_order_qty:
                params["quoteOrderQty"] = quote_order_qty
            elif quantity:
                precision, step_size = self.get_lot_size_info(symbol)
                fmt_qty = round(
                    math.floor(quantity / step_size) if step_size > 0 else quantity,
                    precision,
                )
                if fmt_qty <= 0:
                    log_ctx.warning("Order quantity zero after formatting.")
                    return None
                params["quantity"] = fmt_qty
            else:
                raise ValueError("Either quantity or quote_order_qty must be specified.")

            log_ctx.info(f"{Fore.YELLOW}Placing {side} market order...")
            await self._order_limiter.acquire()
            order = await self._execute_api_call(self._client.create_order, **params)
            log_ctx.info(
                f"{Fore.GREEN}Order placed successfully.", order_id=order.get("orderId")
            )
            return order
        except Exception as e:
            log_ctx.error("Order placement failed", error=e)
            raise

    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        try:
            ticker = await self._execute_api_call(
                self._client.get_symbol_ticker, symbol=symbol
            )
            return float(ticker["price"])
        except Exception as e:
            self._log.error("Failed to get ticker price", symbol=symbol, error=e)
            return None


class DataHandler:
    def __init__(self, client: AsyncClient, config: BotSettings):
        self.client = client
        self.symbols: List[str] = []
        self.config = config.trading
        self.bm = BinanceSocketManager(self.client)
        self._log = structlog.get_logger(self.__class__.__name__)
        self.main_socket_task: Optional[asyncio.Task] = None

    async def _process_market_message(self, msg):
        if msg.get("e") == "kline" and msg["k"]["x"]:
            kline = msg["k"]
            df = pl.DataFrame(
                {
                    "open_time": [
                        datetime.fromtimestamp(kline["t"] / 1000, tz=timezone.utc)
                    ],
                    "open": [float(kline["o"])],
                    "high": [float(kline["h"])],
                    "low": [float(kline["l"])],
                    "close": [float(kline["c"])],
                    "volume": [float(kline["v"])],
                },
                schema={
                    "open_time": pl.Datetime(time_unit="us", time_zone="UTC"),
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                },
            )
            await EVENT_QUEUE.put(MarketEvent(symbol=msg["s"], data=df))

    async def _process_user_message(self, msg):
        if msg.get("e") == "executionReport" and msg.get("X") in ["TRADE", "FILLED"]:
            fill = FillEvent(
                symbol=msg["s"],
                side=msg["S"],
                quantity=float(msg["l"]),
                price=float(msg["L"]),
                commission=float(msg.get("n", 0.0)),
                order_id=str(msg["i"]),
                cummulative_quote_qty=float(msg.get("Z", 0.0)),
                raw_data=msg,
            )
            await EVENT_QUEUE.put(fill)

    def start_streams(self, symbols: Optional[List[str]] = None):
        if symbols is not None:
            self.symbols = symbols
        if not self.symbols:
            self._log.warning("No symbols provided to start streams.")
            return

        kline_streams = [
            f"{symbol.lower()}@kline_{self.config.interval}" for symbol in self.symbols
        ]

        async def market_loop():
            self._log.info(
                f"Starting market data stream for {len(kline_streams)} symbols."
            )
            market_socket = self.bm.multiplex_socket(kline_streams)
            async with market_socket as stream:
                while not SHUTDOWN_EVENT.is_set():
                    try:
                        res = await stream.recv()
                        if "data" in res and res.get("e") != "error":
                            await self._process_market_message(res["data"])
                    except Exception as e:
                        if not SHUTDOWN_EVENT.is_set():
                            self._log.error("Error in market stream.", error=e)
                            await asyncio.sleep(5)

        async def user_loop():
            self._log.info("Starting user data stream...")
            user_socket = self.bm.user_socket()
            async with user_socket as stream:
                while not SHUTDOWN_EVENT.is_set():
                    try:
                        res = await stream.recv()
                        await self._process_user_message(res)
                    except Exception as e:
                        if not SHUTDOWN_EVENT.is_set():
                            self._log.error("Error in user stream.", error=e)
                            await asyncio.sleep(5)

        async def combined_stream_loop():
            await asyncio.gather(market_loop(), user_loop())

        self.main_socket_task = asyncio.create_task(combined_stream_loop())

    async def refresh_streams(self, symbols: List[str]):
        if set(symbols) == set(self.symbols):
            return
        self._log.info(f"Refreshing data streams with {len(symbols)} symbols.")
        if self.main_socket_task:
            self.main_socket_task.cancel()
            try:
                await self.main_socket_task
            except asyncio.CancelledError:
                pass
        self.start_streams(symbols)


class AIAnalysisManager:
    def __init__(self, config: AISettings, api_settings: ApiSettings):
        self.config = config
        if not api_settings.openai_api_key:
            log.warning("OpenAI API key not configured. AI Decider disabled.")
            self.llm_client = None
            self.config.enable_ai_decider = False
        else:
            self.llm_client = AsyncOpenAI(api_key=api_settings.openai_api_key)
        self.decision_cache: OrderedDict[str, dict] = OrderedDict()
        self._cache_max = 128
        self._log = structlog.get_logger(self.__class__.__name__)

    async def get_ai_trade_decision(self, signal_event: SignalEvent) -> dict:
        if not self.config.enable_ai_decider or not self.llm_client:
            return {
                "decision": "PROCEED",
                "confidence_score": 1.0,
                "reasoning": "AI Decider disabled.",
            }

        signal = signal_event.details
        prompt_key = hashlib.sha256(
            json.dumps(
                {
                    "symbol": signal_event.symbol,
                    "action": signal_event.action,
                    "price": round(signal_event.price, 4),
                    "strategies": signal.get("strategy_names", []),
                    "regime": signal.get("market_regime", "UNKNOWN"),
                    "indicators": signal.get("indicators", {}),
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

        if prompt_key in self.decision_cache:
            self._log.debug(
                f"Using cached AI decision for {signal_event.symbol} {signal_event.action}"
            )
            self.decision_cache.move_to_end(prompt_key)
            return self.decision_cache[prompt_key]
        system_prompt = (
            "You are 'MarketMind', a quantitative crypto trading analyst. Evaluate a trade signal\n"
            "by synthesizing all available data. Your output must be a single, valid JSON object.\n"
            "Respond ONLY with a valid JSON object:\n{\n  \"decision\": \"PROCEED or VETO\",\n  \"confidence_score\": float,\n  \"reasoning\": \"Concise explanation, max 50 words.\"\n}"
        )

        user_prompt = (
            f"Analyze the trade signal for {signal_event.symbol}:\n"
            f"Action: {signal_event.action}\n"
            f"Price: {signal_event.price:.4f}\n"
            f"Strategies: {', '.join(signal.get('strategy_names', []))}\n"
            f"Regime: {signal.get('market_regime', 'UNKNOWN')}\n"
            f"Indicators: {json.dumps(signal.get('indicators', {}), separators=(',', ':'))}"
            "\nProvide your analysis in JSON."
        )
        try:
            self._log.info(
                f"Requesting AI decision for {signal_event.symbol} {signal_event.action} signal."
            )
            response = await self.llm_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=200,
            )
            ai_decision = json.loads(response.choices[0].message.content)
            decision_color = (
                Fore.GREEN if ai_decision.get("decision") == "PROCEED" else Fore.RED
            )
            self._log.info(
                f"{decision_color}AI DECISION: {ai_decision.get('decision')}",
                symbol=signal_event.symbol,
                confidence=f"{ai_decision.get('confidence_score', 0.0):.2f}",
                reason=ai_decision.get("reasoning"),
            )
            self.decision_cache[prompt_key] = ai_decision
            if len(self.decision_cache) > self._cache_max:
                self.decision_cache.popitem(last=False)
            return ai_decision
        except Exception as e:
            self._log.error(
                f"Error during AI decision analysis for {signal_event.symbol}: {e}"
            )
            return {
                "decision": "VETO",
                "confidence_score": 0.0,
                "reasoning": "AI analysis failed.",
            }

# =============================================================================
# STRATEGY & PORTFOLIO LOGIC
# =============================================================================

class BaseStrategy:
    def __init__(self, name: StrategyType, config: StrategySettings, log: structlog.BoundLogger):
        self.name = name
        self.config = config
        self.log = log

    def generate_signal(self, data: pl.DataFrame) -> int:
        raise NotImplementedError


class RSIReversalStrategy(BaseStrategy):
    def __init__(self, config: StrategySettings, log: structlog.BoundLogger):
        super().__init__(StrategyType.MEAN_REVERSION_RSI, config, log)
        s = self.config
        self.period, self.oversold, self.overbought = s.rsi_period, s.rsi_oversold, s.rsi_overbought

    def generate_signal(self, data: pl.DataFrame) -> int:
        if data.height < 2: return 0
        last, prev = data.row(-1, named=True), data.row(-2, named=True)
        rsi_col = f"rsi_{self.period}"
        if last.get(rsi_col, 50) > self.oversold and prev.get(rsi_col, 50) <= self.oversold: return 1
        if last.get(rsi_col, 50) < self.overbought and prev.get(rsi_col, 50) >= self.overbought: return -1
        return 0


class MACDCrossoverStrategy(BaseStrategy):
    def __init__(self, config: StrategySettings, log: structlog.BoundLogger):
        super().__init__(StrategyType.TREND_MACD, config, log)
        s = self.config
        self.fast, self.slow, self.signal = s.macd_fast, s.macd_slow, s.macd_signal

    def generate_signal(self, data: pl.DataFrame) -> int:
        if data.height < 2: return 0
        last, prev = data.row(-1, named=True), data.row(-2, named=True)
        if last.get("macd", 0) > last.get("macdsignal", 0) and prev.get("macd", 0) <= prev.get("macdsignal", 0): return 1
        if last.get("macd", 0) < last.get("macdsignal", 0) and prev.get("macd", 0) >= prev.get("macdsignal", 0): return -1
        return 0


class SMACrossoverStrategy(BaseStrategy):
    def __init__(self, config: StrategySettings, log: structlog.BoundLogger):
        super().__init__(StrategyType.SMA_CROSSOVER, config, log)
        s = self.config
        self.fast, self.slow = s.sma_fast, s.sma_slow

    def generate_signal(self, data: pl.DataFrame) -> int:
        if data.height < 2: return 0
        last, prev = data.row(-1, named=True), data.row(-2, named=True)
        fast_col, slow_col = f"sma_{self.fast}", f"sma_{self.slow}"
        if last.get(fast_col, 0) > last.get(slow_col, 0) and prev.get(fast_col, 0) <= prev.get(slow_col, 0): return 1
        if last.get(fast_col, 0) < last.get(slow_col, 0) and prev.get(fast_col, 0) >= prev.get(slow_col, 0): return -1
        return 0


class IchimokuStrategy(BaseStrategy):
    def __init__(self, config: StrategySettings, log: structlog.BoundLogger):
        super().__init__(StrategyType.ICHIMOKU_BREAKOUT, config, log)
        s = self.config
        self.t, self.k, self.s = s.ichimoku_tenkan, s.ichimoku_kijun, s.ichimoku_senkou_b

    def generate_signal(self, data: pl.DataFrame) -> int:
        if data.height < 2: return 0
        last, prev = data.row(-1, named=True), data.row(-2, named=True)
        close = last.get("close")
        if not close: return 0
        tenkan_col, kijun_col = f"tenkan_sen_{self.t}", f"kijun_sen_{self.k}"
        senkou_a_col = f"senkou_a_{self.t}_{self.k}_{self.s}"
        senkou_b_col = f"senkou_b_{self.k}_{self.s}"
        is_above_cloud = close > last.get(senkou_a_col, 0) and close > last.get(senkou_b_col, 0)
        tk_cross = last.get(tenkan_col, 0) > last.get(kijun_col, 0) and prev.get(tenkan_col, 0) <= prev.get(kijun_col, 0)
        if is_above_cloud and tk_cross: return 1
        is_below_cloud = close < last.get(senkou_a_col, 0) and close < last.get(senkou_b_col, 0)
        kt_cross = last.get(tenkan_col, 0) < last.get(kijun_col, 0) and prev.get(tenkan_col, 0) >= prev.get(kijun_col, 0)
        if is_below_cloud and kt_cross: return -1
        return 0

class CombinedStrategy(BaseStrategy):
    def __init__(self, config: StrategySettings, log: structlog.BoundLogger):
        super().__init__(StrategyType.COMBINED, config, log)

    def generate_signal(self, data: pl.DataFrame) -> int:
        self.log.debug(f"[{self.name.value}] Generating signal...")
        if data.height < 2: return 0

        latest = data.row(-1, named=True)
        previous = data.row(-2, named=True)

        is_trend_up = latest['close'] > latest['trend_ema']
        if not is_trend_up:
            self.log.debug(f"[{self.name.value}] No signal: Trend is down.")
            return 0
        
        reasons = []
        if previous['close'] <= previous['ema_slow'] and latest['close'] > latest['ema_slow']: reasons.append("Price>EMA_Slow")
        if previous['ema_fast'] <= previous['ema_slow'] and latest['ema_fast'] > latest['ema_slow']: reasons.append("EMA_Crossover")
        if previous['rsi'] <= self.config.rsi_entry_midline < latest['rsi']: reasons.append("RSI_Cross")
        if previous['macd'] <= previous['macdsignal'] and latest['macd'] > latest['macdsignal']: reasons.append("MACD_Crossover")
        if latest['close'] > previous['recent_high'] * (1 + self.config.breakout_margin_pct): reasons.append("Breakout")

        if reasons:
            self.log.info(f"[{self.name.value}] BUY signal generated.", reasons=reasons)
            return 1
        
        self.log.debug(f"[{self.name.value}] No entry signals found, despite uptrend.")
        return 0


class StrategyHandler:
    def __init__(
        self,
        exchange: ExchangeService,
        config: StrategySettings,
        regime_detector: MarketRegimeDetector,
    ):
        self.exchange = exchange
        self.config = config
        self.regime_detector = regime_detector
        self.historical_data: Dict[str, pl.DataFrame] = {}
        self.indicator_cache: Dict[str, pl.DataFrame] = {}
        self._log = structlog.get_logger(self.__class__.__name__)
        self.strategies: List[BaseStrategy] = self._initialize_strategies()


    def _initialize_strategies(self):
        s_log = self._log.bind(component="Strategy")
        return [
            RSIReversalStrategy(self.config, s_log),
            MACDCrossoverStrategy(self.config, s_log),
            SMACrossoverStrategy(self.config, s_log),
            IchimokuStrategy(self.config, s_log),
            CombinedStrategy(self.config, s_log),
        ]

    async def initialize_data(self, symbols: List[str], interval: str):
        self._log.info(
            f"Fetching initial historical data for {len(symbols)} symbols..."
        )
        tasks = [
            self.exchange.fetch_klines(s, interval, self.config.data_fetch_limit)
            for s in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for symbol, df in zip(symbols, results):
            if isinstance(df, pl.DataFrame) and not df.is_empty():
                self.historical_data[symbol] = df
                self.indicator_cache[symbol] = self._calculate_all_indicators(df, symbol)
            elif isinstance(df, Exception):
                self._log.error(f"Could not fetch initial data for {symbol}", error=df)
        self._log.info("StrategyHandler is ready with initial data.")

    def _calculate_all_indicators(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        s = self.config
        try:
            if df.height < s.ichimoku_senkou_b or df.height < s.trend_ma_period:
                return pl.DataFrame()

            lf = df.lazy()
            # Standard Indicators
            lf = lf.with_columns(
                plta.rsi(pl.col("close"), timeperiod=s.rsi_period).alias("rsi"),
                plta.macd(
                    pl.col("close"), fastperiod=s.macd_fast, slowperiod=s.macd_slow, signalperiod=s.macd_signal
                ).struct.unnest(),
                plta.sma(pl.col("close"), timeperiod=s.sma_fast).alias(f"sma_{s.sma_fast}"),
                plta.sma(pl.col("close"), timeperiod=s.sma_slow).alias(f"sma_{s.sma_slow}"),
            )
            # Indicators from CombinedStrategy
            lf = lf.with_columns(
                pl.col("close").ewm_mean(span=s.trend_ma_period, adjust=False).alias("trend_ema"),
                pl.col("close").ewm_mean(span=s.macd_fast, adjust=False).alias("ema_fast"),
                pl.col("close").ewm_mean(span=s.macd_slow, adjust=False).alias("ema_slow"),
                pl.col("high").rolling_max(window_size=s.volatility_window).alias("recent_high")
            )
            # Ichimoku
            tenkan_high = pl.col("high").rolling_max(s.ichimoku_tenkan)
            tenkan_low = pl.col("low").rolling_min(s.ichimoku_tenkan)
            lf = lf.with_columns(((tenkan_high + tenkan_low) / 2).alias(f"tenkan_sen_{s.ichimoku_tenkan}"))
            kijun_high = pl.col("high").rolling_max(s.ichimoku_kijun)
            kijun_low = pl.col("low").rolling_min(s.ichimoku_kijun)
            lf = lf.with_columns(((kijun_high + kijun_low) / 2).alias(f"kijun_sen_{s.ichimoku_kijun}"))
            senkou_a = ((pl.col(f"tenkan_sen_{s.ichimoku_tenkan}") + pl.col(f"kijun_sen_{s.ichimoku_kijun}"))/ 2).shift(-s.ichimoku_kijun)
            lf = lf.with_columns(senkou_a.alias(f"senkou_a_{s.ichimoku_tenkan}_{s.ichimoku_kijun}_{s.ichimoku_senkou_b}"))
            senkou_b_high = pl.col("high").rolling_max(s.ichimoku_senkou_b)
            senkou_b_low = pl.col("low").rolling_min(s.ichimoku_senkou_b)
            senkou_b = (senkou_b_high + senkou_b_low) / 2
            senkou_b = senkou_b.shift(-s.ichimoku_kijun)
            lf = lf.with_columns(senkou_b.alias(f"senkou_b_{s.ichimoku_kijun}_{s.ichimoku_senkou_b}"))
            
            return lf.drop_nulls().collect()
        except Exception as e:
            self._log.error("Indicator calculation failed", symbol=symbol, error=str(e), exc_info=False)
            return pl.DataFrame()

    def _update_indicator_cache(self, symbol: str) -> pl.DataFrame:
        df = self.historical_data.get(symbol)
        if df is None or df.is_empty(): return pl.DataFrame()

        cached = self.indicator_cache.get(symbol)
        last_time = cached["open_time"].max() if cached is not None and not cached.is_empty() else None
        
        new_df = df.filter(pl.col("open_time") > last_time) if last_time else df
        if new_df.is_empty(): return cached if cached is not None else pl.DataFrame()
        
        new_ind = self._calculate_all_indicators(new_df, symbol)
        if new_ind.is_empty() or new_ind.width == 0: return cached if cached is not None else pl.DataFrame()

        result = pl.concat([cached, new_ind]).unique(subset=["open_time"], keep="last").tail(1000) if cached is not None and not cached.is_empty() else new_ind
        self.indicator_cache[symbol] = result
        return result

    async def on_market_event(self, event: MarketEvent):
        symbol = event.symbol
        if symbol not in self.historical_data:
            self.historical_data[symbol] = await self.exchange.fetch_klines(
                symbol, self.config.interval, self.config.data_fetch_limit
            )
        else:
            self.historical_data[symbol] = (
                pl.concat([self.historical_data[symbol], event.data])
                .unique(subset=["open_time"], keep="last")
                .tail(1000)
            )

        if self.historical_data[symbol].height < self.config.min_candles_for_analysis: return

        data_with_ta = self._update_indicator_cache(symbol)
        if data_with_ta.is_empty(): return

        buy_voting_strategies, sell_voting_strategies = [], []
        for strategy in self.strategies:
            signal = strategy.generate_signal(data_with_ta)
            if signal == 1:
                buy_voting_strategies.append(strategy.name.value)
            elif signal == -1:
                sell_voting_strategies.append(strategy.name.value)

        final_action, strategies_for_signal = None, []
        if len(buy_voting_strategies) >= self.config.signal_consensus_threshold:
            final_action, strategies_for_signal = "BUY", buy_voting_strategies
        elif len(sell_voting_strategies) >= self.config.signal_consensus_threshold:
            final_action, strategies_for_signal = "SELL", sell_voting_strategies

        if final_action:
            regime_ticker = self.regime_detector.config.training_ticker
            market_data_for_regime = self.historical_data.get(regime_ticker)
            current_regime = "UNKNOWN"
            if market_data_for_regime is not None and not market_data_for_regime.is_empty():
                try:
                    returns = (market_data_for_regime["close"] / market_data_for_regime["close"].shift(1)) - 1
                    log_returns = returns.log1p()
                    volatility = log_returns.rolling_std(window_size=21)
                    features_df = pl.DataFrame({"log_returns": log_returns, "volatility": volatility}).drop_nulls()
                    if features_df.height > 30:
                        features = features_df.tail(30).to_numpy()
                        current_regime = await self.regime_detector.predict_current_regime(features)
                except Exception as e:
                    self._log.warning("Could not calculate features for regime prediction", error=e)
            else:
                self._log.warning(f"No historical data for regime ticker {regime_ticker}.")
            
            signal_details = {
                "strategy_names": strategies_for_signal, "market_regime": current_regime,
                "indicators": {
                    col: round(val, 5) for col, val in data_with_ta.row(-1, named=True).items()
                    if col not in ["open_time", "open", "high", "low", "close", "volume"] and val is not None
                },
            }
            self._log.info(f"{Fore.CYAN}🚀 CONSENSUS SIGNAL: {final_action}", symbol=symbol, strategies=strategies_for_signal, regime=current_regime)
            await EVENT_QUEUE.put(SignalEvent(symbol, final_action, data_with_ta["close"].last(), signal_details))

class PortfolioManager:
    def __init__(
        self,
        persistence: PersistenceService,
        exchange: ExchangeService,
        notifier: Notifier,
        ai_manager: AIAnalysisManager,
        config: BotSettings,
        strategy_handler: StrategyHandler,
    ):
        self.persistence, self.exchange, self.notifier, self.ai_manager, self.config = (
            persistence, exchange, notifier, ai_manager, config
        )
        self.strategy_handler = strategy_handler
        self.cash = config.trading.initial_cash_balance
        self.positions: Dict[str, Position] = {}
        self.symbol_states: Dict[str, BotState] = {}
        self.open_trade_details: Dict[str, Dict] = {}
        self.current_risk_pct = config.risk.initial_risk_per_trade_pct
        self.wins, self.losses = 0, 0
        self.peak_equity = config.trading.initial_cash_balance
        self.is_liquidating = False
        self._log = structlog.get_logger(self.__class__.__name__)

    async def on_signal_event(self, event: SignalEvent):
        symbol = event.symbol
        current_regime = event.details.get("market_regime", "UNKNOWN")
        if current_regime in self.config.regime.high_risk_regimes:
            self._log.warning("VETOING TRADE due to high-risk market regime.", symbol=symbol, regime=current_regime)
            return

        if len(self.positions) >= self.config.trading.max_active_positions and event.action == "BUY" and symbol not in self.positions:
            self._log.warning("VETOING TRADE due to max active positions limit.", symbol=symbol, limit=self.config.trading.max_active_positions)
            return
        
        if event.action == "BUY" and symbol not in self.positions:
            await self.handle_buy_signal(event)
        elif event.action == "SELL" and symbol in self.positions:
            await self.create_exit_order(symbol, reason=f"Signal from {','.join(event.details.get('strategy_names', ['Unknown']))}")

    async def handle_buy_signal(self, event: SignalEvent):
        ai_decision = await self.ai_manager.get_ai_trade_decision(event)
        if ai_decision["decision"] == "VETO" or ai_decision["confidence_score"] < self.config.ai.min_ai_confidence_score:
            self._log.warning("AI VETO", symbol=event.symbol, confidence=f"{ai_decision['confidence_score']:.2f}", reason=ai_decision["reasoning"])
            return
        
        trade_amount = self.config.trading.initial_trade_amount
        if self.cash < trade_amount:
            self._log.warning("Insufficient balance for new trade.", symbol=event.symbol, required=trade_amount, available=self.cash)
            return

        self.symbol_states[event.symbol] = BotState.ORDER_PENDING
        self.open_trade_details[event.symbol] = {"signal_event": event}
        await EVENT_QUEUE.put(OrderEvent(symbol=event.symbol, side=SIDE_BUY, quantity=None, quote_order_qty=trade_amount))

    async def on_fill_event(self, event: FillEvent):
        symbol, side, price, quantity = event.symbol, event.side, event.price, event.quantity
        
        if side == "BUY":
            signal_event = self.open_trade_details.get(symbol, {}).get("signal_event")
            if not signal_event: self._log.error("Could not find originating signal for fill.", symbol=symbol); return
            
            spent = event.cummulative_quote_qty
            strategies = signal_event.details.get("strategy_names", [])
            market_regime = signal_event.details.get("market_regime", "UNKNOWN")
            
            if symbol in self.positions: # This is a DCA buy
                pos = self.positions[symbol]
                new_total_qty = pos.quantity + quantity
                new_total_cost = (pos.avg_price * pos.quantity) + spent
                pos.avg_price = new_total_cost / new_total_qty
                pos.quantity = new_total_qty
                pos.dca_count += 1
                self.cash -= spent
                self._log.info(f"DCA successful for {symbol}. New avg price: {pos.avg_price:.4f}, New qty: {pos.quantity:.6f}")
            else: # This is a new position
                self.positions[symbol] = Position(
                    symbol=symbol, quantity=quantity, entry_price=price, avg_price=price, side="LONG",
                    entry_time=datetime.now(timezone.utc), entry_strategy=strategies,
                    stop_loss_price=price * (1 - self.config.risk.stop_loss_pct),
                    take_profit_price=price * (1 + self.config.risk.take_profit_pct),
                    trailing_stop_price=None, market_regime_at_entry=market_regime,
                )
                self.cash -= spent
                self._log.info(f"{Fore.GREEN}FILLED BUY", symbol=symbol, price=f"{price:.4f}", regime=market_regime)
                await self.notifier.send(f"📈 LONG Entry ({','.join(strategies)})", f"**Symbol:** {symbol}\n**Entry:** ${price:,.4f}", color=3066993)

        elif side == "SELL" and symbol in self.positions:
            pos = self.positions.pop(symbol)
            received = event.cummulative_quote_qty
            cost = pos.avg_price * pos.quantity
            pnl = received - cost
            self.cash += received
            reason = self.open_trade_details.pop(symbol, {}).get("reason_for_close", "N/A")

            event.market_regime = pos.market_regime_at_entry
            event.strategy_id = ",".join(pos.entry_strategy)
            await self.persistence.log_trade(event, pnl=pnl, reason=reason)

            pnl_color = Fore.GREEN if pnl >= 0 else Fore.RED
            self._log.info(f"{pnl_color}FILLED SELL (CLOSE)", symbol=symbol, pnl_usd=f"${pnl:.2f}", reason=reason)
            await self.notifier.send(f"💰 Trade Closed: {symbol}", f"**P/L:** `${pnl:,.2f}`\n**Reason:** {reason}", color=3447003 if pnl >= 0 else 15158332)
            self._update_dynamic_risk(pnl)

        self.symbol_states[symbol] = BotState.AWAITING_SIGNAL
        await self.update_and_log_portfolio_status()

    async def manage_open_positions(self):
        if not self.positions: return
        for symbol in list(self.positions.keys()):
            pos = self.positions.get(symbol)
            if not pos: continue
            
            current_price = await self.exchange.get_ticker_price(symbol)
            if not current_price: self._log.error(f"Could not get price for {symbol} to manage position."); continue

            if exit_reason := self._check_exit_conditions(pos, current_price):
                await self.create_exit_order(symbol, reason=exit_reason)
            elif self._check_dca_condition(pos, current_price):
                await self._execute_dca_buy(symbol)

    def _check_exit_conditions(self, pos: Position, current_price: float) -> Optional[str]:
        if current_price >= pos.take_profit_price: return f"Take-Profit hit at {pos.take_profit_price:.4f}"
        if current_price <= pos.stop_loss_price: return f"Stop-Loss hit at {pos.stop_loss_price:.4f}"
        return None

    def _check_dca_condition(self, pos: Position, current_price: float) -> bool:
        return (
            pos.dca_count < self.config.dca.max_dca_orders and
            current_price <= pos.avg_price * (1 - self.config.dca.dca_buy_threshold_pct)
        )

    async def _execute_dca_buy(self, symbol: str):
        pos = self.positions[symbol]
        dca_amount = self.config.trading.initial_trade_amount * (self.config.dca.dca_volume_scale ** (pos.dca_count + 1))
        if self.cash < dca_amount:
            self._log.warning(f"Insufficient balance for DCA on {symbol}.")
            return

        self._log.info(f"DCA condition met for {symbol}. Executing DCA buy.")
        await EVENT_QUEUE.put(OrderEvent(symbol=symbol, side=SIDE_BUY, quantity=None, quote_order_qty=dca_amount))

    async def create_exit_order(self, symbol: str, reason: str):
        if symbol not in self.positions or self.symbol_states.get(symbol) == BotState.EXIT_PENDING: return
        self._log.info(f"Creating exit order for {symbol}", reason=reason)
        self.symbol_states[symbol] = BotState.EXIT_PENDING
        self.open_trade_details.setdefault(symbol, {})["reason_for_close"] = reason
        await EVENT_QUEUE.put(OrderEvent(symbol=symbol, side=SIDE_SELL, quantity=self.positions[symbol].quantity))

    async def update_and_log_portfolio_status(self, check_drawdown=False):
        position_value = 0.0
        for symbol, pos in self.positions.items():
            price = await self.exchange.get_ticker_price(symbol)
            position_value += pos.quantity * (price or pos.avg_price)
        
        total_equity = self.cash + position_value
        self.peak_equity = max(self.peak_equity, total_equity)
        drawdown = (self.peak_equity - total_equity) / self.peak_equity if self.peak_equity > 0 else 0
        
        self._log.info("PORTFOLIO STATUS", total_value=f"${total_equity:.2f}", cash=f"${self.cash:.2f}", positions=len(self.positions), peak_equity=f"${self.peak_equity:.2f}", drawdown=f"{drawdown*100:.2f}%")

        if check_drawdown and drawdown > self.config.risk.max_drawdown_pct and not self.is_liquidating:
            self.is_liquidating = True
            self._log.critical(f"MAX DRAWDOWN EXCEEDED ({drawdown*100:.2f}%). LIQUIDATING ALL POSITIONS.")
            for symbol in list(self.positions.keys()):
                await self.create_exit_order(symbol, "MAX DRAWDOWN LIQUIDATION")
            self._log.critical("Bot has been halted due to max drawdown. Please review and restart manually.")
            SHUTDOWN_EVENT.set()

    def _update_dynamic_risk(self, pnl: float):
        if not self.config.risk.dynamic_risk: return
        if pnl > 0: self.wins += 1
        else: self.losses += 1
        total = self.wins + self.losses
        if total < 5: return
        
        win_rate = self.wins / total
        old_risk = self.current_risk_pct
        if win_rate > 0.6: self.current_risk_pct = min(self.current_risk_pct * 1.1, self.config.risk.max_risk_per_trade_pct)
        elif win_rate < 0.4: self.current_risk_pct = max(self.current_risk_pct * 0.9, self.config.risk.min_risk_per_trade_pct)
        if old_risk != self.current_risk_pct:
            self._log.info("Risk level adjusted", old=f"{old_risk:.4f}", new=f"{self.current_risk_pct:.4f}", win_rate=f"{win_rate:.2f}")


EVENT_QUEUE: asyncio.Queue = asyncio.Queue()
SHUTDOWN_EVENT = asyncio.Event()


class ExecutionHandler:
    def __init__(self, exchange: ExchangeService, portfolio_manager: PortfolioManager, config: TradingSettings):
        self.exchange, self.portfolio_manager, self.config = exchange, portfolio_manager, config
        self._log = structlog.get_logger(self.__class__.__name__)

    async def on_order_event(self, event: OrderEvent):
        if self.config.trading_mode == "PAPER":
            await self._execute_paper_order(event)
        else:
            await self._execute_live_order(event)

    async def _execute_paper_order(self, event: OrderEvent):
        self._log.info(f"PAPER EXECUTION: {event.side} {event.quantity or event.quote_order_qty} {event.symbol}")
        price = await self.exchange.get_ticker_price(event.symbol)
        if not price: self._log.error("Cannot execute paper order, no last price available.", symbol=event.symbol); return
        
        quantity = event.quantity or (event.quote_order_qty / price)
        cummulative_quote_qty = quantity * price
        commission = cummulative_quote_qty * 0.001
        
        fake_trade_data = {"e": "executionReport", "X": "FILLED", "s": event.symbol, "i": f"paper_{int(time.time())}", "S": event.side.upper(), "L": str(price), "l": str(quantity), "n": str(commission), "Z": str(cummulative_quote_qty)}
        fill = FillEvent(symbol=event.symbol, side=event.side.upper(), quantity=quantity, price=price, commission=commission, order_id=fake_trade_data["i"], cummulative_quote_qty=cummulative_quote_qty, raw_data=fake_trade_data)
        await EVENT_QUEUE.put(fill)

    async def _execute_live_order(self, event: OrderEvent):
        try:
            await self.exchange.place_market_order(event.symbol, event.side, event.quantity, event.quote_order_qty)
        except Exception as e:
            self._log.error("Live execution failed", symbol=event.symbol, error=e)


async def _aiohttp_session_provider() -> AsyncGenerator[aiohttp.ClientSession, None]:
    async with aiohttp.ClientSession() as session: yield session


async def _binance_client_provider(config: BotSettings) -> AsyncGenerator[AsyncClient, None]:
    log = structlog.get_logger("BinanceClientProvider")
    try: client = await AsyncClient.create(config.api.binance_key, config.api.binance_secret)
    except Exception as e: log.critical("Failed to initialize Binance client", error=str(e)); raise
    try: yield client
    finally: await client.close_connection()


class AppContainer(containers.DeclarativeContainer):
    config_manager = providers.Singleton(ConfigManager, Path("config.json"))
    config = providers.Singleton(lambda cm: cm.config, config_manager)
    aiohttp_session = providers.Resource(_aiohttp_session_provider)
    binance_client = providers.Resource(_binance_client_provider, config=config)
    exchange = providers.Singleton(ExchangeService, client=binance_client, config=config)
    persistence = providers.Singleton(PersistenceService, db_path=config.provided.db_path)
    notifier = providers.Singleton(Notifier, webhook_url=config.provided.api.discord_webhook_url, session=aiohttp_session)
    regime_detector = providers.Singleton(MarketRegimeDetector, config=config.provided.regime, exchange_service=exchange)
    ai_manager = providers.Singleton(AIAnalysisManager, config=config.provided.ai, api_settings=config.provided.api)
    strategy_handler = providers.Singleton(StrategyHandler, exchange=exchange, config=config.provided.strategy, regime_detector=regime_detector)
    portfolio_manager = providers.Singleton(PortfolioManager, persistence=persistence, exchange=exchange, notifier=notifier, ai_manager=ai_manager, config=config, strategy_handler=strategy_handler)
    execution_handler = providers.Factory(ExecutionHandler, exchange=exchange, portfolio_manager=portfolio_manager, config=config.provided.trading)
    data_handler = providers.Singleton(DataHandler, client=binance_client, config=config)


def print_banner(settings: BotSettings):
    banner = f"""{Fore.CYAN}
    ██████╗ ██████╗  ██████╗  █████╗ ████████╗   ██████╗  ██████╗ ████████╗
    ██╔══██╗██╔══██╗██╔═══██╗██╔══██╗╚══██╔══╝   ██╔══██╗██╔═══██╗╚══██╔══╝
    ██████╔╝██████╔╝██║  ██║███████║   ██║      ██████╔╝██║  ██║   ██║
    ██╔══██╗██╔══██╗██║  ██║██╔══██║   ██║      ██╔══██╗██║  ██║   ██║
    ██████╔╝██║  ██║╚██████╔╝██║  ██║   ██║      ██████╔╝╚██████╔╝   ██║
    ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝      ╚═════╝  ╚═════╝    ╚═╝
    {Style.BRIGHT}Regime-Aware Trading Bot v{settings.version}{Style.RESET_ALL}"""
    print(banner)
    log.info(f"Version {settings.version} Starting...")
    log.info(f"MODE={settings.trading.trading_mode}, QUOTE_ASSET={settings.trading.quote_asset}")


async def periodic_hmm_retraining(regime_detector: MarketRegimeDetector, notifier: Notifier):
    log.info("HMM retraining triggered by schedule.")
    await notifier.send("🧠 HMM Retraining Started", "Periodic retraining of the market regime model has begun.", color=16776960)
    try:
        await regime_detector._train_model()
        regime_detector.reload_model()
        log.info(f"{Fore.GREEN}✅ Periodic HMM retraining completed and model reloaded.")
        await notifier.send("✅ HMM Retraining Complete", "The market regime model has been successfully updated.", color=3066993)
    except Exception as e:
        log.error("Periodic HMM retraining failed.", error=e, exc_info=False)
        await notifier.send("💥 HMM Retraining Failed", f"An error occurred during model retraining: {e}", color=15158332, is_error=True)


async def refresh_symbol_list(exchange: ExchangeService, data_handler: DataHandler, config: BotSettings) -> None:
    try:
        new_symbols = await exchange.get_top_symbols(config.trading.quote_asset, config.trading.top_n_symbols)
        regime_ticker = config.regime.training_ticker
        if regime_ticker not in new_symbols: new_symbols.append(regime_ticker)
        await data_handler.refresh_streams(new_symbols)
    except Exception as e: log.error("Failed to refresh symbol list", error=str(e))


async def main_event_loop(p_manager: PortfolioManager, e_handler: ExecutionHandler, s_handler: StrategyHandler):
    log.info("Event processing loop started.")
    while not SHUTDOWN_EVENT.is_set():
        try:
            event = await asyncio.wait_for(EVENT_QUEUE.get(), timeout=1.0)
            if isinstance(event, MarketEvent): asyncio.create_task(s_handler.on_market_event(event))
            elif isinstance(event, SignalEvent): asyncio.create_task(p_manager.on_signal_event(event))
            elif isinstance(event, OrderEvent): asyncio.create_task(e_handler.on_order_event(event))
            elif isinstance(event, FillEvent): asyncio.create_task(p_manager.on_fill_event(event))
            EVENT_QUEUE.task_done()
        except asyncio.TimeoutError: continue
        except asyncio.CancelledError: break
    log.info("Event processing loop stopped.")


async def periodic_task_loop(task_func, interval: int, task_name: str, *args, **kwargs):
    log.info(f"{task_name} loop started with interval {interval}s.")
    while not SHUTDOWN_EVENT.is_set():
        start_time = time.monotonic()
        try: await task_func(*args, **kwargs)
        except asyncio.CancelledError: break
        except Exception as e: log.error(f"Error in {task_name} loop", error=str(e), exc_info=True)
        elapsed = time.monotonic() - start_time
        sleep_duration = max(0, interval - elapsed)
        try: await asyncio.sleep(sleep_duration)
        except asyncio.CancelledError: break
    log.info(f"{task_name} loop stopped.")


@inject
async def start_bot(
    config: BotSettings = Provide[AppContainer.config],
    config_manager: ConfigManager = Provide[AppContainer.config_manager],
    exchange: ExchangeService = Provide[AppContainer.exchange],
    persistence: PersistenceService = Provide[AppContainer.persistence],
    portfolio_manager: PortfolioManager = Provide[AppContainer.portfolio_manager],
    notifier: Notifier = Provide[AppContainer.notifier],
    strategy_handler: StrategyHandler = Provide[AppContainer.strategy_handler],
    execution_handler: ExecutionHandler = Provide[AppContainer.execution_handler],
    data_handler: DataHandler = Provide[AppContainer.data_handler],
    regime_detector: MarketRegimeDetector = Provide[AppContainer.regime_detector],
):
    print_banner(config)
    loop = asyncio.get_running_loop()
    await container.init_resources()

    def handle_signal(sig):
        log.warning(f"Received signal {sig}. Initiating graceful shutdown.")
        if not SHUTDOWN_EVENT.is_set(): asyncio.create_task(graceful_shutdown(loop, notifier, config_manager))

    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM): loop.add_signal_handler(sig, handle_signal, sig)
    
    try:
        log.info("🚀 LAUNCHING BOT SERVICES")
        config_manager.start_watch()
        await exchange.initialize()
        await persistence.initialize()
        await regime_detector.initialize()

        log.info("Performing initial data pre-computation...")
        top_symbols = await exchange.get_top_symbols(config.trading.quote_asset, config.trading.top_n_symbols)
        regime_ticker = config.regime.training_ticker
        if regime_ticker not in top_symbols: top_symbols.append(regime_ticker); log.info(f"Added regime ticker '{regime_ticker}' to symbol list.")

        await strategy_handler.initialize_data(top_symbols, config.trading.interval)
        data_handler.start_streams(top_symbols)

        await notifier.send(f"🤖 Bot Started (v{config.version})", "Regime-Aware Auto-Training Bot is now online.", 3447003)

        all_tasks = [
            data_handler.main_socket_task,
            asyncio.create_task(main_event_loop(portfolio_manager, execution_handler, strategy_handler)),
            asyncio.create_task(periodic_task_loop(portfolio_manager.manage_open_positions, config.position_manage_interval_sec, "Position Management")),
            asyncio.create_task(periodic_task_loop(portfolio_manager.update_and_log_portfolio_status, config.portfolio_status_interval_sec, "Portfolio Status", check_drawdown=True)),
            asyncio.create_task(periodic_task_loop(refresh_symbol_list, config.symbol_refresh_interval_sec, "Symbol Refresh", exchange, data_handler, config)),
        ]
        if config.regime.enabled and config.regime.retraining_interval_sec > 0:
            all_tasks.append(asyncio.create_task(periodic_task_loop(periodic_hmm_retraining, config.regime.retraining_interval_sec, "HMM Retraining", regime_detector, notifier)))

        await asyncio.gather(*[task for task in all_tasks if task is not None])

    except asyncio.CancelledError: log.info("Main tasks cancelled during shutdown.")
    except Exception as e:
        log.critical("Fatal error during bot startup.", error=str(e), exc_info=True)
        await notifier.send("💥 BOT CRASHED 💥", f"A critical error occurred during startup: {e}", 15158332, is_error=True)
    finally:
        if not SHUTDOWN_EVENT.is_set(): await graceful_shutdown(loop, notifier, config_manager)


async def graceful_shutdown(loop: asyncio.AbstractEventLoop, notifier: Notifier, config_manager: ConfigManager):
    if SHUTDOWN_EVENT.is_set(): return
    log.warning(f"{Fore.YELLOW}Shutdown signal received, initiating graceful shutdown...{Style.RESET_ALL}")
    SHUTDOWN_EVENT.set()
    await notifier.send("🤖 Bot Stopping...", "Bot has received a shutdown signal.", 16776960)
    
    await asyncio.sleep(1)
    tasks = [t for t in asyncio.all_tasks(loop=loop) if t is not asyncio.current_task()]
    for task in tasks: task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

    await config_manager.stop_watch()
    await container.shutdown_resources()
    log.info(f"{Fore.GREEN}✅ Shutdown complete.")


if __name__ == "__main__":
    container = AppContainer()
    try:
        cfg_manager = container.config_manager()
        cfg = cfg_manager.config
        container.wire(modules=[__name__])
    except Exception as e:
        print(f"CRITICAL: Failed to initialize configuration. Check your .env file and settings. Error: {e}")
        sys.exit(1)

    setup_logging(cfg.log_level, cfg.is_production)
    log = structlog.get_logger("TradingBotV8.7-Unified-Main")

    if not cfg.api.binance_key or not cfg.api.binance_secret:
        log.critical("FATAL: API keys are missing. Please configure them in your .env file before running.")
        sys.exit(1)

    while True:
        try:
            asyncio.run(start_bot())
        except (KeyboardInterrupt, SystemExit):
            log.info("Bot stopped by user or system.")
            break
        except Exception as e:
            log.critical("Unhandled exception in main execution block.", error=str(e), exc_info=True)
            SHUTDOWN_EVENT.clear()
            time.sleep(5)
            log.info("Restarting bot after unexpected error...")
            continue
        else:
            if SHUTDOWN_EVENT.is_set():
                log.info("Bot halted by max drawdown. Exiting.")
                break
            SHUTDOWN_EVENT.clear()
            log.warning("Bot exited and will be restarted automatically.")
            time.sleep(5)
            continue
            
    log.info("Flushing logs and shutting down logging system.")
    logging.shutdown()
