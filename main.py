"""
Binance USDC spot momentum bot (long-only) with TUI.
This build force-liquidates 100% of a position in ONE market sell order
whenever stop-loss or profit-take (trailing stop) or trend-fail triggers.

Key exit changes:
- cancel_open_orders(symbol) first, wait for locked->free (CANCEL_WAIT_SEC)
- compute full free base balance, floor to step, check minQty/minNotional
- place a SINGLE MARKET SELL for the full amount (newOrderRespType="FULL")
- remove position from tracking immediately (dust below exchange limits is ignored)

Other features:
- USDC-only universe; excludes leveraged tokens.
- WS-first prices (miniTicker); REST fallback if stale.
- Liquidity screen with adaptive fallback.
- Entry: EMA8>EMA21 + RSI>52 + 20-bar breakout on 15m; regime Bullish via 4h RSI>50 vote across proxies.
- Risk: protective stop + trailing stop (ratchet), trend-fail exit, %risk sizing, daily DD pause.
- Imports & manages positions not opened by the bot.
- TUI shows regime vote, equity & day P/L, rate-limit stats, WS status, positions with PnL$ and notional, insights.

Install:
  pip install --upgrade binance-connector websocket-client python-dotenv rich
"""

import os
import sys
import json
import time
import math
import signal
import logging
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from binance.spot import Spot
import websocket

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.align import Align
from rich.live import Live
from rich.text import Text
from rich.rule import Rule

# ---------------------- env helpers ----------------------
def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if v is not None else default

def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return int(float(v))
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    try:
        return float(v)
    except Exception:
        return default

def env_level(name: str, default="INFO") -> int:
    s = env_str(name, default).upper()
    return getattr(logging, s, logging.INFO)

# ---------------------- load env ----------------------
load_dotenv()
API_KEY = env_str("BINANCE_API_KEY", "")
API_SECRET = env_str("BINANCE_API_SECRET", "")
BASE_URL_OVERRIDE = env_str("BINANCE_BASE_URL", "")

VOLUME_THRESHOLD = env_float("VOLUME_THRESHOLD", 1_000_000.0)
VOLUME_FLOOR = env_float("VOLUME_FLOOR", 10_000.0)
HIGHER_TF = env_str("HIGHER_TF", "4h")
ENTRY_TF = env_str("ENTRY_TF", "15m")
RSI_PERIOD = env_int("RSI_PERIOD", 14)
TREND_RSI_THRESHOLD = env_float("TREND_RSI_THRESHOLD", 50.0)
MOMENTUM_RSI_THRESHOLD = env_float("MOMENTUM_RSI_THRESHOLD", 52.0)
BREAKOUT_LOOKBACK = env_int("BREAKOUT_LOOKBACK", 20)
EMA_FAST = env_int("EMA_FAST", 8)
EMA_SLOW = env_int("EMA_SLOW", 21)
STOP_LOSS_PCT = env_float("STOP_LOSS_PCT", 0.02)
TRAILING_STOP_PCT = env_float("TRAILING_STOP_PCT", 0.02)
RISK_PCT = env_float("RISK_PCT", 0.01)
DAILY_LOSS_LIMIT = env_float("DAILY_LOSS_LIMIT", 0.05)
SCAN_INTERVAL_SEC = env_int("SCAN_INTERVAL_SEC", 2)
WS_STALE_SEC = env_int("WS_STALE_SEC", 3)
REST_RETRIES = env_int("REST_RETRIES", 5)
REST_BACKOFF_BASE = env_float("REST_BACKOFF_BASE", 0.3)
LOG_LEVEL = env_level("LOG_LEVEL", "WARNING")
RATE_LIMIT_PER_MIN = env_int("RATE_LIMIT_PER_MIN", 1100)
MAX_KLINES_PER_TICK = env_int("MAX_KLINES_PER_TICK", 200)
TUI_REFRESH_HZ = env_int("TUI_REFRESH_HZ", 10)
INSIGHTS_TOP_N = env_int("INSIGHTS_TOP_N", 5)
CANCEL_WAIT_SEC = env_int("CANCEL_WAIT_SEC", 4)

if not API_KEY or not API_SECRET:
    print("ERROR: Missing BINANCE_API_KEY or BINANCE_API_SECRET in .env", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
console = Console()

# ---------------------- REST client + rate limiter ----------------------
BASE_URL_CANDIDATES = [u for u in [BASE_URL_OVERRIDE] if u] + [
    "https://api.binance.com",
    "https://api4.binance.com",
    "https://api1.binance.com",
    "https://api2.binance.com",
    "https://api3.binance.com",
]

def choose_base_url() -> str:
    for url in BASE_URL_CANDIDATES:
        try:
            c = Spot(api_key=API_KEY, api_secret=API_SECRET, base_url=url, timeout=10)
            c.ping()
            logging.info(f"Using REST base: {url}")
            return url
        except Exception as e:
            logging.warning(f"Base URL {url} not healthy: {e}")
    return "https://api.binance.com"

BASE_URL = choose_base_url()
spot = Spot(api_key=API_KEY, api_secret=API_SECRET, base_url=BASE_URL, timeout=15)

# Approximate request weights
W_EXCHANGE_INFO = 10
W_ACCOUNT = 10
W_TICKER_24HR_ALL = 40
W_TICKER_PRICE = 1
W_KLINES = 1
W_NEW_ORDER = 1
W_GET_ORDER = 1
W_CANCEL_ALL = 1

class MinuteRateLimiter:
    def __init__(self, budget_per_min: int):
        self.budget = max(100, budget_per_min)
        self.window = int(time.time() // 60)
        self.used = 0
        self.lock = threading.Lock()

    def _roll(self):
        now_win = int(time.time() // 60)
        if now_win != self.window:
            self.window = now_win
            self.used = 0

    def remaining(self) -> int:
        with self.lock:
            self._roll()
            return max(0, self.budget - self.used)

    def consumed(self) -> int:
        with self.lock:
            self._roll()
            return self.used

    def consume(self, weight: int):
        while True:
            with self.lock:
                self._roll()
                if self.used + weight <= self.budget:
                    self.used += weight
                    return
                sleep_s = (self.window + 1) * 60 - time.time() + 0.05
            if sleep_s > 0:
                time.sleep(sleep_s)

rate = MinuteRateLimiter(RATE_LIMIT_PER_MIN)

def rest_call(fn, *args, **kwargs):
    for attempt in range(REST_RETRIES):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            sleep_s = REST_BACKOFF_BASE * (2 ** attempt)
            logging.warning(f"REST error ({fn.__name__}): {e} | retry in {sleep_s:.2f}s")
            time.sleep(sleep_s)
    raise RuntimeError(f"REST failed after {REST_RETRIES} attempts: {fn.__name__}")

def rest_call_w(weight: int, fn, *args, **kwargs):
    rate.consume(weight)
    return rest_call(fn, *args, **kwargs)

# ---------------------- Exchange filters cache + symbol maps ----------------------
SYMBOL_INFO: Dict[str, dict] = {}
BASE_ASSET_CACHE: Dict[str, str] = {}
BASE_TO_USDC: Dict[str, str] = {}

def load_symbol_info():
    info = rest_call_w(W_EXCHANGE_INFO, spot.exchange_info)
    for s in info.get("symbols", []):
        sym = s.get("symbol")
        if not sym:
            continue
        base = s.get("baseAsset", "")
        quote = s.get("quoteAsset", "")
        status = s.get("status", "")
        if status == "TRADING" and quote == "USDC":
            if not any(x in sym for x in ("UP", "DOWN", "BULL", "BEAR")):
                BASE_TO_USDC[base] = sym
        BASE_ASSET_CACHE[sym] = base
        f = {fl["filterType"]: fl for fl in s.get("filters", [])}
        step = f.get("MARKET_LOT_SIZE", f.get("LOT_SIZE", {})).get("stepSize", "0")
        min_qty = f.get("MARKET_LOT_SIZE", f.get("LOT_SIZE", {})).get("minQty", "0")
        min_notional = f.get("NOTIONAL", f.get("MIN_NOTIONAL", {})).get("minNotional", "0")
        tick = f.get("PRICE_FILTER", {}).get("tickSize", "0")
        SYMBOL_INFO[sym] = {
            "stepSize": float(step or 0),
            "minQty": float(min_qty or 0),
            "minNotional": float(min_notional or 0),
            "tickSize": float(tick or 0),
        }

def get_filters(symbol: str) -> dict:
    return SYMBOL_INFO.get(symbol, {"stepSize": 0.0, "minQty": 0.0, "minNotional": 0.0, "tickSize": 0.0})

def get_filters_tuple(symbol: str) -> Tuple[float, float, float]:
    f = get_filters(symbol)
    return float(f.get("stepSize", 0.0) or 0.0), float(f.get("minQty", 0.0) or 0.0), float(f.get("minNotional", 0.0) or 0.0)

def base_asset(symbol: str) -> str:
    return BASE_ASSET_CACHE.get(symbol, symbol.replace("USDC", ""))

def base_to_usdc_symbol(asset: str) -> Optional[str]:
    return BASE_TO_USDC.get(asset)

def floor_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return math.floor(qty / step) * step

# ---------------------- WebSocket miniTicker ----------------------
WS_URL = "wss://stream.binance.com:9443/ws/!miniTicker@arr"
ws_thread = None
ws_app = None
ws_prices: Dict[str, float] = {}
ws_ts: Dict[str, float] = {}
ws_should_run = True
ws_last_heartbeat = 0.0

def on_ws_message(_ws, message):
    now = time.time()
    global ws_last_heartbeat
    ws_last_heartbeat = now
    try:
        data = json.loads(message)
        for item in data:
            s = item.get("s")
            c = item.get("c")
            if s and c is not None:
                ws_prices[s] = float(c)
                ws_ts[s] = now
    except Exception:
        pass

def on_ws_error(_ws, error):
    logging.warning(f"WS error: {error}")

def on_ws_close(_ws, *_):
    logging.info("WS closed.")

def run_ws():
    global ws_app
    while ws_should_run:
        try:
            ws_app = websocket.WebSocketApp(
                WS_URL,
                on_message=on_ws_message,
                on_error=on_ws_error,
                on_close=on_ws_close,
            )
            ws_app.run_forever(ping_interval=25, ping_timeout=10)
        except Exception as e:
            logging.warning(f"WS exception: {e}")
        if ws_should_run:
            time.sleep(2)

def start_ws():
    global ws_thread
    ws_thread = threading.Thread(target=run_ws, daemon=True)
    ws_thread.start()
    logging.info("Websocket connected")

def stop_ws():
    global ws_should_run, ws_app
    ws_should_run = False
    try:
        if ws_app:
            ws_app.close()
    except Exception:
        pass
    if ws_thread and ws_thread.is_alive():
        ws_thread.join(timeout=5)

def last_price(symbol: str) -> Optional[float]:
    t = ws_ts.get(symbol, 0.0)
    if time.time() - t <= WS_STALE_SEC and symbol in ws_prices:
        return ws_prices[symbol]
    try:
        data = rest_call_w(W_TICKER_PRICE, spot.ticker_price, symbol=symbol)
        p = float(data["price"])
        ws_prices[symbol] = p
        ws_ts[symbol] = time.time()
        return p
    except Exception as e:
        logging.warning(f"REST price error {symbol}: {e}")
        return None

# ---------------------- indicators ----------------------
def rsi(closes: List[float], period: int = RSI_PERIOD) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        diff = closes[-i] - closes[-(i + 1)]
        gains += max(0.0, diff)
        losses += max(0.0, -diff)
    if losses == 0:
        return 100.0
    rs = gains / losses
    return 100.0 - (100.0 / (1.0 + rs))

def ema(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    e = values[-period]
    for v in values[-period + 1:]:
        e = (v - e) * k + e
    return e

# ---------------------- klines + caching ----------------------
KLINE_CACHE: Dict[Tuple[str, str], Tuple[List[float], List[float], int]] = {}

def recent_klines(symbol: str, interval: str, limit: int) -> List[List]:
    return rest_call_w(W_KLINES, spot.klines, symbol, interval, limit=limit)

def get_cached_klines(symbol: str, interval: str, limit: int) -> Tuple[List[float], List[float], int]:
    key = (symbol, interval)
    kl = recent_klines(symbol, interval, limit)
    if not kl:
        return [], [], 0
    last_open = int(kl[-1][0])
    cached = KLINE_CACHE.get(key)
    if cached and cached[2] == last_open:
        return cached
    closes = [float(k[4]) for k in kl]
    highs = [float(k[2]) for k in kl]
    KLINE_CACHE[key] = (closes, highs, last_open)
    return KLINE_CACHE[key]

# ---------------------- universe + 24h ticker snapshot ----------------------
VOL_MAP: Dict[str, float] = {}
TICKER_24H_MAP: Dict[str, dict] = {}

def all_usdc_symbols() -> List[str]:
    info = rest_call_w(W_EXCHANGE_INFO, spot.exchange_info)
    out = []
    for s in info.get("symbols", []):
        if s.get("status") != "TRADING":
            continue
        if s.get("quoteAsset") != "USDC":
            continue
        name = s.get("symbol") or ""
        if any(x in name for x in ("UP", "DOWN", "BULL", "BEAR")):
            continue
        out.append(name)
    return out

def volume_screen(symbols: List[str]) -> List[str]:
    global VOL_MAP, TICKER_24H_MAP
    tickers = rest_call_w(W_TICKER_24HR_ALL, spot.ticker_24hr)
    TICKER_24H_MAP = {t.get("symbol"): t for t in tickers}
    VOL_MAP = {t.get("symbol"): float(t.get("quoteVolume", 0.0) or 0.0) for t in tickers}
    vols = {s: VOL_MAP.get(s, 0.0) for s in symbols}
    thresh = VOLUME_THRESHOLD
    while True:
        selected = [s for s in symbols if vols.get(s, 0.0) >= thresh]
        if selected or thresh <= VOLUME_FLOOR:
            return selected
        thresh = max(VOLUME_FLOOR, thresh / 2.0)

def top_by_volume(n: int) -> List[Tuple[str, float]]:
    if not VOL_MAP:
        return []
    usdc_syms = [s for s in VOL_MAP.keys() if s.endswith("USDC")]
    usdc_syms.sort(key=lambda s: VOL_MAP.get(s, 0.0), reverse=True)
    return [(s, VOL_MAP.get(s, 0.0)) for s in usdc_syms[:n]]

def top_gainers(n: int) -> List[Tuple[str, float]]:
    if not TICKER_24H_MAP:
        return []
    items = []
    for s, row in TICKER_24H_MAP.items():
        if not s.endswith("USDC"):
            continue
        try:
            pct = float(row.get("priceChangePercent", 0.0))
        except Exception:
            pct = 0.0
        items.append((s, pct))
    items = [x for x in items if x[0] != "USDCUSDC"]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:n]

# ---------------------- regime proxies ----------------------
REGIME_PROXIES: List[str] = []
LAST_KNOWN_REGIME: str = "Neutral"
CURRENT_REGIME = "Neutral"

def compute_regime_proxies():
    global REGIME_PROXIES
    candidates = ["BTCUSDC", "ETHUSDC", "SOLUSDC", "BNBUSDC"]
    proxies = [s for s in candidates if s in SYMBOL_INFO]
    if len(proxies) < 3 and VOL_MAP:
        usdc_syms = [s for s in VOL_MAP.keys() if s.endswith("USDC") and s in SYMBOL_INFO]
        usdc_syms.sort(key=lambda s: VOL_MAP.get(s, 0.0), reverse=True)
        for s in usdc_syms:
            if s not in proxies:
                proxies.append(s)
            if len(proxies) >= 3:
                break
    if not proxies:
        usdc_all = [s for s in SYMBOL_INFO.keys() if s.endswith("USDC")]
        proxies = usdc_all[:1]
    REGIME_PROXIES = proxies

def higher_tf_trend_up(symbol: str) -> bool:
    limit = RSI_PERIOD + 2
    closes, _, _ = get_cached_klines(symbol, HIGHER_TF, limit)
    r = rsi(closes, RSI_PERIOD) if closes else None
    return bool(r is not None and r > TREND_RSI_THRESHOLD)

def update_regime() -> Tuple[str, List[Tuple[str, Optional[float]]]]:
    global CURRENT_REGIME, LAST_KNOWN_REGIME
    if not REGIME_PROXIES:
        compute_regime_proxies()
    ups = downs = 0
    any_ok = False
    rsi_list: List[Tuple[str, Optional[float]]] = []
    for sym in REGIME_PROXIES:
        closes, _, _ = get_cached_klines(sym, HIGHER_TF, RSI_PERIOD + 2)
        r = rsi(closes, RSI_PERIOD) if closes else None
        rsi_list.append((sym, r))
        if r is None:
            continue
        any_ok = True
        if r > TREND_RSI_THRESHOLD:
            ups += 1
        else:
            downs += 1
    if any_ok:
        CURRENT_REGIME = "Bullish" if ups >= max(1, downs) else "Bearish"
        LAST_KNOWN_REGIME = CURRENT_REGIME
    else:
        CURRENT_REGIME = LAST_KNOWN_REGIME
    return CURRENT_REGIME, rsi_list

# ---------------------- account / orders ----------------------
def account_snapshot() -> dict:
    return rest_call_w(W_ACCOUNT, spot.account)

def account_equity_usdc(open_positions: List[dict], balances: Optional[List[dict]] = None) -> float:
    a = {"balances": balances} if balances is not None else account_snapshot()
    balances = a.get("balances", [])
    free_usdc = 0.0
    for b in balances:
        if b.get("asset") == "USDC":
            free_usdc = float(b.get("free", "0"))
            break
    mtm = sum((last_price(p["symbol"]) or p["entry"]) * p["qty"] for p in open_positions)
    return free_usdc + mtm

def free_usdc_balance() -> float:
    a = account_snapshot()
    for b in a.get("balances", []):
        if b.get("asset") == "USDC":
            return float(b.get("free", "0"))
    return 0.0

def balances_base(asset: str) -> Tuple[float, float, float]:
    a = account_snapshot()
    free = locked = 0.0
    for b in a.get("balances", []):
        if b.get("asset") == asset:
            free = float(b.get("free", "0"))
            locked = float(b.get("locked", "0"))
            break
    return free, locked, free + locked

def cancel_open_orders(symbol: str):
    try:
        rest_call_w(W_CANCEL_ALL, spot.cancel_open_orders, symbol=symbol)
    except Exception as e:
        logging.warning(f"cancel_open_orders({symbol}) failed: {e}")

def ensure_min_notional(symbol: str, quote_amount: float) -> float:
    _, _, min_notional = get_filters_tuple(symbol)
    return max(quote_amount, min_notional * 1.05) if min_notional > 0 else quote_amount

def place_market_buy(symbol: str, quote_amount: float) -> Optional[dict]:
    quote_amount = ensure_min_notional(symbol, quote_amount)
    free_q = free_usdc_balance()
    spend = min(quote_amount, max(0.0, free_q * 0.98))
    if spend <= 10.0:
        return None
    try:
        return rest_call_w(W_NEW_ORDER, spot.new_order,
                           symbol=symbol, side="BUY", type="MARKET",
                           quoteOrderQty=f"{spend:.2f}")
    except Exception as e:
        logging.warning(f"Buy failed {symbol}: {e}")
        return None

def derive_executed_qty(symbol: str, order: dict) -> float:
    try:
        oid = order.get("orderId")
        if oid:
            od = rest_call_w(W_GET_ORDER, spot.get_order, symbol=symbol, orderId=oid)
            q = float(od.get("executedQty", "0"))
            if q > 0:
                return q * 0.999
    except Exception:
        pass
    free, _, _ = balances_base(base_asset(symbol))
    return free * 0.999

def sell_all_now(symbol: str) -> bool:
    """
    Force-liquidate 100% in ONE market order:
      1) cancel all open orders for the symbol
      2) wait up to CANCEL_WAIT_SEC for locked to clear
      3) sell full free base balance in a single MARKET order
    Returns True if a sell was submitted (or position is below exchange limits), False on hard failure.
    """
    cancel_open_orders(symbol)

    asset = base_asset(symbol)
    deadline = time.time() + max(1, CANCEL_WAIT_SEC)
    free, locked, total = balances_base(asset)
    while time.time() < deadline and locked > 0:
        time.sleep(0.25)
        free, locked, total = balances_base(asset)

    px = last_price(symbol)
    if not px or px <= 0:
        return False

    step, min_qty, min_notional = get_filters_tuple(symbol)
    qty = floor_step(free, step if step > 0 else 1.0)

    # If entire holding is below exchange limits, treat as closed (can't be sold).
    if qty <= 0 or (min_qty and qty < min_qty) or (min_notional and qty * px < min_notional):
        logging.info(f"{symbol} total below limits (qty={qty}), treating as dust/closed.")
        return True

    try:
        rest_call_w(W_NEW_ORDER, spot.new_order,
                    symbol=symbol, side="SELL", type="MARKET",
                    quantity=f"{qty:.8f}", newOrderRespType="FULL")
        return True
    except Exception as e:
        logging.warning(f"Single-shot SELL failed {symbol} qty={qty}: {e}")
        return False

# ---------------------- strategy ----------------------
def entry_signal(symbol: str) -> bool:
    limit = max(BREAKOUT_LOOKBACK + 1, EMA_SLOW + 5, RSI_PERIOD + 2)
    closes, highs, _ = get_cached_klines(symbol, ENTRY_TF, limit)
    if len(closes) < limit or len(highs) < limit:
        return False
    r = rsi(closes, RSI_PERIOD)
    if r is None or r < MOMENTUM_RSI_THRESHOLD:
        return False
    efast = ema(closes, EMA_FAST)
    eslow = ema(closes, EMA_SLOW)
    if efast is None or eslow is None or efast <= eslow:
        return False
    last_close = closes[-1]
    prior_high = max(highs[-(BREAKOUT_LOOKBACK + 1):-1])
    return last_close > prior_high

def trend_fail_exit(symbol: str) -> bool:
    limit = max(EMA_SLOW + 5, RSI_PERIOD + 2)
    closes, _, _ = get_cached_klines(symbol, ENTRY_TF, limit)
    if len(closes) < limit:
        return False
    efast = ema(closes, EMA_FAST)
    eslow = ema(closes, EMA_SLOW)
    return bool(efast is not None and eslow is not None and efast < eslow)

# ---------------------- state ----------------------
POSITIONS: List[dict] = []  # {symbol, qty, entry, highest, stop, imported}
ALL_USDC: List[str] = []
UNIVERSE: List[str] = []
DAY_START_EQUITY: Optional[float] = None
LAST_SCANNED = 0
SCAN_CURSOR = 0

def refresh_universe():
    global ALL_USDC, UNIVERSE
    if not ALL_USDC:
        ALL_USDC = all_usdc_symbols()
    UNIVERSE = volume_screen(ALL_USDC)
    compute_regime_proxies()

def maybe_reset_daily_equity():
    global DAY_START_EQUITY
    now = datetime.utcnow()
    if DAY_START_EQUITY is None or (now.hour == 0 and now.minute < 5):
        a = account_snapshot()
        DAY_START_EQUITY = account_equity_usdc(POSITIONS, a.get("balances", []))

def is_paused_by_dd(equity_now: float) -> bool:
    return bool(DAY_START_EQUITY and equity_now < DAY_START_EQUITY * (1.0 - DAILY_LOSS_LIMIT))

def import_external_positions():
    """Sync POSITIONS with current USDC-quoted holdings (even if not opened by the bot)."""
    global POSITIONS
    a = account_snapshot()
    balances = a.get("balances", [])
    desired: Dict[str, float] = {}
    for b in balances:
        asset = b.get("asset") or ""
        if asset in ("", "USDC"):
            continue
        symbol = base_to_usdc_symbol(asset)
        if not symbol:
            continue
        free, locked, total = balances_base(asset)
        if total <= 0:
            continue
        step, _, _ = get_filters_tuple(symbol)
        qty = floor_step(total, step if step > 0 else 1.0)
        if qty <= 0:
            continue
        # minNotional/minQty check will happen on exit; here we track anything tradeable-ish
        desired[symbol] = qty

    by_sym = {p["symbol"]: p for p in POSITIONS}
    for sym, qty in desired.items():
        if sym in by_sym:
            p = by_sym[sym]
            p["qty"] = qty
            if p.get("entry", 0.0) <= 0:
                px = last_price(sym) or 0.0
                p["entry"] = px
                p["highest"] = px
                p["stop"] = px * (1.0 - STOP_LOSS_PCT) if px > 0 else 0.0
            p["imported"] = True
        else:
            px = last_price(sym) or 0.0
            POSITIONS.append({
                "symbol": sym,
                "qty": qty,
                "entry": px,
                "highest": px,
                "stop": px * (1.0 - STOP_LOSS_PCT) if px > 0 else 0.0,
                "imported": True
            })
    POSITIONS[:] = [p for p in POSITIONS if (p["symbol"] in desired) or (not p.get("imported"))]

def evaluate_positions():
    """Stops/TPs sell 100% in ONE shot; remove from book immediately on success or dust."""
    global POSITIONS
    survivors = []
    for pos in POSITIONS:
        sym = pos["symbol"]
        px = last_price(sym)
        if px is None:
            survivors.append(pos)
            continue

        # Trailing stop ratchet up only
        if px > pos["highest"]:
            pos["highest"] = px
            new_stop = pos["highest"] * (1.0 - TRAILING_STOP_PCT)
            if new_stop > pos["stop"]:
                pos["stop"] = new_stop

        stop_hit = (pos["stop"] > 0) and (px <= pos["stop"])
        trend_fail = trend_fail_exit(sym)

        if stop_hit or trend_fail:
            ok = sell_all_now(sym)
            if ok:
                # Position considered closed (dust, if any, ignored)
                continue
            else:
                # If single-shot failed, keep position but tighten stop to retry next tick
                pos["stop"] = min(pos["stop"], px * 0.999)
                survivors.append(pos)
        else:
            survivors.append(pos)
    POSITIONS[:] = survivors

def try_entry(sym: str, equity_now: float):
    if any(p["symbol"] == sym for p in POSITIONS):
        return
    if CURRENT_REGIME != "Bullish":
        return
    if not entry_signal(sym):
        return
    px = last_price(sym)
    if px is None or px <= 0:
        return
    risk_amt = equity_now * RISK_PCT
    quote_amt = risk_amt / STOP_LOSS_PCT
    order = place_market_buy(sym, quote_amt)
    if not order:
        return
    qty = derive_executed_qty(sym, order)
    step, min_qty, _ = get_filters_tuple(sym)
    if step > 0:
        qty = floor_step(qty, step)
    if qty <= 0 or (min_qty > 0 and qty < min_qty):
        return
    POSITIONS.append({
        "symbol": sym,
        "qty": qty,
        "entry": px,
        "highest": px,
        "stop": px * (1.0 - STOP_LOSS_PCT),
        "imported": False
    })

# ---------------------- TUI ----------------------
def build_layout(summary_panel: Panel, positions_panel: Panel, insights_panel: Panel) -> Layout:
    layout = Layout()
    layout.split_column(
        Layout(summary_panel, name="summary", size=9),
        Layout(name="body")
    )
    layout["body"].split_row(
        Layout(positions_panel, name="positions"),
        Layout(insights_panel, name="insights", size=48),
    )
    return layout

def summary_view(equity: float, paused: bool, regime_details: List[Tuple[str, Optional[float]]]) -> Panel:
    uni_size = len(UNIVERSE)
    status = "PAUSED" if paused else "ACTIVE"
    remaining = rate.remaining()
    used = rate.consumed()
    fresh = sum(1 for _, ts in ws_ts.items() if time.time() - ts <= WS_STALE_SEC)
    last_hb_age = time.time() - ws_last_heartbeat if ws_last_heartbeat > 0 else float("inf")

    dd_str = ""
    if DAY_START_EQUITY:
        dd = (equity - DAY_START_EQUITY) / DAY_START_EQUITY
        dd_str = f" | Day P/L: {dd:+.2%}"

    regime_text = Text()
    regime_text.append(" Regime: ", style="bold")
    regime_text.append(f"{CURRENT_REGIME}", style=("bold green" if CURRENT_REGIME == "Bullish" else ("bold red" if CURRENT_REGIME == "Bearish" else "bold yellow")))
    regime_text.append("  [")
    if regime_details:
        chunks = []
        for s, r in regime_details:
            chunks.append(f"{s}: {'n/a' if r is None else f'{r:.1f}'}")
        regime_text.append(", ".join(chunks))
    else:
        regime_text.append("n/a")
    regime_text.append("]\n")

    txt = Text()
    txt.append(regime_text)
    txt.append(f" Equity: {equity:,.2f} USDC{dd_str}\n")
    txt.append(f" Universe: {uni_size}  (scanned last tick: {LAST_SCANNED})  Positions: {len(POSITIONS)}\n")
    txt.append(" Status: ")
    txt.append(status, style="yellow" if paused else "green")
    txt.append(f"   Base: {BASE_URL}\n")
    txt.append(f" RateLimit: used {used}/{RATE_LIMIT_PER_MIN} | rem {remaining}   WS: fresh={fresh} last_hb={0 if last_hb_age==float('inf') else int(last_hb_age)}s\n")
    return Panel(Align.left(txt), title="USDC Momentum Bot (FORCE-EXIT)", border_style="cyan")

def positions_view() -> Panel:
    table = Table(expand=True, show_edge=True)
    table.add_column("Symbol", style="bold")
    table.add_column("Qty", justify="right")
    table.add_column("Entry", justify="right")
    table.add_column("Last", justify="right")
    table.add_column("Notional", justify="right")
    table.add_column("Stop", justify="right")
    table.add_column("High", justify="right")
    table.add_column("Src", justify="left")
    table.add_column("PnL %", justify="right")
    table.add_column("PnL $", justify="right")
    if not POSITIONS:
        return Panel(Align.center("[dim]No open positions[/dim]"), title="Positions")
    for p in POSITIONS:
        sym = p["symbol"]
        last = last_price(sym) or p["entry"]
        notional = (last or 0) * (p["qty"] or 0)
        pnl_pct = ((last - p["entry"]) / p["entry"]) * 100.0 if p["entry"] else 0.0
        pnl_usd = (last - p["entry"]) * (p["qty"] or 0)
        src = "import" if p.get("imported") else "bot"
        table.add_row(
            sym,
            f"{p['qty']:.6f}",
            f"{(p['entry'] or 0):.6f}",
            f"{last:.6f}",
            f"{notional:,.2f}",
            f"{(p['stop'] or 0):.6f}",
            f"{(p['highest'] or 0):.6f}",
            src,
            f"{pnl_pct:+.2f}%",
            f"{pnl_usd:+.2f}"
        )
    return Panel(table, title="Positions")

def insights_view(equity: float) -> Panel:
    free_q = free_usdc_balance()
    open_notional = sum((last_price(p["symbol"]) or p["entry"]) * (p["qty"] or 0) for p in POSITIONS)
    bots = sum(1 for p in POSITIONS if not p.get("imported"))
    imports = sum(1 for p in POSITIONS if p.get("imported"))

    top_vol = top_by_volume(INSIGHTS_TOP_N)
    gainers = top_gainers(INSIGHTS_TOP_N)

    t = Table.grid(expand=True)
    t.add_row(Text("Account", style="bold underline"))
    t.add_row(f" Free USDC: {free_q:,.2f}   Open Exposure: {open_notional:,.2f}   Equity≈ {equity:,.2f}")
    t.add_row(f" Positions: total={len(POSITIONS)} (bot={bots}, imported={imports})")
    t.add_row(Rule())

    t.add_row(Text("Top USDC by 24h Volume", style="bold underline"))
    if top_vol:
        tv = Table(box=None, expand=True)
        tv.add_column("Symbol")
        tv.add_column("24h Quote Vol (USDC)", justify="right")
        for s, v in top_vol:
            tv.add_row(s, f"{v:,.0f}")
        t.add_row(tv)
    else:
        t.add_row("[dim]n/a[/dim]")

    t.add_row(Rule())
    t.add_row(Text("Top Gainers (24h %)", style="bold underline"))
    if gainers:
        tg = Table(box=None, expand=True)
        tg.add_column("Symbol")
        tg.add_column("%", justify="right")
        for s, pct in gainers:
            tg.add_row(s, f"{pct:+.2f}%")
        t.add_row(tg)
    else:
        t.add_row("[dim]n/a[/dim]")

    return Panel(t, title="Insights", border_style="magenta")

def render_ui(equity: float, paused: bool, regime_details: List[Tuple[str, Optional[float]]]) -> Layout:
    return build_layout(
        summary_view(equity, paused, regime_details),
        positions_view(),
        insights_view(equity),
    )

# ---------------------- main loop ----------------------
ws_ts: Dict[str, float] = {}
stop_event = threading.Event()

def handle_sig(signum, frame):
    stop_event.set()

for sig in (getattr(signal, "SIGINT", None), getattr(signal, "SIGTERM", None)):
    if sig is not None:
        try:
            signal.signal(sig, handle_sig)
        except Exception:
            pass

def main():
    global LAST_SCANNED, SCAN_CURSOR, ALL_USDC, UNIVERSE, CURRENT_REGIME
    start_ws()
    load_symbol_info()
    last_universe_refresh = 0.0

    a = account_snapshot()
    eq = account_equity_usdc(POSITIONS, a.get("balances", [])) or 0.0
    paused = False
    refresh_universe()

    regime, regime_details = update_regime()

    with Live(render_ui(eq, paused, regime_details), refresh_per_second=TUI_REFRESH_HZ, screen=True, console=console) as live:
        while not stop_event.is_set():
            try:
                maybe_reset_daily_equity()
                regime, regime_details = update_regime()

                now = time.time()
                if now - last_universe_refresh > max(30, SCAN_INTERVAL_SEC * 5):
                    ALL_USDC = all_usdc_symbols()
                    UNIVERSE = volume_screen(ALL_USDC)
                    compute_regime_proxies()
                    last_universe_refresh = now
                    SCAN_CURSOR = SCAN_CURSOR % max(1, len(UNIVERSE))

                # Always import/refresh positions first
                import_external_positions()

                a = account_snapshot()
                eq = account_equity_usdc(POSITIONS, a.get("balances", [])) or 0.0
                paused = is_paused_by_dd(eq)

                # Determine batch size for entries
                remain = rate.remaining()
                reserve = 50
                sym_budget = max(0, remain - reserve) // 2
                batch = int(min(MAX_KLINES_PER_TICK, sym_budget, len(UNIVERSE) or 0))
                if paused:
                    batch = 0

                LAST_SCANNED = 0
                if batch > 0 and UNIVERSE and CURRENT_REGIME == "Bullish":
                    end = SCAN_CURSOR + batch
                    if end <= len(UNIVERSE):
                        batch_syms = UNIVERSE[SCAN_CURSOR:end]
                        SCAN_CURSOR = end % len(UNIVERSE)
                    else:
                        batch_syms = UNIVERSE[SCAN_CURSOR:] + UNIVERSE[:end - len(UNIVERSE)]
                        SCAN_CURSOR = (end - len(UNIVERSE)) % len(UNIVERSE)

                    for sym in batch_syms:
                        LAST_SCANNED += 1
                        try_entry(sym, eq)

                # Manage exits (force 100% liquidation in one order)
                evaluate_positions()

                live.update(render_ui(eq, paused, regime_details))

                for _ in range(SCAN_INTERVAL_SEC):
                    if stop_event.is_set():
                        break
                    time.sleep(1)

            except Exception as e:
                console.log(f"[red]Error:[/red] {e}")
                time.sleep(1.0)

    stop_ws()

if __name__ == "__main__":
    main()

# ==============================================================================
# 🚀 AGGRESSIVE MOMENTUM TUNING (v9.2.0-tuned)
# FIX: The volume filter has been drastically lowered to ensure the bot can
# build a trading universe even in very low-volume market conditions.
# ==============================================================================
# ---------------------------------
# Binance API Keys (template values - replace before running)
# ---------------------------------
# BINANCE_API_KEY="YOUR_BINANCE_API_KEY"
# BINANCE_API_SECRET="YOUR_BINANCE_API_SECRET"
# DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/your-webhook"
# BINANCE_BASE_URL=https://api-gcp.binance.com
#
# VOLUME_THRESHOLD=1000000
# VOLUME_FLOOR=10000
# HIGHER_TF=4h
# ENTRY_TF=15m
# RSI_PERIOD=14
# TREND_RSI_THRESHOLD=50
# MOMENTUM_RSI_THRESHOLD=52
# BREAKOUT_LOOKBACK=20
# EMA_FAST=8
# EMA_SLOW=21
# STOP_LOSS_PCT=0.02
# TRAILING_STOP_PCT=0.02
# RISK_PCT=0.01
# DAILY_LOSS_LIMIT=0.05
# SCAN_INTERVAL_SEC=1
# WS_STALE_SEC=3
# REST_RETRIES=5
# REST_BACKOFF_BASE=0.3
# LOG_LEVEL=WARNING
# RATE_LIMIT_PER_MIN=1100
# MAX_KLINES_PER_TICK=333
# TUI_REFRESH_HZ=10
# INSIGHTS_TOP_N=5
# CANCEL_WAIT_SEC=4
