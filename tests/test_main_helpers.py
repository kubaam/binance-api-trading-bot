import importlib
import os
import sys
import types
import unittest
from typing import List
from unittest import mock



class DummySpot:
    """Minimal stub replicating required Spot client behaviour for tests."""

    def __init__(self, *args, **kwargs):
        self._orders = []

    # Connectivity -------------------------------------------------
    def ping(self):
        return None

    # Market + account data ---------------------------------------
    def exchange_info(self):
        return {"symbols": []}

    def account(self):
        return {"balances": [{"asset": "USDC", "free": "100", "locked": "0"}]}

    def ticker_24hr(self):
        return []

    def ticker_price(self, symbol):
        return {"price": "1"}

    def klines(self, symbol, interval, limit):
        # deterministic increasing series for indicator tests
        data = []
        base = 100.0
        for i in range(limit):
            ts = (i + 1) * 60_000
            close = base + (i * 0.5)
            data.append([ts, base, base + 1, base - 1, close, 0, 0, 0, 0, 0, 0, 0])
        return data

    # Trading -----------------------------------------------------
    def cancel_open_orders(self, symbol):
        return {"symbol": symbol}

    def new_order(self, **kwargs):
        order_id = len(self._orders) + 1
        self._orders.append({"orderId": order_id, **kwargs})
        return {"orderId": order_id, **kwargs}

    def get_order(self, **kwargs):
        return {"executedQty": "1"}


def register_stubbed_dependency(module_name: str, **attrs):
    module = types.ModuleType(module_name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[module_name] = module
    return module


class _DummyConsole:
    def __init__(self, *args, **kwargs):
        pass

    def log(self, *args, **kwargs):
        pass


class _DummyTable:
    def __init__(self, *args, **kwargs):
        self.columns = []
        self.rows = []

    def add_column(self, *args, **kwargs):
        self.columns.append((args, kwargs))

    def add_row(self, *args, **kwargs):
        self.rows.append((args, kwargs))


class _DummyPanel:
    def __init__(self, *args, **kwargs):
        pass


class _DummyLayout:
    def __init__(self, *args, **kwargs):
        self.children = {}

    def split_column(self, *args, **kwargs):
        return None

    def split_row(self, *args, **kwargs):
        return None

    def __getitem__(self, key):
        self.children.setdefault(key, _DummyLayout())
        return self.children[key]


class _DummyAlign:
    def __init__(self, renderable, *args, **kwargs):
        self.renderable = renderable


class _DummyLive:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def update(self, *args, **kwargs):
        pass


class _DummyText:
    def __init__(self, *args, **kwargs):
        self.buffer = []

    def append(self, text, *args, **kwargs):
        self.buffer.append(text)


class _DummyRule:
    def __init__(self, *args, **kwargs):
        pass


class _DummyWebSocketApp:
    def __init__(self, *args, **kwargs):
        pass

    def run_forever(self, *args, **kwargs):
        return None

    def close(self):
        return None


def import_main_with_stubs():
    if "main" in sys.modules:
        del sys.modules["main"]

    # Inject stubbed binance module hierarchy before import
    fake_binance = types.ModuleType("binance")
    fake_spot = types.ModuleType("binance.spot")
    fake_spot.Spot = DummySpot
    fake_binance.spot = fake_spot
    sys.modules["binance"] = fake_binance
    sys.modules["binance.spot"] = fake_spot

    # Stub dotenv
    register_stubbed_dependency("dotenv", load_dotenv=lambda *args, **kwargs: None)

    # Stub websocket client
    register_stubbed_dependency("websocket", WebSocketApp=_DummyWebSocketApp)

    # Stub rich namespace and submodules accessed by the bot
    rich_module = register_stubbed_dependency("rich")
    register_stubbed_dependency("rich.console", Console=_DummyConsole)
    register_stubbed_dependency("rich.table", Table=_DummyTable)
    register_stubbed_dependency("rich.panel", Panel=_DummyPanel)
    register_stubbed_dependency("rich.layout", Layout=_DummyLayout)
    register_stubbed_dependency("rich.align", Align=_DummyAlign)
    register_stubbed_dependency("rich.live", Live=_DummyLive)
    register_stubbed_dependency("rich.text", Text=_DummyText)
    register_stubbed_dependency("rich.rule", Rule=_DummyRule)
    # Ensure attribute resolution via parent package works
    rich_module.console = sys.modules["rich.console"]
    rich_module.table = sys.modules["rich.table"]
    rich_module.panel = sys.modules["rich.panel"]
    rich_module.layout = sys.modules["rich.layout"]
    rich_module.align = sys.modules["rich.align"]
    rich_module.live = sys.modules["rich.live"]
    rich_module.text = sys.modules["rich.text"]
    rich_module.rule = sys.modules["rich.rule"]

    os.environ.setdefault("BINANCE_API_KEY", "test-key")
    os.environ.setdefault("BINANCE_API_SECRET", "test-secret")

    return importlib.import_module("main")


MAIN = import_main_with_stubs()


class EnvHelperTests(unittest.TestCase):
    def test_env_str_default_and_strip(self):
        os.environ["TEST_ENV_STR"] = " value "
        self.assertEqual(MAIN.env_str("TEST_ENV_STR", "fallback"), "value")
        del os.environ["TEST_ENV_STR"]
        self.assertEqual(MAIN.env_str("TEST_ENV_STR", "fallback"), "fallback")

    def test_env_numeric_parsing_with_invalid(self):
        os.environ["TEST_ENV_INT"] = "123.7"
        self.assertEqual(MAIN.env_int("TEST_ENV_INT", 0), 123)
        os.environ["TEST_ENV_FLOAT"] = "abc"
        self.assertAlmostEqual(MAIN.env_float("TEST_ENV_FLOAT", 1.5), 1.5)


class MathHelperTests(unittest.TestCase):
    def test_floor_step_handles_zero_step(self):
        self.assertEqual(MAIN.floor_step(5.4321, 0), 5.4321)
        self.assertEqual(MAIN.floor_step(5.4321, 0.1), 5.4)

    def test_rsi_monotonic_increase(self):
        closes = [float(i) for i in range(30)]
        value = MAIN.rsi(closes, period=14)
        self.assertIsNotNone(value)
        self.assertGreater(value, 70.0)


class RestCallTests(unittest.TestCase):
    @mock.patch("main.time.sleep", return_value=None)
    def test_rest_call_retries_then_raises(self, mocked_sleep):
        call_counter = {"count": 0}

        def failing_fn():
            call_counter["count"] += 1
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            MAIN.rest_call(failing_fn)
        self.assertEqual(call_counter["count"], MAIN.REST_RETRIES)


class TrendMLTests(unittest.TestCase):
    def setUp(self):
        MAIN.TREND_ML_CACHE.clear()
        MAIN.TREND_ML_DATASETS.clear()
        MAIN.TREND_ML_DATA_VERSION = 0

    def test_trend_ml_probability_prefers_uptrend(self):
        required = MAIN.TREND_ML_REQUIRED_BARS + 20
        price = 100.0
        closes = [price]
        pattern = [0.8, -0.3, 0.9, -0.2, 1.0, -0.1]
        idx = 0
        while len(closes) < required:
            delta = pattern[idx % len(pattern)]
            price = max(1.0, price + delta)
            closes.append(price)
            idx += 1
        prob = MAIN.trend_ml_probability("TESTUSDC", MAIN.ENTRY_TF, closes, last_open=123456)
        self.assertIsNotNone(prob)
        self.assertGreater(prob, 0.5)

    def test_trend_ml_probability_requires_mixed_classes(self):
        MAIN.TREND_ML_CACHE.clear()
        closes = [100.0 + 0.5 * i for i in range(MAIN.TREND_ML_REQUIRED_BARS + 5)]
        prob = MAIN.trend_ml_probability("ONEUSDC", MAIN.ENTRY_TF, closes, last_open=789)
        self.assertIsNone(prob)

    def test_trend_ml_probability_requires_history(self):
        prob = MAIN.trend_ml_probability("SMALLUSDC", MAIN.ENTRY_TF, [100.0, 100.5], last_open=1)
        self.assertIsNone(prob)

    def test_trend_ml_probability_aggregates_across_symbols(self):
        def build_closes(start: float, deltas: List[float], count: int) -> List[float]:
            closes = [start]
            idx = 0
            while len(closes) < count:
                closes.append(max(1.0, closes[-1] + deltas[idx % len(deltas)]))
                idx += 1
            return closes

        required = MAIN.TREND_ML_REQUIRED_BARS + 10
        closes_a = build_closes(100.0, [0.6, -0.4, 0.7, -0.2], required)
        closes_b = build_closes(80.0, [0.5, -0.6, 0.8, -0.3], required)

        class DummyModel:
            def __init__(self):
                self.trained = None

            def fit(self, X, y):
                self.trained = (MAIN.np.array(X, copy=True), MAIN.np.array(y, copy=True))
                return self

            def predict_proba(self, X):
                return MAIN.np.array([0.6])

        # Prime dataset for first symbol
        first_model = DummyModel()
        with mock.patch.object(MAIN, "TrendLogisticModel", return_value=first_model):
            prob_a = MAIN.trend_ml_probability("AAAUSDC", MAIN.ENTRY_TF, closes_a, last_open=101)
        self.assertIsNotNone(prob_a)
        self.assertIsNotNone(first_model.trained)

        second_model = DummyModel()
        with mock.patch.object(MAIN, "TrendLogisticModel", return_value=second_model):
            prob_b = MAIN.trend_ml_probability("BBBUSDC", MAIN.ENTRY_TF, closes_b, last_open=202)
        self.assertIsNotNone(prob_b)
        self.assertIsNotNone(second_model.trained)

        returns_a = MAIN._log_returns(closes_a)
        returns_b = MAIN._log_returns(closes_b)
        dataset_a = MAIN._build_trend_dataset(returns_a)
        dataset_b = MAIN._build_trend_dataset(returns_b)
        self.assertIsNotNone(dataset_a)
        self.assertIsNotNone(dataset_b)
        expected_rows = dataset_a[0].shape[0] + dataset_b[0].shape[0]
        trained_rows = second_model.trained[0].shape[0]
        self.assertEqual(trained_rows, expected_rows)


class EntrySignalIntegrationTests(unittest.TestCase):
    def test_entry_signal_respects_ml_gate(self):
        limit = MAIN.TREND_ML_REQUIRED_BARS + 5
        closes = [float(i + 1) for i in range(limit)]
        highs = list(closes)

        def ema_side_effect(values, period):
            if period == MAIN.EMA_FAST:
                return 110.0
            if period == MAIN.EMA_SLOW:
                return 100.0
            return 100.0

        with mock.patch.object(MAIN, "get_cached_klines", return_value=(closes, highs, 999)), \
             mock.patch.object(MAIN, "rsi", return_value=MAIN.MOMENTUM_RSI_THRESHOLD + 5), \
             mock.patch.object(MAIN, "ema", side_effect=ema_side_effect):
            with mock.patch.object(MAIN, "trend_ml_decision", return_value=False):
                self.assertFalse(MAIN.entry_signal("XYZUSDC"))
            with mock.patch.object(MAIN, "trend_ml_decision", return_value=True):
                self.assertTrue(MAIN.entry_signal("XYZUSDC"))

if __name__ == "__main__":
    unittest.main()
