import importlib
import os
import sys
import types
import unittest
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


if __name__ == "__main__":
    unittest.main()
