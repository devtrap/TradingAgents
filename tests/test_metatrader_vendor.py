"""MetaTrader 5 vendor: symbol resolution, look-ahead safety, and output shape.

The MetaTrader5 package is Windows-only, so every test here runs against a stub
terminal injected into sys.modules. That is enough to pin the logic that actually
breaks in practice: broker-suffix resolution, the look-ahead cut, and the router's
expected exception types.
"""

from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta

import numpy as np
import pytest

RATE_DTYPE = np.dtype([
    ("time", "<i8"), ("open", "<f8"), ("high", "<f8"), ("low", "<f8"),
    ("close", "<f8"), ("tick_volume", "<u8"), ("spread", "<i4"), ("real_volume", "<u8"),
])


def _synthetic_rates(days: int = 400, end: datetime | None = None):
    """Business-day bars ending today, drifting gently upward."""
    end = end or datetime.utcnow()
    rows = []
    price = 1.0850
    day = end - timedelta(days=days)
    while day <= end:
        if day.weekday() < 5:
            price += 0.0004 if day.day % 3 else -0.0006
            stamp = int(datetime(day.year, day.month, day.day).timestamp())
            rows.append((stamp, price, price + 0.0021, price - 0.0018, price + 0.0007,
                         14000, 8, 0))
        day += timedelta(days=1)
    return np.array(rows, dtype=RATE_DTYPE)


class _StubMT5(types.ModuleType):
    """Minimal MetaTrader5 stand-in. Lists EURUSD under a broker suffix."""

    TIMEFRAME_D1 = 16408
    LISTED = ("EURUSD.a", "XAUUSD", "US500", "AAPL")

    def __init__(self, name="MetaTrader5", rates=None, can_init=True):
        super().__init__(name)
        self._rates = rates if rates is not None else _synthetic_rates()
        self._can_init = can_init
        self.selected = []

    def initialize(self, *a, **k):
        return self._can_init

    def last_error(self):
        return (-1, "stub")

    def terminal_info(self):
        return types.SimpleNamespace(company="Stub Broker")

    def account_info(self):
        return types.SimpleNamespace(login=1234)

    def symbol_info(self, name):
        return types.SimpleNamespace(name=name) if name in self.LISTED else None

    def symbol_select(self, name, enable=True):
        self.selected.append(name)
        return name in self.LISTED

    def symbols_get(self):
        return [types.SimpleNamespace(name=n) for n in self.LISTED]

    def copy_rates_range(self, symbol, timeframe, utc_from, utc_to):
        return self._rates

    def shutdown(self):
        return None


@pytest.fixture()
def mt5_vendor(monkeypatch):
    """Fresh vendor module bound to a fresh stub terminal."""
    stub = _StubMT5()
    monkeypatch.setitem(sys.modules, "MetaTrader5", stub)
    sys.modules.pop("tradingagents.dataflows.metatrader", None)
    import tradingagents.dataflows.metatrader as mt

    mt.clear_cache()
    yield mt, stub
    mt.clear_cache()
    sys.modules.pop("tradingagents.dataflows.metatrader", None)


# --- symbol resolution ----------------------------------------------------

def test_forex_yahoo_symbol_resolves_through_broker_suffix(mt5_vendor):
    mt, stub = mt5_vendor
    assert mt.resolve_mt5_symbol("EURUSD=X") == "EURUSD.a"
    assert "EURUSD.a" in stub.selected  # must be added to MarketWatch


def test_metal_future_maps_back_to_spot_cfd(mt5_vendor):
    mt, _ = mt5_vendor
    assert mt.resolve_mt5_symbol("GC=F") == "XAUUSD"


def test_index_future_maps_to_cfd_name(mt5_vendor):
    mt, _ = mt5_vendor
    assert mt.resolve_mt5_symbol("^GSPC") == "US500"


def test_equity_passes_through(mt5_vendor):
    mt, _ = mt5_vendor
    assert mt.resolve_mt5_symbol("AAPL") == "AAPL"


def test_unlisted_symbol_raises_no_market_data(mt5_vendor):
    mt, _ = mt5_vendor
    from tradingagents.dataflows.errors import NoMarketDataError

    with pytest.raises(NoMarketDataError):
        mt.resolve_mt5_symbol("NOPE=X")


# --- look-ahead safety ----------------------------------------------------

def test_no_bars_after_the_analysis_date(mt5_vendor):
    mt, _ = mt5_vendor
    cutoff = (datetime.utcnow() - timedelta(days=90)).strftime("%Y-%m-%d")
    frame = mt.load_mt5_ohlcv("EURUSD=X", cutoff)
    assert frame["Date"].max().strftime("%Y-%m-%d") <= cutoff


def test_stock_data_window_respects_both_bounds(mt5_vendor):
    mt, _ = mt5_vendor
    end = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=60)).strftime("%Y-%m-%d")
    out = mt.get_stock_data("EURUSD=X", start, end)
    dates = [ln.split(",")[0] for ln in out.splitlines() if ln[:2].isdigit()]
    assert dates and min(dates) >= start and max(dates) <= end


# --- output shape ---------------------------------------------------------

def test_stock_data_has_header_and_csv(mt5_vendor):
    mt, _ = mt5_vendor
    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=20)).strftime("%Y-%m-%d")
    out = mt.get_stock_data("EURUSD=X", start, end)
    assert out.startswith("# MetaTrader 5 daily data for EURUSD.a")
    assert "Date,Open,High,Low,Close,Volume" in out


def test_indicator_output_matches_yfinance_shape(mt5_vendor):
    mt, _ = mt5_vendor
    curr = datetime.utcnow().strftime("%Y-%m-%d")
    out = mt.get_indicators("EURUSD=X", "rsi", curr, 10)
    assert out.startswith("## rsi values from")
    assert "RSI: Measures momentum" in out
    assert len([ln for ln in out.splitlines() if ln.startswith("20")]) == 11


def test_long_window_indicator_computes(mt5_vendor):
    mt, _ = mt5_vendor
    curr = datetime.utcnow().strftime("%Y-%m-%d")
    out = mt.get_indicators("EURUSD=X", "close_200_sma", curr, 5)
    values = [ln.split(": ", 1)[1] for ln in out.splitlines() if ln.startswith("20")]
    assert any(v not in ("N/A",) and "Not a trading day" not in v for v in values)


def test_unsupported_indicator_raises_value_error(mt5_vendor):
    mt, _ = mt5_vendor
    with pytest.raises(ValueError, match="not supported"):
        mt.get_indicators("EURUSD=X", "supertrend", "2026-08-31", 10)


# --- router contract ------------------------------------------------------

def test_missing_package_is_not_configured(monkeypatch):
    """Router must be able to fall through to yfinance on a machine without MT5."""
    monkeypatch.setitem(sys.modules, "MetaTrader5", None)
    sys.modules.pop("tradingagents.dataflows.metatrader", None)
    import tradingagents.dataflows.metatrader as mt
    from tradingagents.dataflows.errors import VendorNotConfiguredError

    monkeypatch.setattr(mt, "_initialized", False)
    with pytest.raises(VendorNotConfiguredError):
        mt.resolve_mt5_symbol("EURUSD=X")
    sys.modules.pop("tradingagents.dataflows.metatrader", None)


def test_terminal_refusing_to_init_is_not_configured(monkeypatch):
    stub = _StubMT5(can_init=False)
    monkeypatch.setitem(sys.modules, "MetaTrader5", stub)
    sys.modules.pop("tradingagents.dataflows.metatrader", None)
    import tradingagents.dataflows.metatrader as mt
    from tradingagents.dataflows.errors import VendorNotConfiguredError

    with pytest.raises(VendorNotConfiguredError, match="Could not connect"):
        mt.resolve_mt5_symbol("EURUSD=X")
    sys.modules.pop("tradingagents.dataflows.metatrader", None)


# --- router integration ---------------------------------------------------

def test_router_dispatches_to_metatrader(mt5_vendor, monkeypatch):
    """data_vendors={'core_stock_apis': 'metatrader'} must reach this vendor."""
    mt, _ = mt5_vendor
    from tradingagents.dataflows.config import set_config
    from tradingagents.dataflows.interface import VENDOR_METHODS, route_to_vendor

    monkeypatch.setitem(VENDOR_METHODS["get_stock_data"], "metatrader", mt.get_stock_data)
    set_config({"data_vendors": {"core_stock_apis": "metatrader"}})

    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=15)).strftime("%Y-%m-%d")
    out = route_to_vendor("get_stock_data", "EURUSD=X", start, end)
    assert "MetaTrader 5 daily data" in out


def test_router_falls_back_to_yfinance_when_terminal_absent(monkeypatch):
    """The whole point of VendorNotConfiguredError: 'metatrader,yfinance' degrades."""
    monkeypatch.setitem(sys.modules, "MetaTrader5", None)
    sys.modules.pop("tradingagents.dataflows.metatrader", None)
    import tradingagents.dataflows.metatrader as mt
    from tradingagents.dataflows.config import set_config
    from tradingagents.dataflows.interface import VENDOR_METHODS, route_to_vendor

    monkeypatch.setattr(mt, "_initialized", False)
    monkeypatch.setitem(VENDOR_METHODS["get_stock_data"], "metatrader", mt.get_stock_data)
    monkeypatch.setitem(
        VENDOR_METHODS["get_stock_data"], "yfinance",
        lambda *a, **k: "# yfinance served this\n",
    )
    set_config({"data_vendors": {"core_stock_apis": "metatrader,yfinance"}})

    out = route_to_vendor("get_stock_data", "EURUSD=X", "2026-08-01", "2026-08-31")
    assert out == "# yfinance served this\n"
    sys.modules.pop("tradingagents.dataflows.metatrader", None)
