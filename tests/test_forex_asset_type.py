"""Non-equity instruments must not be analysed as if they were listed companies.

Before the forex/commodity/index asset types existed, everything that wasn't
crypto fell through to STOCK. Two consequences, both pinned here:

1. The Fundamentals Analyst ran against a currency pair, spending ~4 LLM calls
   to reach the NO_DATA sentinel and then feeding "fundamentals unavailable"
   into every downstream debate prompt.
2. The realised outcome was benchmarked against SPY, so the decision log stored
   a meaningless "alpha" and re-injected it into later runs as a lesson.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from cli.models import NON_EQUITY_ASSET_TYPES, AnalystType, AssetType
from cli.utils import detect_asset_type, filter_analysts_for_asset_type

ALL_ANALYSTS = [
    AnalystType.MARKET,
    AnalystType.SOCIAL,
    AnalystType.NEWS,
    AnalystType.FUNDAMENTALS,
]


# --- classification -------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("EURUSD", AssetType.FOREX),
    ("GBPJPY", AssetType.FOREX),
    ("EURUSD=X", AssetType.FOREX),
    ("XAUUSD", AssetType.COMMODITY),
    ("USOIL", AssetType.COMMODITY),
    ("GC=F", AssetType.COMMODITY),
    ("US500", AssetType.INDEX),
    ("GER40", AssetType.INDEX),
    ("^GSPC", AssetType.INDEX),
    ("BTC-USD", AssetType.CRYPTO),
    ("AAPL", AssetType.STOCK),
    ("0700.HK", AssetType.STOCK),
])
def test_classification(raw, expected):
    assert detect_asset_type(raw) == expected


# --- analyst filtering ----------------------------------------------------

@pytest.mark.parametrize("asset_type", sorted(NON_EQUITY_ASSET_TYPES, key=str))
def test_fundamentals_dropped_for_every_non_equity_type(asset_type):
    kept = filter_analysts_for_asset_type(ALL_ANALYSTS, asset_type)
    assert AnalystType.FUNDAMENTALS not in kept
    assert AnalystType.MARKET in kept and AnalystType.NEWS in kept


def test_equities_keep_every_analyst():
    assert filter_analysts_for_asset_type(ALL_ANALYSTS, AssetType.STOCK) == ALL_ANALYSTS


def test_forex_ticker_end_to_end_drops_fundamentals():
    """The path the CLI actually takes: ticker string -> analyst list."""
    kept = filter_analysts_for_asset_type(ALL_ANALYSTS, detect_asset_type("EURUSD"))
    assert [a.value for a in kept] == ["market", "social", "news"]


# --- benchmark resolution -------------------------------------------------

def _graph():
    """A TradingAgentsGraph with the constructor's LLM/graph wiring stubbed out."""
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    with patch.object(TradingAgentsGraph, "__init__", lambda self, *a, **k: None):
        graph = TradingAgentsGraph()
    graph.config = DEFAULT_CONFIG.copy()
    return graph


@pytest.mark.parametrize("ticker", [
    "EURUSD=X", "GBPJPY=X", "GC=F", "CL=F", "^GSPC", "^N225", "BTC-USD",
])
def test_non_equity_has_no_benchmark(ticker):
    assert _graph()._resolve_benchmark(ticker) is None


@pytest.mark.parametrize("ticker,expected", [
    ("AAPL", "SPY"),
    ("BRK.B", "SPY"),
    ("7203.T", "^N225"),
    ("0700.HK", "^HSI"),
    ("RELIANCE.NS", "^NSEI"),
])
def test_equities_keep_their_regional_benchmark(ticker, expected):
    assert _graph()._resolve_benchmark(ticker) == expected


def test_explicit_benchmark_still_overrides_everything():
    graph = _graph()
    graph.config["benchmark_ticker"] = "DX-Y.NYB"
    assert graph._resolve_benchmark("EURUSD=X") == "DX-Y.NYB"


# --- outcome resolution with no benchmark ---------------------------------

def _price_history(n=12, start=1.08, step=0.002):
    idx = pd.date_range("2026-08-03", periods=n, freq="B")
    return pd.DataFrame({"Close": [start + i * step for i in range(n)]}, index=idx)


def test_fetch_returns_skips_benchmark_lookup_when_none():
    """No benchmark means no second yfinance call, and alpha comes back None."""
    graph = _graph()
    with patch("yfinance.Ticker") as ticker_cls:
        ticker_cls.return_value.history.return_value = _price_history()
        raw, alpha, days, resolved = graph._fetch_returns(
            "EURUSD=X", "2026-08-03", holding_days=5, benchmark=None
        )
    assert raw is not None and alpha is None
    assert days == 5 and resolved is not None
    assert ticker_cls.call_count == 1  # the instrument only, never a benchmark


def test_fetch_returns_still_computes_alpha_for_equities():
    graph = _graph()
    with patch("yfinance.Ticker") as ticker_cls:
        ticker_cls.return_value.history.return_value = _price_history()
        raw, alpha, _, _ = graph._fetch_returns(
            "AAPL", "2026-08-03", holding_days=5, benchmark="SPY"
        )
    assert raw is not None and alpha is not None
    assert ticker_cls.call_count == 2


# --- memory log and reflection with no alpha ------------------------------

def test_memory_log_records_na_alpha_without_crashing(tmp_path):
    from tradingagents.agents.utils.memory import TradingMemoryLog

    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "log.md")})
    log.store_decision("EURUSD=X", "2026-08-03", "**Rating**: Buy\n\nLong EUR.")
    log.batch_update_with_outcomes([{
        "ticker": "EURUSD=X", "trade_date": "2026-08-03",
        "raw_return": 0.012, "alpha_return": None, "holding_days": 5,
        "reflection": "Call was right on the raw move.",
        "resolution_date": "2026-08-10",
    }])

    entry = log.load_entries()[0]
    assert entry["pending"] is False
    assert entry["raw"] == "+1.2%"
    assert entry["alpha"] == "n/a"
    assert "Call was right" in entry["reflection"]


def test_reflection_prompt_omits_alpha_line_when_no_benchmark():
    from tradingagents.graph.reflection import Reflector

    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="ok")
    Reflector(llm).reflect_on_final_decision(
        final_decision="**Rating**: Buy", raw_return=0.012,
        alpha_return=None, benchmark_name=None,
    )
    human = llm.invoke.call_args[0][0][1][1]
    assert "Raw return: +1.2%" in human
    assert "Alpha" not in human
    assert "do not cite an alpha figure" in human


def test_reflection_prompt_keeps_alpha_line_for_equities():
    from tradingagents.graph.reflection import Reflector

    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content="ok")
    Reflector(llm).reflect_on_final_decision(
        final_decision="**Rating**: Buy", raw_return=0.012,
        alpha_return=0.004, benchmark_name="SPY",
    )
    human = llm.invoke.call_args[0][0][1][1]
    assert "Alpha vs SPY: +0.4%" in human
