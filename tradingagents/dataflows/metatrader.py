"""MetaTrader 5 market-data vendor for forex, CFDs, metals, indices and equities.

Why this exists
---------------
The yfinance vendor prices the *underlying* instrument: ``US500`` resolves to
``^GSPC``, ``XAUUSD`` to the COMEX future ``GC=F``. That is fine for direction
but it is not what a broker actually quotes you, and it carries no spread. This
vendor reads bars straight from a running MetaTrader 5 terminal, so the agents
see the same instrument and pricing the account trades.

Contract
--------
Implements the two ``core_stock_apis`` / ``technical_indicators`` router methods:

    get_stock_data(symbol, start_date, end_date) -> str
    get_indicators(symbol, indicator, curr_date, look_back_days) -> str

Both mirror the yfinance implementations byte-for-byte in output shape, so the
analyst prompts, the report writers and the no-data sentinel all keep working.

Error behaviour is deliberate, because the router reacts to exception *type*:

* ``VendorNotConfiguredError`` — the ``MetaTrader5`` package is missing or the
  terminal will not initialize. The router logs it and moves to the next vendor,
  so ``data_vendors={"core_stock_apis": "metatrader,yfinance"}`` degrades to
  Yahoo automatically on a machine with no terminal (e.g. CI, or Linux).
* ``NoMarketDataError`` — the broker does not list the symbol, or returned no
  bars. The router turns this into the single ``NO_DATA_AVAILABLE`` sentinel
  that tells the model not to fabricate a price.

Requirements
------------
``pip install MetaTrader5`` (Windows-only) and a running, logged-in MT5 terminal.
On Linux run the terminal under Wine, or keep ``yfinance`` in the vendor chain.
"""

from __future__ import annotations

import functools
import logging
import re
import threading
from datetime import datetime, timedelta, timezone

import pandas as pd
from stockstats import wrap

from .errors import NoMarketDataError, VendorNotConfiguredError
from .stockstats_utils import _assert_ohlcv_not_stale, _clean_dataframe, _fill_price_gaps

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Configuration knobs
# --------------------------------------------------------------------------

# Daily bars. The framework works in dates, not intraday timestamps.
MT5_TIMEFRAME = "TIMEFRAME_D1"

# How much history to pull per symbol. Matches the yfinance path's 5-year window
# so the 200 SMA and other long indicators have enough warm-up.
MT5_HISTORY_YEARS = 5

# Bars are stamped in the broker's *server* time (commonly UTC+2/+3), so a D1
# bar's date can sit one day off a UTC calendar date at the boundary. We pad the
# request on both sides and then filter on the bar's own date, which is the
# behaviour a trader expects: "the bar the broker labels 2026-08-29".
_REQUEST_PAD_DAYS = 5

# Broker symbol suffixes seen in the wild: EURUSD.a, EURUSDm, XAUUSD+, US500.cash
_SUFFIX_CANDIDATES = ("", ".a", ".r", ".m", "m", "+", ".cash", ".spot", "-ECN", ".pro", ".raw")


# --------------------------------------------------------------------------
# Terminal lifecycle
# --------------------------------------------------------------------------

_init_lock = threading.Lock()
_initialized = False


def _mt5():
    """Import and initialize the terminal once; raise VendorNotConfiguredError otherwise.

    Kept lazy so merely importing this module (test collection, a Linux box with
    no terminal) never fails — the vendor simply reports itself unavailable and
    the router falls through to the next one in the chain.
    """
    global _initialized
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise VendorNotConfiguredError(
            "The 'MetaTrader5' package is not installed. Install it with "
            "`pip install MetaTrader5` (Windows only) and make sure a MetaTrader 5 "
            "terminal is running and logged in."
        ) from exc

    if _initialized:
        return mt5

    with _init_lock:
        if _initialized:
            return mt5
        if not mt5.initialize():
            raise VendorNotConfiguredError(
                f"Could not connect to a MetaTrader 5 terminal: {mt5.last_error()}. "
                f"Start the terminal, log in to an account, and enable "
                f"'Allow automated trading' in Tools > Options > Expert Advisors."
            )
        _initialized = True
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        logger.info(
            "MetaTrader 5 connected: %s (account %s)",
            getattr(terminal, "company", "unknown"),
            getattr(account, "login", "unknown"),
        )
    return mt5


def shutdown() -> None:
    """Close the terminal connection. Optional; the process exit also releases it."""
    global _initialized
    if not _initialized:
        return
    try:
        import MetaTrader5 as mt5

        mt5.shutdown()
    finally:
        _initialized = False


# --------------------------------------------------------------------------
# Symbol resolution: Yahoo canonical -> broker symbol
# --------------------------------------------------------------------------

# The pipeline normalizes user input to *Yahoo* symbols before it reaches a
# vendor (``EURUSD`` -> ``EURUSD=X``, ``XAUUSD`` -> ``GC=F``, ``US500`` ->
# ``^GSPC``). MT5 brokers use the original trading names, so we invert that.
# Values are ordered preference lists — the first one the broker actually lists
# wins.
_YAHOO_TO_MT5: dict[str, tuple[str, ...]] = {
    # Metals: back from the Yahoo future to the spot CFD brokers quote.
    "GC=F": ("XAUUSD", "GOLD", "XAUUSD.spot"),
    "SI=F": ("XAGUSD", "SILVER"),
    "PL=F": ("XPTUSD", "PLATINUM"),
    "PA=F": ("XPDUSD", "PALLADIUM"),
    "HG=F": ("XCUUSD", "COPPER"),
    # Energy
    "CL=F": ("XTIUSD", "USOIL", "WTI", "CRUDOIL"),
    "BZ=F": ("XBRUSD", "UKOIL", "BRENT"),
    "NG=F": ("XNGUSD", "NATGAS"),
    # Index CFDs
    "^GSPC": ("US500", "SPX500", "SP500", "US500.cash"),
    "^NDX": ("US100", "NAS100", "USTEC", "NDX100"),
    "^DJI": ("US30", "DJI30", "WS30", "DOW"),
    "^GDAXI": ("GER40", "DE40", "GER30", "DAX40"),
    "^FTSE": ("UK100", "FTSE100"),
    "^N225": ("JP225", "JPN225", "NIKKEI"),
    "^FCHI": ("FRA40", "CAC40"),
    "^STOXX50E": ("EU50", "STOXX50"),
    "^HSI": ("HK50", "HSI"),
}

_FOREX_YAHOO = re.compile(r"^([A-Z]{6})=X$")
_CRYPTO_YAHOO = re.compile(r"^([A-Z0-9]{2,10})-USD$")


def _candidate_names(symbol: str) -> list[str]:
    """Preference-ordered broker names to try for a pipeline symbol."""
    s = (symbol or "").strip().upper()
    if not s:
        return []

    if s in _YAHOO_TO_MT5:
        base = list(_YAHOO_TO_MT5[s])
    elif (m := _FOREX_YAHOO.match(s)):
        base = [m.group(1)]                      # EURUSD=X -> EURUSD
    elif (m := _CRYPTO_YAHOO.match(s)):
        base = [f"{m.group(1)}USD", m.group(1)]  # BTC-USD  -> BTCUSD
    else:
        base = [s]                               # equities pass through: AAPL

    # Always keep the raw input as a last resort — some brokers list ``EURUSD=X``
    # style names verbatim, and equities may already match exactly.
    if s not in base:
        base.append(s)
    return base


@functools.lru_cache(maxsize=512)
def resolve_mt5_symbol(symbol: str) -> str:
    """Map a pipeline symbol to a symbol this broker actually lists.

    Tries the preference list, then each name with the common broker suffixes
    (``EURUSD.a``, ``EURUSDm``, ``XAUUSD+``, ...), then falls back to a prefix
    scan of the broker's full symbol table. Selects the winner into MarketWatch,
    which ``copy_rates_range`` requires — an unselected symbol returns no bars
    even when it exists.

    Raises ``NoMarketDataError`` when the broker lists nothing that matches.
    """
    mt5 = _mt5()
    candidates = _candidate_names(symbol)

    for name in candidates:
        for suffix in _SUFFIX_CANDIDATES:
            trial = f"{name}{suffix}"
            if mt5.symbol_info(trial) is not None:
                if not mt5.symbol_select(trial, True):
                    logger.warning("Could not select %s in MarketWatch", trial)
                    continue
                if trial != symbol:
                    logger.info("Resolved %r to broker symbol %r", symbol, trial)
                return trial

    # Last resort: scan the broker's table for anything starting with a candidate.
    try:
        available = mt5.symbols_get() or ()
    except Exception:  # noqa: BLE001 — a scan failure must not mask the no-data verdict
        available = ()
    for name in candidates:
        for info in available:
            if info.name.upper().startswith(name) and mt5.symbol_select(info.name, True):
                logger.info("Resolved %r to broker symbol %r (prefix scan)", symbol, info.name)
                return info.name

    raise NoMarketDataError(
        symbol,
        candidates[0],
        f"broker lists no symbol matching any of {candidates}",
    )


# --------------------------------------------------------------------------
# Bar loading
# --------------------------------------------------------------------------

_frame_cache: dict[tuple[str, str], pd.DataFrame] = {}


def load_mt5_ohlcv(symbol: str, curr_date: str, use_cache: bool = True) -> pd.DataFrame:
    """Daily OHLCV up to and including ``curr_date``, look-ahead filtered.

    Returns a frame with the same capitalized columns the yfinance path produces
    (``Date, Open, High, Low, Close, Volume``) so ``stockstats`` and the verified
    snapshot builder consume it unchanged.

    Rows after ``curr_date`` are dropped before anything else touches the frame —
    this is the look-ahead guard, and it is the reason we fetch a wide window once
    and slice per call rather than requesting exactly the range asked for.
    """
    mt5 = _mt5()
    broker_symbol = resolve_mt5_symbol(symbol)
    curr_dt = pd.to_datetime(curr_date).normalize()

    key = (broker_symbol, MT5_TIMEFRAME)
    frame = _frame_cache.get(key) if use_cache else None

    if frame is None:
        timeframe = getattr(mt5, MT5_TIMEFRAME)
        # Pad both ends: the far end covers server-time drift at the boundary,
        # the near end covers the warm-up needed by the 200 SMA.
        utc_to = datetime.now(timezone.utc) + timedelta(days=_REQUEST_PAD_DAYS)
        utc_from = utc_to - timedelta(days=365 * MT5_HISTORY_YEARS + _REQUEST_PAD_DAYS)

        rates = mt5.copy_rates_range(broker_symbol, timeframe, utc_from, utc_to)
        if rates is None or len(rates) == 0:
            raise NoMarketDataError(
                symbol,
                broker_symbol,
                f"MetaTrader 5 returned no bars ({mt5.last_error()})",
            )

        frame = pd.DataFrame(rates)
        frame["Date"] = pd.to_datetime(frame["time"], unit="s")
        # MT5 reports tick_volume for CFDs/forex (real_volume is 0 on most
        # brokers), so prefer real volume when the broker publishes it.
        volume = frame.get("real_volume")
        if volume is None or (volume == 0).all():
            volume = frame.get("tick_volume", 0)
        frame = pd.DataFrame(
            {
                "Date": frame["Date"],
                "Open": frame["open"],
                "High": frame["high"],
                "Low": frame["low"],
                "Close": frame["close"],
                "Volume": volume,
            }
        )
        if use_cache:
            _frame_cache[key] = frame

    data = _clean_dataframe(frame.copy())

    # Look-ahead cut: nothing after the analysis date reaches an agent.
    data = data[data["Date"] <= curr_dt]
    if data.empty:
        raise NoMarketDataError(
            symbol, broker_symbol, f"no bars on or before {curr_date}"
        )

    # A newest bar with no close is "not settled yet", not "does not exist" —
    # same guard the yfinance path applies (#1201).
    if pd.isna(data["Close"].iloc[-1]):
        raise NoMarketDataError(
            symbol, broker_symbol, "latest in-range bar has no closing price"
        )

    data = _fill_price_gaps(data)
    _assert_ohlcv_not_stale(data, curr_date, symbol, broker_symbol)
    return data.reset_index(drop=True)


def clear_cache() -> None:
    """Drop the in-process bar cache (use after the terminal reconnects)."""
    _frame_cache.clear()
    resolve_mt5_symbol.cache_clear()


# --------------------------------------------------------------------------
# Router method 1: OHLCV
# --------------------------------------------------------------------------


def get_stock_data(symbol: str, start_date: str, end_date: str) -> str:
    """Return OHLCV between two dates as a CSV block with a header comment.

    Output shape matches ``y_finance.get_YFin_data_online`` so the market
    analyst's prompt and the report writers need no changes.
    """
    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    broker_symbol = resolve_mt5_symbol(symbol)
    data = load_mt5_ohlcv(symbol, end_date)

    window = data[data["Date"] >= pd.to_datetime(start_date).normalize()]
    if window.empty:
        raise NoMarketDataError(
            symbol, broker_symbol, f"no bars between {start_date} and {end_date}"
        )

    window = window.copy()
    for col in ("Open", "High", "Low", "Close"):
        if col in window.columns:
            window[col] = window[col].round(5)
    window["Date"] = window["Date"].dt.strftime("%Y-%m-%d")

    label = broker_symbol if broker_symbol == symbol.upper() else f"{broker_symbol} (from {symbol})"
    header = (
        f"# MetaTrader 5 daily data for {label} from {start_date} to {end_date}\n"
        f"# Total records: {len(window)}\n"
        f"# Source: broker feed (prices are the broker's own quotes)\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + window.to_csv(index=False)


# --------------------------------------------------------------------------
# Router method 2: technical indicators
# --------------------------------------------------------------------------

# Same keys and wording as the yfinance path — the market analyst's system prompt
# lists these exact names and its tool call fails on anything else.
_INDICATOR_DESCRIPTIONS = {
    "close_50_sma": (
        "50 SMA: A medium-term trend indicator. Usage: Identify trend direction and "
        "serve as dynamic support/resistance. Tips: It lags price; combine with faster "
        "indicators for timely signals."
    ),
    "close_200_sma": (
        "200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and "
        "identify golden/death cross setups. Tips: It reacts slowly; best for strategic "
        "trend confirmation rather than frequent trading entries."
    ),
    "close_10_ema": (
        "10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum "
        "and potential entry points. Tips: Prone to noise in choppy markets; use alongside "
        "longer averages for filtering false signals."
    ),
    "macd": (
        "MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and "
        "divergence as signals of trend changes. Tips: Confirm with other indicators in "
        "low-volatility or sideways markets."
    ),
    "macds": (
        "MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the "
        "MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid "
        "false positives."
    ),
    "macdh": (
        "MACD Histogram: Shows the gap between the MACD line and its signal. Usage: "
        "Visualize momentum strength and spot divergence early. Tips: Can be volatile; "
        "complement with additional filters in fast-moving markets."
    ),
    "rsi": (
        "RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 "
        "thresholds and watch for divergence to signal reversals. Tips: In strong trends, "
        "RSI may remain extreme; always cross-check with trend analysis."
    ),
    "boll": (
        "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts "
        "as a dynamic benchmark for price movement. Tips: Combine with the upper and lower "
        "bands to effectively spot breakouts or reversals."
    ),
    "boll_ub": (
        "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
        "Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm "
        "signals with other tools; prices may ride the band in strong trends."
    ),
    "boll_lb": (
        "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
        "Usage: Indicates potential oversold conditions. Tips: Use additional analysis to "
        "avoid false reversal signals."
    ),
    "atr": (
        "ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and "
        "adjust position sizes based on current market volatility. Tips: It's a reactive "
        "measure, so use it as part of a broader risk management strategy."
    ),
    "vwma": (
        "VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating "
        "price action with volume data. Tips: On forex and CFDs the broker reports tick "
        "volume, not traded size, so treat this as an activity proxy rather than real flow."
    ),
    "mfi": (
        "MFI: The Money Flow Index uses both price and volume to measure buying and selling "
        "pressure. Usage: Identify overbought (>80) or oversold (<20) conditions. Tips: On "
        "forex and CFDs this is computed from tick volume, so read it as activity, not flow."
    ),
}


def get_indicators(
    symbol: str,
    indicator: str,
    curr_date: str,
    look_back_days: int = 30,
) -> str:
    """Return one indicator's value per calendar day over the look-back window.

    Indicators are computed locally with ``stockstats`` — exactly what the
    yfinance path does — so switching vendors changes the *prices*, never the
    indicator maths. MT5's Python API exposes no indicator functions, and
    matching the existing semantics matters more than using the terminal's.
    """
    if indicator not in _INDICATOR_DESCRIPTIONS:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: "
            f"{list(_INDICATOR_DESCRIPTIONS.keys())}"
        )

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_dt - timedelta(days=look_back_days)

    data = load_mt5_ohlcv(symbol, curr_date)
    stock_df = wrap(data.copy())
    stock_df[indicator]  # triggers the stockstats calculation

    by_date: dict[str, str] = {}
    for _, row in stock_df.iterrows():
        stamp = row["date"] if "date" in row else row.get("Date")
        if pd.isna(stamp):
            continue
        value = row[indicator]
        by_date[pd.Timestamp(stamp).strftime("%Y-%m-%d")] = (
            "N/A" if pd.isna(value) else str(value)
        )

    lines = []
    cursor = curr_dt
    while cursor >= before:
        key = cursor.strftime("%Y-%m-%d")
        lines.append(f"{key}: {by_date.get(key, 'N/A: Not a trading day (weekend or holiday)')}")
        cursor -= timedelta(days=1)

    return (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        + "\n".join(lines)
        + "\n\n\n"
        + _INDICATOR_DESCRIPTIONS[indicator]
    )
