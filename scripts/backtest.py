"""Walk-forward backtest for TradingAgents, with a null control.

What this measures
------------------
Not "did it make money" — in a trending market almost any long-biased signal
makes money. It measures whether the signal carries INFORMATION: does knowing
the rating change the distribution of the forward return, compared with not
knowing it?

Three numbers decide that, and they are printed together:

  conditional mean   the average forward return when the agent says Buy
  unconditional mean the average forward return over the same dates regardless
  buy & hold         what you'd have made doing nothing

A strategy whose conditional mean is not meaningfully above the unconditional
mean has learned nothing, however good its equity curve looks. The bootstrap
p-value quantifies "meaningfully".

Why a --null run matters
------------------------
``--null`` replaces the agent with a random rating drawn from the same
distribution the agent actually produced, and re-scores. Run it several times
and you get the spread of outcomes attributable to luck alone on YOUR dates and
YOUR instrument. If the real run sits inside that spread, the agent added
nothing. This is the single most important control, and it costs no LLM calls.

Honest limitations (read before trusting a result)
--------------------------------------------------
* Prices, indicators, FRED vintages and news are point-in-time safe. Social
  sentiment is NOT fully recoverable: the StockTwits and Reddit public feeds
  only serve recent items, so a historical run gets a placeholder and the
  Sentiment Analyst is weaker than it would be live. Expect a backtest to
  understate (or simply differ from) live behaviour.
* LLMs are non-deterministic. One run per date is a sample of one. Use enough
  dates that noise averages out, and read the confidence interval, not the mean.
* No spread, no swap/financing, no slippage, no leverage. On a leveraged CFD the
  realised result will be worse than anything printed here.
* The decision log learns across runs, so dates MUST be walked in chronological
  order (this script does). Re-running a date after later dates have resolved
  leaks information backwards — use --resume, never re-run a subset.

Usage
-----
    # local Ollama
    python scripts/backtest.py --ticker EURUSD --start 2026-01-01 --end 2026-06-30 \
        --provider ollama --quick-model qwen3:latest --deep-model gpt-oss:latest \
        --step 5 --holding-days 5 --analysts market,news --depth 1 \
        --out results/eurusd.csv

    # score-only re-analysis of an existing CSV (no LLM calls)
    python scripts/backtest.py --score results/eurusd.csv --holding-days 5

    # null control (no LLM calls)
    python scripts/backtest.py --score results/eurusd.csv --holding-days 5 --null 200
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("backtest")

# Rating -> position. Overweight/Underweight are half-size: the framework
# defines them as "gradually increase / reduce exposure", not full positions.
POSITION = {
    "Buy": 1.0,
    "Overweight": 0.5,
    "Hold": 0.0,
    "Underweight": -0.5,
    "Sell": -1.0,
    "REVIEW": 0.0,   # unparseable output is not a tradeable signal
}

FIELDS = ["date", "signal", "position", "forward_return", "strategy_return"]


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------

def load_prices(ticker: str, through: str) -> pd.DataFrame:
    """Daily bars through ``through``, from whichever vendor is configured.

    This is the SCORER, not an agent, so it is allowed to see dates after a
    decision — that is the whole point. It deliberately reuses the configured
    vendor so an MT5-backed run is scored on broker prices, not Yahoo's.
    """
    from tradingagents.dataflows.config import get_config

    vendors = get_config().get("data_vendors", {}).get("core_stock_apis", "")
    if "metatrader" in vendors:
        from tradingagents.dataflows.metatrader import load_mt5_ohlcv

        return load_mt5_ohlcv(ticker, through)

    from tradingagents.dataflows.stockstats_utils import load_ohlcv

    return load_ohlcv(ticker, through)


def decision_dates(prices: pd.DataFrame, start: str, end: str, step: int) -> list[str]:
    """Trading days between start and end, every ``step`` bars."""
    mask = (prices["Date"] >= pd.to_datetime(start)) & (prices["Date"] <= pd.to_datetime(end))
    days = prices.loc[mask, "Date"].dt.strftime("%Y-%m-%d").tolist()
    return days[::step]


def forward_return(prices: pd.DataFrame, date: str, holding_days: int) -> float | None:
    """Return from the close on ``date`` to the close ``holding_days`` bars later."""
    idx = prices.index[prices["Date"] == pd.to_datetime(date)]
    if len(idx) == 0:
        return None
    i = idx[0]
    if i + holding_days >= len(prices):
        return None  # window hasn't traded yet
    entry = float(prices["Close"].iloc[i])
    exit_ = float(prices["Close"].iloc[i + holding_days])
    return (exit_ - entry) / entry if entry else None


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(args) -> Path:
    from tradingagents.default_config import DEFAULT_CONFIG
    from tradingagents.graph.trading_graph import TradingAgentsGraph

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from cli.models import AnalystType
    from cli.utils import detect_asset_type, filter_analysts_for_asset_type

    asset_type = detect_asset_type(args.ticker)
    selected = [AnalystType(a.strip()) for a in args.analysts.split(",")]
    selected = filter_analysts_for_asset_type(selected, asset_type)
    analyst_keys = [a.value for a in selected]

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = args.depth
    config["max_risk_discuss_rounds"] = args.depth
    config["checkpoint_enabled"] = True

    # The interactive CLI asks for these; a script has to be told. Explicit
    # flags win, then TRADINGAGENTS_* env vars (already applied to
    # DEFAULT_CONFIG), then the built-in default.
    if args.provider:
        config["llm_provider"] = args.provider.lower()
    if args.quick_model:
        config["quick_think_llm"] = args.quick_model
    if args.deep_model:
        config["deep_think_llm"] = args.deep_model
    if args.backend_url:
        config["backend_url"] = args.backend_url

    prices = load_prices(args.ticker, datetime.now().strftime("%Y-%m-%d"))
    dates = decision_dates(prices, args.start, args.end, args.step)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists() and args.resume:
        with open(out, newline="", encoding="utf-8") as fh:
            done = {row["date"] for row in csv.DictReader(fh)}
        logger.info("Resuming: %d dates already recorded", len(done))

    todo = [d for d in dates if d not in done]
    print(
        f"{args.ticker} ({asset_type.value}) | {len(todo)} decisions to run "
        f"| analysts={analyst_keys} | depth={args.depth}\n"
        f"Provider: {config['llm_provider']} | quick={config['quick_think_llm']} "
        f"| deep={config['deep_think_llm']}\n"
        f"Rough cost: {len(todo)} runs x ~{12 + 5 * args.depth} LLM calls each.\n"
    )
    if len(dates) < 20:
        print(
            f"NOTE: {len(dates)} decisions is too few to conclude anything. The "
            f"confidence interval will span zero whatever the result. Widen "
            f"--start/--end or lower --step before reading the numbers as signal.\n"
        )
    if not todo:
        return score(out, args.ticker, args.holding_days, args.null)

    graph = TradingAgentsGraph(analyst_keys, config=config, debug=False)

    write_header = not out.exists() or not done
    with open(out, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()

        # Chronological order is required: the decision log resolves outcomes
        # and injects lessons forward in time. Walking out of order would let a
        # later date's outcome inform an earlier one.
        for n, date in enumerate(todo, 1):
            try:
                _, signal = graph.propagate(args.ticker, date, asset_type=asset_type.value)
            except Exception as exc:  # noqa: BLE001 — one bad date must not kill a 6h run
                logger.warning("%s failed (%s); recording as REVIEW", date, exc)
                signal = "REVIEW"

            writer.writerow({
                "date": date,
                "signal": signal,
                "position": POSITION.get(signal, 0.0),
                "forward_return": "",
                "strategy_return": "",
            })
            fh.flush()  # a 6-hour run must survive a crash
            print(f"[{n}/{len(todo)}] {date}  ->  {signal}")

    return score(out, args.ticker, args.holding_days, args.null)


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def _bootstrap_ci(values: np.ndarray, n: int = 10000, seed: int = 0):
    """Percentile bootstrap CI for the mean — no scipy dependency."""
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _null_distribution(positions: np.ndarray, forwards: np.ndarray, trials: int, seed: int = 0):
    """Mean strategy return when the SAME position mix is assigned at random.

    Preserves the agent's own long/short/flat balance, so the comparison isolates
    timing skill rather than directional bias. If the real result sits inside
    this distribution, the agent's timing added nothing.
    """
    rng = np.random.default_rng(seed)
    return np.array([
        float((rng.permutation(positions) * forwards).mean()) for _ in range(trials)
    ])


def score(path: Path, ticker: str, holding_days: int, null_trials: int = 0) -> Path:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("No rows to score.")
        return path

    prices = load_prices(ticker, datetime.now().strftime("%Y-%m-%d"))

    scored = []
    for row in rows:
        fwd = forward_return(prices, row["date"], holding_days)
        if fwd is None:
            continue
        pos = float(row["position"])
        row["forward_return"] = f"{fwd:.6f}"
        row["strategy_return"] = f"{pos * fwd:.6f}"
        scored.append(row)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(scored)

    if not scored:
        print("No decisions have a settled holding window yet.")
        return path

    positions = np.array([float(r["position"]) for r in scored])
    forwards = np.array([float(r["forward_return"]) for r in scored])
    strategy = positions * forwards

    print(f"\n{'=' * 62}\n  {ticker} — {len(scored)} decisions, {holding_days}-day holding\n{'=' * 62}")

    # Signal distribution first: a model that always says Buy is not a model.
    print("\nSignal distribution")
    counts: dict[str, int] = {}
    for r in scored:
        counts[r["signal"]] = counts.get(r["signal"], 0) + 1
    for sig, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        share = count / len(scored)
        flag = "  <-- dominant, signal may be constant" if share > 0.7 else ""
        print(f"  {sig:<12} {count:>4}  ({share:>5.1%}){flag}")

    print("\nMean forward return, conditional on the signal")
    print(f"  {'signal':<12} {'n':>4} {'mean':>9} {'vs uncond.':>12}")
    uncond = forwards.mean()
    for sig in ("Buy", "Overweight", "Hold", "Underweight", "Sell", "REVIEW"):
        sel = np.array([r["signal"] == sig for r in scored])
        if not sel.any():
            continue
        m = forwards[sel].mean()
        print(f"  {sig:<12} {sel.sum():>4} {m:>8.3%} {m - uncond:>+11.3%}")
    print(f"  {'(all dates)':<12} {len(scored):>4} {uncond:>8.3%}")

    lo, hi = _bootstrap_ci(strategy)
    hit = float((strategy > 0).mean())
    total = float(np.prod(1 + strategy) - 1)
    bh = float(np.prod(1 + forwards) - 1)

    print("\nStrategy vs doing nothing")
    print(f"  mean return per decision   {strategy.mean():>9.3%}   95% CI [{lo:.3%}, {hi:.3%}]")
    print(f"  hit rate                   {hit:>9.1%}")
    print(f"  compounded (strategy)      {total:>9.2%}")
    print(f"  compounded (buy & hold)    {bh:>9.2%}")
    if strategy.std(ddof=1) > 0:
        sharpe = strategy.mean() / strategy.std(ddof=1) * np.sqrt(252 / holding_days)
        print(f"  annualised Sharpe (approx) {sharpe:>9.2f}")

    if null_trials:
        null = _null_distribution(positions, forwards, null_trials)
        pct = float((null >= strategy.mean()).mean())
        print(f"\nNull control — same positions, shuffled dates, {null_trials} trials")
        print(f"  null mean                  {null.mean():>9.3%}")
        print(f"  null 95th percentile       {np.percentile(null, 95):>9.3%}")
        print(f"  actual                     {strategy.mean():>9.3%}")
        print(f"  p-value (luck alone)       {pct:>9.3f}")
        verdict = (
            "the timing beat chance on this sample" if pct < 0.05
            else "indistinguishable from random timing"
        )
        print(f"  verdict: {verdict}")

    print(
        "\nNo spread, swap or slippage is modelled. On a leveraged CFD the realised\n"
        "result will be worse. Treat this as a research signal, not a strategy.\n"
    )
    return path


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ticker")
    p.add_argument("--start")
    p.add_argument("--end")
    p.add_argument("--step", type=int, default=5, help="trading days between decisions")
    p.add_argument("--holding-days", type=int, default=5)
    p.add_argument("--analysts", default="market,social,news,fundamentals")
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--provider", help="ollama, deepseek, openai, ... (default: config/env)")
    p.add_argument("--quick-model", help="model for the ~15-40 analyst/debate calls")
    p.add_argument("--deep-model", help="model for the 2 manager calls")
    p.add_argument("--backend-url", help="override the provider endpoint")
    p.add_argument("--out", default="results/backtest.csv")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--score", help="score an existing CSV without running the agents")
    p.add_argument("--null", type=int, default=0, help="null-control trials (0 = skip)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    random.seed(0)

    if args.score:
        path = Path(args.score)
        ticker = args.ticker or path.stem.split("_")[0].upper()
        return score(path, ticker, args.holding_days, args.null)

    if not (args.ticker and args.start and args.end):
        p.error("--ticker, --start and --end are required unless --score is given")
    return run(args)


if __name__ == "__main__":
    main()
