"""Unit tests for the scripts/backtest.py harness.

Everything here operates on plain dicts/lists — no TradingAgentsGraph
construction, no LLM calls, no network — mirroring the mocking style in
tests/test_memory_pointintime.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest import (  # noqa: E402
    already_logged_dates,
    apply_llm_overrides,
    build_trade_dates,
    parse_analysts,
    render_table,
    split_run_entries,
    summarize,
    write_markdown_report,
)


@pytest.mark.unit
def test_build_trade_dates_steps_by_cadence():
    assert build_trade_dates("2025-01-01", "2025-01-10", 5) == [
        "2025-01-01", "2025-01-06",
    ]


@pytest.mark.unit
def test_build_trade_dates_inclusive_of_end_when_it_lands_exactly():
    assert build_trade_dates("2025-01-01", "2025-01-11", 5) == [
        "2025-01-01", "2025-01-06", "2025-01-11",
    ]


@pytest.mark.unit
def test_build_trade_dates_rejects_end_before_start():
    with pytest.raises(ValueError):
        build_trade_dates("2025-01-10", "2025-01-01", 5)


@pytest.mark.unit
def test_build_trade_dates_rejects_non_positive_cadence():
    with pytest.raises(ValueError):
        build_trade_dates("2025-01-01", "2025-01-10", 0)


@pytest.mark.unit
def test_already_logged_dates_filters_by_ticker():
    entries = [
        {"ticker": "NVDA", "date": "2025-01-01"},
        {"ticker": "AAPL", "date": "2025-01-01"},
        {"ticker": "NVDA", "date": "2025-01-06"},
    ]
    assert already_logged_dates(entries, "NVDA") == {"2025-01-01", "2025-01-06"}
    assert already_logged_dates(entries, "AAPL") == {"2025-01-01"}
    assert already_logged_dates(entries, "MSFT") == set()


def _entry(ticker, date, rating, raw, alpha, pending=False):
    return {
        "ticker": ticker, "date": date, "rating": rating,
        "raw": raw, "alpha": alpha, "pending": pending,
    }


@pytest.mark.unit
def test_split_run_entries_separates_pending_from_resolved():
    entries = [
        _entry("NVDA", "2025-01-01", "Buy", "+2.0%", "+1.0%"),
        _entry("NVDA", "2025-01-06", "Hold", None, None, pending=True),
        _entry("AAPL", "2025-01-01", "Sell", "-1.0%", "-0.5%"),  # not requested
    ]
    requested = [("NVDA", "2025-01-01"), ("NVDA", "2025-01-06")]
    resolved, pending = split_run_entries(entries, requested)
    assert [e["date"] for e in resolved] == ["2025-01-01"]
    assert [e["date"] for e in pending] == ["2025-01-06"]


@pytest.mark.unit
def test_summarize_computes_win_rate_and_averages_overall_and_by_rating():
    resolved = [
        _entry("NVDA", "2025-01-01", "Buy", "+4.0%", "+2.0%"),
        _entry("NVDA", "2025-01-06", "Buy", "-2.0%", "-1.0%"),
        _entry("NVDA", "2025-01-11", "Hold", "+1.0%", "+0.5%"),
    ]
    stats = summarize(resolved)

    overall = stats["overall"]
    assert overall["count"] == 3
    assert overall["win_rate"] == pytest.approx(2 / 3)
    assert overall["avg_raw"] == pytest.approx((0.04 - 0.02 + 0.01) / 3)
    assert overall["avg_alpha"] == pytest.approx((0.02 - 0.01 + 0.005) / 3)

    buy = stats["by_rating"]["Buy"]
    assert buy["count"] == 2
    assert buy["win_rate"] == pytest.approx(0.5)
    assert buy["avg_alpha"] == pytest.approx((0.02 - 0.01) / 2)

    hold = stats["by_rating"]["Hold"]
    assert hold["count"] == 1
    assert hold["win_rate"] == 1.0


@pytest.mark.unit
def test_summarize_empty_input_reports_none_not_a_crash():
    stats = summarize([])
    assert stats["overall"]["count"] == 0
    assert stats["overall"]["win_rate"] is None
    assert stats["by_rating"] == {}


@pytest.mark.unit
def test_render_table_notes_pending_count_in_caption():
    resolved = [_entry("NVDA", "2025-01-01", "Buy", "+1.0%", "+0.5%")]
    stats = summarize(resolved)
    table = render_table(stats, pending_count=2)
    assert "2 decision(s) still pending" in table.caption


@pytest.mark.unit
def test_render_table_zero_pending_has_no_caption():
    stats = summarize([_entry("NVDA", "2025-01-01", "Buy", "+1.0%", "+0.5%")])
    table = render_table(stats, pending_count=0)
    assert table.caption is None


@pytest.mark.unit
def test_write_markdown_report_includes_win_rate_and_per_rating_breakdown(tmp_path):
    resolved = [
        _entry("NVDA", "2025-01-01", "Buy", "+4.0%", "+2.0%"),
        _entry("NVDA", "2025-01-06", "Hold", "+1.0%", "-0.5%"),
    ]
    stats = summarize(resolved)
    out = tmp_path / "summary.md"
    write_markdown_report(stats, pending_count=1, path=out)

    text = out.read_text()
    assert "Decisions: 2 (1 still pending)" in text
    assert "Win rate (alpha>0): 50%" in text
    assert "Buy (1): avg alpha +2.0%" in text
    assert "Hold (1): avg alpha -0.5%" in text


@pytest.mark.unit
def test_apply_llm_overrides_provider_only_uses_catalog_default_models():
    config = {"llm_provider": "openai", "deep_think_llm": "gpt-5.6", "quick_think_llm": "gpt-5.6-luna"}
    apply_llm_overrides(config, "ollama", None, None, None)
    assert config["llm_provider"] == "ollama"
    assert config["deep_think_llm"] == "glm-4.7-flash:latest"
    assert config["quick_think_llm"] == "qwen3:latest"


@pytest.mark.unit
def test_apply_llm_overrides_explicit_model_wins_over_catalog_default():
    config = {"llm_provider": "openai", "deep_think_llm": "x", "quick_think_llm": "y"}
    apply_llm_overrides(config, "ollama", "custom-deep:latest", None, None)
    assert config["deep_think_llm"] == "custom-deep:latest"
    assert config["quick_think_llm"] == "qwen3:latest"


@pytest.mark.unit
def test_apply_llm_overrides_backend_url_without_provider():
    config = {"llm_provider": "openai", "backend_url": None}
    apply_llm_overrides(config, None, None, None, "http://localhost:11434/v1")
    assert config["llm_provider"] == "openai"
    assert config["backend_url"] == "http://localhost:11434/v1"


@pytest.mark.unit
def test_apply_llm_overrides_no_args_leaves_config_untouched():
    config = {"llm_provider": "openai", "deep_think_llm": "gpt-5.6", "quick_think_llm": "gpt-5.6-luna",
              "backend_url": None}
    original = dict(config)
    apply_llm_overrides(config, None, None, None, None)
    assert config == original


@pytest.mark.unit
def test_apply_llm_overrides_rejects_custom_only_provider_without_model():
    config = {"llm_provider": "openai"}
    with pytest.raises(ValueError, match="deep-model"):
        apply_llm_overrides(config, "bedrock", None, None, None)


@pytest.mark.unit
def test_parse_analysts_none_returns_all_four():
    assert parse_analysts(None) == ("market", "social", "news", "fundamentals")


@pytest.mark.unit
def test_parse_analysts_drops_fundamentals_for_forex():
    assert parse_analysts("market,social,news") == ("market", "social", "news")


@pytest.mark.unit
def test_parse_analysts_strips_whitespace():
    assert parse_analysts(" market , news ") == ("market", "news")


@pytest.mark.unit
def test_parse_analysts_rejects_unknown_analyst():
    with pytest.raises(ValueError, match="fundamental"):
        parse_analysts("market,fundamental")  # missing trailing 's'


@pytest.mark.unit
def test_parse_analysts_rejects_empty_string():
    with pytest.raises(ValueError, match="empty"):
        parse_analysts("")


@pytest.mark.unit
def test_write_markdown_report_handles_no_resolved_decisions(tmp_path):
    stats = summarize([])
    out = tmp_path / "summary.md"
    write_markdown_report(stats, pending_count=3, path=out)
    assert "No resolved decisions yet (3 pending)" in out.read_text()
