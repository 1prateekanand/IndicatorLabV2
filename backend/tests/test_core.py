"""Core correctness: registry, causality, execution timing, costs, metrics, data validation."""
import os

import numpy as np
import pandas as pd
import pytest

from indicatorlab.config import BacktestConfig, CostModel
from indicatorlab.data import clean_candles, import_csv
from indicatorlab.engine import Exit, run_backtest
from indicatorlab.metrics import compute_metrics
from indicatorlab.registry import REGISTRY, ParamSpec
from indicatorlab.strategy import StrategySpec, spec_from_dict


def _frame(opens, highs, lows, closes, start="2025-01-06 09:15"):
    idx = pd.date_range(start, periods=len(closes), freq="min")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes,
                         "volume": 1.0}, index=idx)


def _flat_bars(n, price=100.0):
    return _frame([price] * n, [price + 0.5] * n, [price - 0.5] * n, [price] * n)


# -- registry ----------------------------------------------------------------------------------
def test_builtin_indicators_discovered_and_self_tested(tmp_root):
    summary = REGISTRY.discover(plugin_dir=os.path.join(tmp_root, "plugins"))
    assert summary["total"] >= 7
    assert summary["available"] == summary["total"]
    assert all(d.status_reason == "ok" for d in REGISTRY.available())


def test_plugin_file_is_discovered_without_code_changes(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "my_indicator.py").write_text(
        "import pandas as pd\n"
        "from indicatorlab.registry import indicator, ParamSpec\n"
        "@indicator('user_always_long', category='Custom', params=(ParamSpec('x', 1, 1, 3),))\n"
        "def f(df, x=1):\n"
        "    return pd.Series(1, index=df.index)\n")
    REGISTRY.discover(plugin_dir=plugin_dir)
    assert "user_always_long" in REGISTRY.names(only_available=True)
    REGISTRY.discover(plugin_dir=tmp_path / "empty")  # restore a clean registry for other tests
    assert "user_always_long" not in REGISTRY.names()


def test_broken_plugin_is_reported_not_fatal(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "broken.py").write_text("raise RuntimeError('boom')\n")
    summary = REGISTRY.discover(plugin_dir=plugin_dir)
    assert any("boom" in e["error"] for e in summary["load_errors"])
    assert summary["available"] >= 7
    REGISTRY.discover(plugin_dir=tmp_path / "none")


def test_indicator_failing_self_test_is_marked_unavailable(tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "bad_signal.py").write_text(
        "import pandas as pd\n"
        "from indicatorlab.registry import indicator\n"
        "@indicator('bad_signal', category='Custom')\n"
        "def f(df):\n"
        "    return pd.Series(5, index=df.index)\n")
    REGISTRY.discover(plugin_dir=plugin_dir)
    bad = REGISTRY.get("bad_signal")
    assert bad.available is False
    assert "signal values" in bad.status_reason
    with pytest.raises(ValueError, match="unavailable"):
        StrategySpec.build([{"name": "bad_signal"}])
    REGISTRY.discover(plugin_dir=tmp_path / "none")


def test_unknown_parameter_rejected_and_values_clamped():
    d = REGISTRY.get("ref_rsi")
    with pytest.raises(ValueError, match="no parameter"):
        d.clean_params({"nope": 1})
    assert d.clean_params({"period": 999})["period"] == d.param("period").high


# -- causality: no indicator may use future bars ------------------------------------------------
def test_every_builtin_indicator_is_causal(tmp_root):
    REGISTRY.discover(plugin_dir=os.path.join(tmp_root, "plugins"))
    rng = np.random.default_rng(11)
    n = 600
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    df = _frame(close, close + rng.uniform(0.1, 1, n), close - rng.uniform(0.1, 1, n), close)
    cut = 400
    for d in REGISTRY.available():
        full = d.compute(df, **d.defaults).to_numpy()
        prefix = d.compute(df.iloc[:cut], **d.defaults).to_numpy()
        # Values up to the cut must be identical when later bars are removed.
        np.testing.assert_array_equal(
            np.nan_to_num(full[:cut]), np.nan_to_num(prefix),
            err_msg=f"{d.name} uses future data")


# -- execution timing --------------------------------------------------------------------------
def test_order_decided_at_close_fills_at_next_open_with_slippage():
    cfg = BacktestConfig(initial_capital=1_000_000, position_pct=0.1,
                         costs=CostModel(brokerage_per_order=0, charges_pct=0, slippage_points=0.5))
    df = _flat_bars(10)
    df.iloc[3, df.columns.get_loc("open")] = 101.0  # open of bar 3 is where the fill must happen
    votes = np.zeros(10, dtype=int)
    votes[2] = 1  # signal known at close of bar 2
    res = run_backtest(df, votes, min_votes=1, direction="long", exit_rules=Exit(), cfg=cfg)
    entry = res.trades[0]
    assert entry["entry_time"] == df.index[3].isoformat()
    assert entry["entry_price"] == pytest.approx(101.0 + 0.5)


def test_no_trade_when_signal_is_on_last_bar():
    cfg = BacktestConfig(initial_capital=1_000_000, position_pct=0.1)
    df = _flat_bars(10)
    votes = np.zeros(10, dtype=int)
    votes[-1] = 1
    res = run_backtest(df, votes, min_votes=1, direction="both", exit_rules=Exit(), cfg=cfg)
    assert res.trades == []


def test_stop_loss_triggers_intrabar_at_stop_price():
    cfg = BacktestConfig(initial_capital=1_000_000, position_pct=0.1,
                         costs=CostModel(brokerage_per_order=0, charges_pct=0, slippage_points=0))
    opens = [100, 100, 100, 100, 100, 100]
    highs = [100.5, 100.5, 100.5, 100.5, 100.5, 100.5]
    lows = [99.7, 99.7, 99.7, 99.0, 99.5, 99.5]
    closes = [100, 100, 100, 99.5, 100, 100]
    df = _frame(opens, highs, lows, closes)
    votes = np.array([1, 1, 0, 0, 0, 0])  # long decided at bar 1 close, filled at bar 2 open
    res = run_backtest(df, votes, min_votes=1, direction="long",
                       exit_rules=Exit(stop_loss_pct=0.5, target_pct=0), cfg=cfg)
    t = res.trades[0]
    assert t["reason"] == "stop_loss"
    assert t["exit_price"] == pytest.approx(100 * (1 - 0.005))


def test_gap_through_stop_fills_at_open_not_stop():
    cfg = BacktestConfig(initial_capital=1_000_000, position_pct=0.1,
                         costs=CostModel(brokerage_per_order=0, charges_pct=0, slippage_points=0))
    opens = [100, 100, 100, 97.0, 97.0]
    highs = [100.5, 100.5, 100.5, 97.5, 97.5]
    lows = [99.7, 99.7, 99.7, 96.5, 96.5]
    closes = [100, 100, 100, 97, 97]
    df = _frame(opens, highs, lows, closes)
    votes = np.array([1, 1, 0, 0, 0])
    res = run_backtest(df, votes, min_votes=1, direction="long",
                       exit_rules=Exit(stop_loss_pct=0.5, target_pct=0), cfg=cfg)
    assert res.trades[0]["reason"] == "stop_loss"
    assert res.trades[0]["exit_price"] == pytest.approx(97.0)


def test_pnl_and_cash_reconcile_exactly_with_costs():
    cfg = BacktestConfig(initial_capital=1_000_000, position_pct=0.2,
                         costs=CostModel(brokerage_per_order=20, charges_pct=0.001, slippage_points=0.3))
    rng = np.random.default_rng(5)
    close = 100 + np.cumsum(rng.normal(0, 0.4, 800))
    df = _frame(close, close + 0.4, close - 0.4, close)
    votes = np.sign(np.sin(np.arange(800) / 25)).astype(int) * 2
    res = run_backtest(df, votes, min_votes=2, direction="both",
                       exit_rules=Exit(stop_loss_pct=0.4, target_pct=0.8), cfg=cfg)
    assert res.trades, "scenario should produce trades"
    net = sum(t["net_pnl"] for t in res.trades)
    assert res.final_capital == pytest.approx(1_000_000 + net, abs=0.05)
    assert res.equity[-1] == pytest.approx(res.final_capital, abs=0.05)
    assert all(t["net_pnl"] == pytest.approx(t["gross_pnl"] - t["costs"], abs=0.05) for t in res.trades)


def test_square_off_flattens_position_and_no_entries_late():
    cfg = BacktestConfig(initial_capital=1_000_000, position_pct=0.1,
                         no_entry_after_minute=9 * 60 + 20, square_off_minute=9 * 60 + 25,
                         costs=CostModel(brokerage_per_order=0, charges_pct=0, slippage_points=0))
    df = _flat_bars(30)  # 09:15 .. 09:44
    votes = np.ones(30, dtype=int)
    res = run_backtest(df, votes, min_votes=1, direction="long", exit_rules=Exit(), cfg=cfg)
    assert res.trades[0]["reason"] == "square_off"
    assert res.trades[0]["exit_time"].endswith("09:25:00")
    assert all(t["entry_time"][11:16] < "09:21" for t in res.trades)


# -- metrics -----------------------------------------------------------------------------------
def test_metrics_on_known_trades():
    trades = [{"net_pnl": 100.0, "costs": 10, "side": "LONG", "reason": "target"},
              {"net_pnl": -50.0, "costs": 10, "side": "SHORT", "reason": "stop_loss"},
              {"net_pnl": 200.0, "costs": 10, "side": "LONG", "reason": "target"},
              {"net_pnl": -100.0, "costs": 10, "side": "LONG", "reason": "stop_loss"}]
    equity = np.array([1000, 1100, 1050, 1250, 1150], dtype=float)
    m = compute_metrics(trades, equity, 1000)
    assert m["trades"] == 4
    assert m["net_profit"] == 150.0
    assert m["win_rate"] == 50.0
    assert m["profit_factor"] == pytest.approx(300 / 150)
    assert m["max_drawdown"] == pytest.approx(100.0)  # 1250 -> 1150
    assert m["max_drawdown_pct"] == pytest.approx(8.0)
    assert m["longs"] == 3 and m["shorts"] == 1


def test_profit_factor_without_losses_is_capped():
    m = compute_metrics([{"net_pnl": 5.0, "costs": 0, "side": "LONG", "reason": "t"}],
                        np.array([1, 2.0]), 1)
    assert m["profit_factor"] == 999.0


# -- data validation ---------------------------------------------------------------------------
def _csv(tmp_path, text, name="d.csv"):
    p = tmp_path / name
    p.write_text(text)
    return p


def _rows(n=600, price=100.0):
    lines = ["timestamp,open,high,low,close,volume"]
    t = pd.Timestamp("2025-01-06 09:15")
    for i in range(n):
        ts = (t + pd.Timedelta(minutes=i)).isoformat()
        lines.append(f"{ts},{price},{price + 1},{price - 1},{price},10")
    return "\n".join(lines) + "\n"


def test_csv_missing_columns_rejected(tmp_path):
    p = _csv(tmp_path, "timestamp,open,high,low\n2025-01-06 09:15,1,1,1\n")
    with pytest.raises(ValueError, match="missing required columns"):
        import_csv(p, name="x", symbol="X")


def test_csv_inconsistent_ohlc_rejected(tmp_path):
    lines = _rows(600).splitlines()
    lines[5] = lines[5].replace(",101.0,", ",50.0,")  # high below close
    with pytest.raises(ValueError, match="inconsistent OHLC"):
        import_csv(_csv(tmp_path, "\n".join(lines) + "\n", "bad.csv"), name="bad", symbol="X")


def test_csv_duplicate_timestamps_collapsed_and_separate_date_time_accepted(tmp_path):
    lines = ["date,time,open,high,low,close"]
    t = pd.Timestamp("2025-01-06 09:15")
    for i in range(600):
        ts = t + pd.Timedelta(minutes=i)
        lines.append(f"{ts.date()},{ts.strftime('%H:%M')},100,101,99,100")
    lines.append(lines[1])  # duplicate row
    ds = import_csv(_csv(tmp_path, "\n".join(lines) + "\n", "dt.csv"), name="dt", symbol="X", replace=True)
    assert ds["rows"] == 600


def test_reimport_without_replace_is_refused(tmp_path):
    p = _csv(tmp_path, _rows(600), "same.csv")
    import_csv(p, name="same", symbol="X", replace=True)
    with pytest.raises(ValueError, match="already exists"):
        import_csv(p, name="same", symbol="X")


def test_clean_candles_requires_minimum_history():
    df = _frame([100] * 100, [101] * 100, [99] * 100, [100] * 100)
    with pytest.raises(ValueError, match="at least 500"):
        clean_candles(df)


# -- strategy specification ---------------------------------------------------------------------
def test_strategy_hash_is_order_independent_and_stable():
    a = StrategySpec.build([{"name": "ref_rsi"}, {"name": "ref_macd"}], min_votes=2)
    b = StrategySpec.build([{"name": "ref_macd"}, {"name": "ref_rsi"}], min_votes=2)
    assert a.strategy_hash == b.strategy_hash
    assert spec_from_dict(a.to_dict()).strategy_hash == a.strategy_hash


def test_strategy_validation_rules():
    with pytest.raises(ValueError, match="min_votes"):
        StrategySpec.build([{"name": "ref_rsi"}], min_votes=2)
    with pytest.raises(ValueError, match="once"):
        StrategySpec.build([{"name": "ref_rsi"}, {"name": "ref_rsi"}])
    with pytest.raises(ValueError, match="direction"):
        StrategySpec.build([{"name": "ref_rsi"}], direction="sideways")
    with pytest.raises(ValueError, match="unknown indicator"):
        StrategySpec.build([{"name": "does_not_exist"}])


def test_registry_param_spec_rejects_bad_default():
    with pytest.raises(ValueError):
        ParamSpec("p", 99, 1, 10)
