"""Individual indicator research: scoring formulas, discovery, evaluation, cache, runs and dynamic registration."""
import time

import numpy as np
import pandas as pd
import pytest

from indicatorlab import studio
from indicatorlab.config import DEFAULT_BACKTEST
from indicatorlab.db import get_db
from indicatorlab.engine import Exit, run_backtest
from indicatorlab.research import individual as ind
from indicatorlab.research import service
from indicatorlab.research.scoring import (beta_binomial_mean, shrunk_scores, wilson_lower_bound)

GOOD = '''def compute(df, length=5):
    closes = df["close"]
    return [0 if i < length else (1 if closes[i] > closes[i - length] else -1) for i in range(len(closes))]'''

GOOD_V2 = GOOD.replace("length=5", "length=7")


def _wait(run_id, db, timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        s = service.run_status(run_id, db)
        if s["status"] not in ("running", "paused"):
            return s
        time.sleep(0.3)
    raise AssertionError(f"run {run_id} did not finish: {service.run_status(run_id, db)}")


# -- scoring -----------------------------------------------------------------------------------
def test_beta_binomial_mean_matches_formula():
    assert beta_binomial_mean(60, 100) == pytest.approx((60 + 3) / (100 + 6))
    assert beta_binomial_mean(0, 0) == pytest.approx(0.5)  # prior mean with no trades
    with pytest.raises(ValueError):
        beta_binomial_mean(5, 3)


def test_wilson_lower_bound_known_values():
    # Wilson interval lower bound, z = 1.28, for p = 0.5 and n = 100
    assert wilson_lower_bound(0.5, 100, 1.28) == pytest.approx(0.43651793, abs=1e-6)
    assert wilson_lower_bound(0.5, 100, 1.28) < 0.5
    assert wilson_lower_bound(0.0, 50) <= 0.0 + 1e-12


def test_shrunk_scores_zero_below_minimum_trades():
    out = shrunk_scores(9, 14)  # 14 trades < 15
    assert out == {"shrunk_score": 0.0, "wilson_lb": 0.0, "eligible_for_score": False}
    ok = shrunk_scores(9, 15)
    assert ok["eligible_for_score"] is True and 0 < ok["wilson_lb"] < ok["shrunk_score"] < 1


def test_wilson_is_more_conservative_with_fewer_trades():
    assert shrunk_scores(6, 10)["wilson_lb"] == 0.0  # below the minimum trade count
    assert shrunk_scores(60, 100)["wilson_lb"] < shrunk_scores(600, 1000)["wilson_lb"]


# -- holding period -----------------------------------------------------------------------------
def _ramp(n=60):
    idx = pd.date_range("2024-01-02 09:15", periods=n, freq="min")
    close = 100 + np.arange(n) * 0.01
    return pd.DataFrame({"open": close, "high": close + 0.02, "low": close - 0.02, "close": close,
                         "volume": 1.0}, index=idx)


def test_max_hold_bars_closes_after_holding_period():
    df = _ramp()
    votes = np.zeros(len(df), dtype=int)
    votes[2] = 1  # long stays signalled, so only the holding rule can close it
    votes[3:] = 1
    res = run_backtest(df, votes, min_votes=1, direction="long", exit_rules=Exit(0.0, 0.0),
                       cfg=DEFAULT_BACKTEST, max_hold_bars=5)
    assert res.trades and res.trades[0]["reason"] == "holding_period"
    assert res.trades[0]["bars_held"] <= 6


def test_no_holding_limit_keeps_position_open_until_end():
    df = _ramp()
    votes = np.zeros(len(df), dtype=int)
    votes[2:] = 1
    res = run_backtest(df, votes, min_votes=1, direction="long", exit_rules=Exit(0.0, 0.0), cfg=DEFAULT_BACKTEST)
    assert res.trades[-1]["reason"] == "end_of_data"


# -- discovery ----------------------------------------------------------------------------------
def test_discovery_includes_registered_studio_and_reports_unavailable(dataset_id):
    db = get_db()
    ok = studio.create_indicator({"name": "disc_ok", "code": GOOD}, db)
    studio.update_indicator(ok["id"], {"name": "disc_ok", "code": GOOD}, db)
    draft = studio.create_indicator({"name": "disc_draft", "code": GOOD}, db)
    available, unavailable = ind.discover_sources(db)
    keys = {s.key for s in available}
    assert f"studio:{ok['id']}" in keys
    assert f"studio:{draft['id']}" not in keys
    assert any(s.key == f"studio:{draft['id']}" for s in unavailable)
    assert any(s.source == "code" for s in available), "code-registry indicators are discovered too"
    studio.delete_item("indicator", ok["id"], db)
    studio.delete_item("indicator", draft["id"], db)


# -- evaluation consistency ---------------------------------------------------------------------
def test_single_indicator_matches_direct_backtest(dataset_id):
    rules = ind.Rules(direction="both", stop_loss_pct=0.5, target_pct=1.0)
    win = ind.load_window(dataset_id, "1min", "train")
    src = next(s for s in ind.discover_sources()[0] if s.name == "ref_rsi")
    metrics = ind.evaluate_source(src, win, rules)
    # reference computation with the same engine, same window and same rules
    votes_full = ind.compute_votes(src, win.full.iloc[:win.eval_end])
    res = run_backtest(win.full.iloc[win.eval_start:win.eval_end], votes_full[win.eval_start:win.eval_end].astype(int),
                       min_votes=1, direction="both", exit_rules=Exit(0.5, 1.0), cfg=rules.backtest_config())
    assert metrics["trades"] == len(res.trades)
    assert metrics["wins"] == sum(1 for t in res.trades if t["net_pnl"] > 0)
    assert metrics["wins"] + metrics["losses"] == metrics["trades"]
    assert metrics["net_profit"] == pytest.approx(round(res.final_capital - rules.backtest_config().initial_capital, 2), abs=0.02)


def test_costs_reduce_net_profit(dataset_id):
    src = next(s for s in ind.discover_sources()[0] if s.name == "ref_rsi")
    win = ind.load_window(dataset_id, "1min", "train")
    cheap = ind.evaluate_source(src, win, ind.Rules(brokerage_per_order=0, charges_pct=0, slippage_points=0))
    dear = ind.evaluate_source(src, win, ind.Rules(brokerage_per_order=50, charges_pct=0.001, slippage_points=2))
    # Costs are charged per trade. Position sizing uses available cash, so the trade count may differ
    # slightly between cost settings; the claim tested is that costs are charged and reduce net profit.
    assert dear["total_costs"] > cheap["total_costs"] >= 0
    assert dear["net_profit"] < cheap["net_profit"]


def test_oos_split_is_refused(dataset_id):
    with pytest.raises(ValueError, match="OOS"):
        ind.load_window(dataset_id, "1min", "oos")
    with pytest.raises(ValueError):
        ind.Rules(direction="sideways").validate()


# -- runs: completion, cache, stop, dynamic registration ----------------------------------------
def test_full_run_then_cache_then_dynamic_registration(dataset_id):
    db = get_db()
    run = service.start_run(service.RunRequest(dataset_id=dataset_id, split="train",
                                               rules=ind.Rules(stop_loss_pct=0.5, target_pct=1.0)), db)
    done = _wait(run["id"], db)
    assert done["status"] == "completed", done
    table = service.individual_table(dataset_id, "train", done["id"], db)
    evaluated = [r for r in table["rows"] if r["status"] == "evaluated"]
    assert {r["name"] for r in evaluated} >= {"ref_rsi", "ref_macd"}
    rsi = next(r for r in evaluated if r["name"] == "ref_rsi")
    assert set(rsi) >= {"trades", "wins", "losses", "raw_win_rate", "shrunk_score", "wilson_lb", "net_profit",
                        "profit_factor", "max_drawdown_pct"}

    # identical request: every result comes from the cache, no new completed rows are written
    before = db.one("SELECT COUNT(*) AS n FROM indicator_evaluations WHERE status='completed'")["n"]
    again = _wait(service.start_run(service.RunRequest(dataset_id=dataset_id, split="train",
                                                       rules=ind.Rules(stop_loss_pct=0.5, target_pct=1.0)), db)["id"], db)
    after = db.one("SELECT COUNT(*) AS n FROM indicator_evaluations WHERE status='completed'")["n"]
    assert again["status"] == "completed" and after == before
    assert again["config_hash"] == done["config_hash"]

    # a newly registered indicator is discovered without any code change and appears as not evaluated
    new = studio.create_indicator({"name": "dyn_new", "code": GOOD}, db)
    studio.update_indicator(new["id"], {"name": "dyn_new", "code": GOOD}, db)
    table2 = service.individual_table(dataset_id, "train", again["id"], db)
    row = next(r for r in table2["rows"] if r["name"] == "dyn_new")
    assert row["status"] == "not_evaluated"
    # evaluate it once so it has a result under the current code
    evald = _wait(service.start_run(service.RunRequest(dataset_id=dataset_id, split="train",
                                                       rules=ind.Rules(stop_loss_pct=0.5, target_pct=1.0)), db)["id"], db)
    row = next(r for r in service.individual_table(dataset_id, "train", evald["id"], db)["rows"]
               if r["name"] == "dyn_new")
    assert row["status"] == "evaluated"
    # editing its code changes the cache key: the old result is never reused, and the table reports the change
    studio.update_indicator_meta(new["id"], {"name": "dyn_new", "code": GOOD_V2}, db)
    table3 = service.individual_table(dataset_id, "train", again["id"], db)
    assert next(r for r in table3["rows"] if r["name"] == "dyn_new")["status"] == "changed_since_run"
    studio.delete_item("indicator", new["id"], db)


def test_rules_change_creates_a_new_configuration(dataset_id):
    db = get_db()
    a = _wait(service.start_run(service.RunRequest(dataset_id=dataset_id, split="train",
                                                   rules=ind.Rules(target_pct=1.0)), db)["id"], db)
    b = _wait(service.start_run(service.RunRequest(dataset_id=dataset_id, split="train",
                                                   rules=ind.Rules(target_pct=1.5)), db)["id"], db)
    assert a["config_hash"] != b["config_hash"]
    assert b["status"] == "completed"


def test_stop_ends_run_as_stopped(dataset_id):
    db = get_db()
    run = service.start_run(service.RunRequest(dataset_id=dataset_id, split="validation",
                                               rules=ind.Rules(stop_loss_pct=0.3, target_pct=0.8)), db)
    service.control_run(run["id"], "stop", db)
    done = _wait(run["id"], db)
    assert done["status"] in ("stopped", "completed")  # completes first if it is already finished


def test_restart_marks_running_runs_interrupted(dataset_id):
    db = get_db()
    with db.transaction() as cur:
        cur.execute("INSERT INTO research_runs(created_at, kind, dataset_id, split, config_hash, config_json, status,"
                    " message) VALUES ('2026-01-01T00:00:00+00:00','individual',?,'train','ghost','{}','running','x')",
                    (dataset_id,))
        ghost = cur.lastrowid
    assert service.recover_after_restart(db) >= 1
    assert db.one("SELECT status FROM research_runs WHERE id=?", (ghost,))["status"] == "interrupted"


def test_http_contract(dataset_id):
    from fastapi.testclient import TestClient

    from indicatorlab.api import app

    with TestClient(app) as client:
        r = client.post("/api/research/individual/start", json={"dataset_id": dataset_id, "split": "oos"})
        assert r.status_code == 422 or r.status_code == 400
        bad = client.post("/api/research/individual/start",
                          json={"dataset_id": dataset_id, "split": "train", "rules": {"direction": "sideways"}})
        assert bad.status_code == 400
        table = client.get("/api/research/individual", params={"dataset_id": dataset_id, "split": "train"})
        assert table.status_code == 200 and "rows" in table.json()
        assert client.get("/api/research/runs/999999").status_code == 404
