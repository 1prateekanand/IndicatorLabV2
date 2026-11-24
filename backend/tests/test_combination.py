"""Combination research: AND logic, cache keys, walk-forward geometry, Spearman filter, budget,
run lifecycle, one-shot OOS and refusal after an indicator changes."""
import json
import time

import numpy as np
import pytest

from indicatorlab import studio
from indicatorlab.db import get_db
from indicatorlab.research import combo_service as cs
from indicatorlab.research import service as svc
from indicatorlab.research.combination import (ComboRequest, Evaluator, and_votes, combo_key, fold_plan,
                                               select, _rho, _ranks)
from indicatorlab.research.individual import Rules, Source, discover_sources, load_window

GOOD = '''def compute(df, length=5):
    closes = df["close"]
    return [0 if i < length else (1 if closes[i] > closes[i - length] else -1) for i in range(len(closes))]'''


def _src(name, code_hash="h1"):
    return Source(key=f"code:{name}", name=name, source="code", category="t", code_hash=code_hash)


def _wait(run_id, db, timeout=240):
    end = time.time() + timeout
    while time.time() < end:
        s = svc.run_status(run_id, db)
        if s["status"] not in ("running", "paused"):
            return s
        time.sleep(0.2)
    raise AssertionError("run did not finish")


# -- AND logic ------------------------------------------------------------------------------------
def test_and_votes_requires_agreement():
    a = np.array([1, 1, 1, -1, 0, 1])
    b = np.array([1, -1, 1, -1, 1, 0])
    c = np.array([1, 1, 0, -1, 1, 1])
    out = and_votes([a, b, c])
    assert out.tolist() == [1, 0, 0, -1, 0, 0]  # long only where all long, short only where all short
    assert and_votes([a]).tolist() == a.tolist()


def test_and_votes_treats_any_nonzero_value_as_its_sign():
    out = and_votes([np.array([3, -2, 5]), np.array([1, -1, 1])])
    assert out.tolist() == [1, -1, 1]


def test_combo_key_is_order_free_and_tracks_code_hash():
    x, y = _src("x"), _src("y")
    assert combo_key([x, y]) == combo_key([y, x])
    assert combo_key([x, y]) != combo_key([x, _src("y", "h2")])  # an edited member is a new combination


# -- walk-forward geometry -------------------------------------------------------------------------
def test_fold_plan_purges_and_never_tests_what_it_selected():
    plan = fold_plan(0, 100_000, segments=4, purge=60)
    assert [p["fold"] for p in plan] == [1, 2, 3]
    for p in plan:
        sel_a, sel_b = p["sel"]
        test_a, test_b = p["test"]
        assert sel_a == 0                       # expanding window starts at the train start
        assert sel_b == test_a - 60             # purge gap before the test chunk
        assert test_a < test_b
    # the test chunk of one fold can be part of a later selection window (expanding walk-forward), never its own
    assert plan[1]["sel"][1] >= plan[0]["test"][1] - 60


def test_fold_plan_zero_purge_is_contiguous():
    plan = fold_plan(0, 1000, segments=2, purge=0)
    assert plan == [{"fold": 1, "sel": (0, 500), "test": (500, 1000), "purge": 0}]


# -- correlation helpers ----------------------------------------------------------------------------
def test_spearman_helpers_identity_and_constant():
    x = np.array([0, 1, 1, -1, 0, 1, -1], dtype=float)
    assert _rho(_ranks(x), _ranks(x)) == pytest.approx(1.0)
    assert _rho(_ranks(x), _ranks(-x)) == pytest.approx(-1.0)
    assert np.isnan(_rho(_ranks(x), _ranks(np.zeros(7))))  # constant signal has no rank correlation


# -- selection on real indicator code ----------------------------------------------------------------
def test_select_filters_correlated_indicators_and_reports_coverage(dataset_id):
    db = get_db()
    from indicatorlab.registry import REGISTRY
    REGISTRY.discover()
    win = load_window(dataset_id, "1min", "train", db=db)
    rules = Rules(stop_loss_pct=0.5, target_pct=1.0)
    sources, _ = discover_sources(db)
    ev = Evaluator(db, win, rules, "cfg-select", budget=5000, run_id=None)
    sel = select(ev, sources, win.eval_start, win.eval_end, "unit", corr_threshold=0.5, pool_size=12, top_n=5)
    assert sel["coverage"]["indicators_available"] == len(sources)
    for d in sel["dropped"]:
        assert abs(d["rho"]) >= 0.5 and d["kept_partner"]
    pool_names = {p["name"] for p in sel["pool"]}
    assert not any(d["name"] in pool_names for d in sel["dropped"])
    if sel["best"]:
        assert sel["best"]["metrics"]["eligible_for_score"] is True
        assert sel["best"]["size"] <= 5
    pairs = sel["coverage"]["pairs_evaluated"]
    assert pairs == sel["coverage"]["pairs_possible"] or sel["coverage"]["budget_reached"]


def test_budget_is_enforced_and_reported(dataset_id):
    db = get_db()
    from indicatorlab.registry import REGISTRY
    REGISTRY.discover()
    win = load_window(dataset_id, "1min", "train", db=db)
    sources, _ = discover_sources(db)
    ev = Evaluator(db, win, Rules(), "cfg-budget", budget=3, run_id=None)
    sel = select(ev, sources, win.eval_start, win.eval_end, "budget", corr_threshold=0.8, pool_size=12, top_n=5)
    assert sel["coverage"]["budget_reached"] is True
    assert ev.fresh == 3


def test_cache_hits_do_not_consume_budget(dataset_id):
    db = get_db()
    from indicatorlab.registry import REGISTRY
    REGISTRY.discover()
    win = load_window(dataset_id, "1min", "train", db=db)
    src = next(s for s in discover_sources(db)[0] if s.name == "ref_rsi")
    ev1 = Evaluator(db, win, Rules(), "cfg-cache", budget=10, run_id=None)
    first = ev1.evaluate([src], "cache", win.eval_start, win.eval_end)
    ev2 = Evaluator(db, win, Rules(), "cfg-cache", budget=0, run_id=None)  # zero budget: must still hit the cache
    again = ev2.evaluate([src], "cache", win.eval_start, win.eval_end)
    assert again["trades"] == first["trades"] and ev2.cache_hits == 1 and ev2.fresh == 0


# -- run lifecycle, dashboard payload, one-shot OOS ---------------------------------------------------
def test_combination_run_completes_and_exposes_results(dataset_id):
    db = get_db()
    req = ComboRequest(dataset_id=dataset_id, rules=Rules(stop_loss_pct=0.5, target_pct=1.0), segments=3,
                       pool_size=5, top_n=5, max_evaluations=5000)
    run = cs.start_combination(req, db)
    done = _wait(run["id"], db)
    assert done["status"] == "completed", done
    payload = cs.combination_payload(run["id"], db)
    assert len(payload["folds"]) == 2
    assert payload["selection"]["coverage"]["indicators_available"] >= 7
    assert payload["correlation"]["matrix"] and len(payload["correlation"]["matrix"]) == len(payload["correlation"]["keys"])
    assert payload["train_curve"] is None or len(payload["train_curve"]["equity"]) > 0
    if payload["frozen"]:
        assert payload["frozen"]["members"]
    dash = cs.dashboard_payload(dataset_id, db)
    assert dash["combination"]["run"]["id"] == run["id"]
    assert all(r["status"] == "evaluated" for r in dash["individual"]["ranking"])


def test_oos_is_one_shot_and_refuses_changed_indicators(dataset_id):
    db = get_db()
    from indicatorlab.registry import REGISTRY
    REGISTRY.discover()
    run = cs.start_combination(ComboRequest(dataset_id=dataset_id, pool_size=5, top_n=5, segments=3), db)
    _wait(run["id"], db)
    payload = cs.combination_payload(run["id"], db)
    if payload["frozen"] is None:
        pytest.skip("no eligible combination on this synthetic sample")
    first = cs.run_oos(run["id"], db)
    assert first["metrics"]["trades"] >= 0 and len(first["curve"]["equity"]) > 0
    with pytest.raises(ValueError, match="one-shot"):
        cs.run_oos(run["id"], db)
    assert cs.oos_for_run(run["id"], db)["metrics"] == first["metrics"]


def test_oos_refuses_a_frozen_indicator_whose_code_changed(dataset_id):
    db = get_db()
    from indicatorlab.registry import REGISTRY
    REGISTRY.discover()
    req = ComboRequest(dataset_id=dataset_id, rules=Rules(), segments=3, pool_size=5, top_n=5)
    run = cs.start_combination(req, db)
    _wait(run["id"], db)
    frozen = {"combo_key": "x", "members": [{"key": "code:ref_rsi", "name": "ref_rsi", "source": "code",
                                              "category": "t", "code_hash": "STALE"}],
              "size": 1, "train_metrics": {}, "rules": {}, "frozen_at": "now"}
    db.execute("INSERT OR REPLACE INTO run_artifacts(run_id, kind, payload_json) VALUES (?,?,?)",
               (run["id"], "frozen", json.dumps(frozen)))
    db.execute("DELETE FROM oos_results WHERE run_id=?", (run["id"],))
    with pytest.raises(ValueError, match="changed after the freeze"):
        cs.run_oos(run["id"], db)


def test_new_studio_indicator_is_discovered_by_the_next_run(dataset_id):
    db = get_db()
    from indicatorlab.registry import REGISTRY
    REGISTRY.discover()
    before = cs.combination_payload(cs.latest_combination_run(dataset_id, db), db)["selection"]["coverage"]["indicators_available"]
    item = studio.create_indicator({"name": "dyn_combo_new", "code": GOOD}, db)
    studio.update_indicator(item["id"], {"name": "dyn_combo_new", "code": GOOD}, db)
    run = cs.start_combination(ComboRequest(dataset_id=dataset_id, pool_size=5, top_n=3, segments=3), db)
    _wait(run["id"], db)
    after = cs.combination_payload(run["id"], db)["selection"]["coverage"]["indicators_available"]
    assert after == before + 1
    studio.delete_item("indicator", item["id"], db)
