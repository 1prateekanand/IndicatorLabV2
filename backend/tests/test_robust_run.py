"""Robustness runs end to end on a finished combination run, and their refusal when code changed."""
import json
import time

import pytest

from indicatorlab.db import get_db
from indicatorlab.research import combo_service as cs
from indicatorlab.research import robust_service as rs
from indicatorlab.research import service as svc
from indicatorlab.research.combination import ComboRequest
from indicatorlab.research.individual import Rules


def _wait(run_id, db, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        s = svc.run_status(run_id, db)
        if s["status"] not in ("running", "paused"):
            return s
        time.sleep(0.2)
    raise AssertionError("run did not finish")


@pytest.fixture(scope="module")
def finished_combination(dataset_id):
    db = get_db()
    from indicatorlab.registry import REGISTRY
    REGISTRY.discover()
    run = cs.start_combination(ComboRequest(dataset_id=dataset_id, rules=Rules(stop_loss_pct=0.5, target_pct=1.0),
                                            segments=3, pool_size=5, top_n=5), db)
    done = _wait(run["id"], db)
    assert done["status"] == "completed", done
    return run["id"]


def test_robustness_refuses_an_unfinished_or_missing_parent(dataset_id):
    db = get_db()
    with pytest.raises(LookupError):
        rs.start_robustness(999999, db=db)
    with pytest.raises(ValueError, match="between"):
        rs.start_robustness(999999, permutations=5, db=db)


def test_robustness_run_produces_all_five_tests(dataset_id, finished_combination):
    db = get_db()
    run = rs.start_robustness(finished_combination, candidates=3, permutations=20, mc_iterations=500,
                              spa_draws=100, db=db)
    done = _wait(run["id"], db)
    assert done["status"] == "completed", done
    got = rs.latest_robustness(finished_combination, db)
    r = got["result"]
    assert r["dataset"]["label"] in ("market data", "SYNTHETIC DATA: not market performance")
    assert "p_value" in r["spa"] and 0 <= r["spa"]["p_value"] <= 1
    assert r["permutation"]["draws"] == 20 and 0 < r["permutation"]["p_value"] <= 1
    assert r["monte_carlo"].get("iterations") == 500 or r["monte_carlo"].get("status") == "insufficient_sample"
    assert r["window"]["span"].startswith("train split")
    assert r["pbo"]["status"] in ("ok", "insufficient_sample")
    assert r["dsr"]["status"] in ("ok", "insufficient_sample", "insufficient_obs") or "status" in r["dsr"]


def test_robustness_refuses_candidates_whose_code_changed(dataset_id, finished_combination):
    db = get_db()
    row = db.one("SELECT members_json FROM combination_rankings WHERE run_id=? AND rank=1", (finished_combination,))
    members = json.loads(row["members_json"])
    stale = [{**m, "code_hash": "STALE"} for m in members]
    db.execute("UPDATE combination_rankings SET members_json=? WHERE run_id=? AND rank=1",
               (json.dumps(stale), finished_combination))
    try:
        run = rs.start_robustness(finished_combination, candidates=2, permutations=20, mc_iterations=500,
                                  spa_draws=100, db=db)
        done = _wait(run["id"], db)
        assert done["status"] == "failed" and "changed after" in (done["error"] or "")
    finally:
        db.execute("UPDATE combination_rankings SET members_json=? WHERE run_id=? AND rank=1",
                   (json.dumps(members), finished_combination))
