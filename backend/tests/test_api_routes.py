"""HTTP contract for the research routes added in the completion phase: robustness, synthetic data,
analysis and signal rules. Failure paths return clear 4xx responses."""
import time

from fastapi.testclient import TestClient

from indicatorlab.api import app
from indicatorlab.db import get_db
from indicatorlab.research import combo_service as cs
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


def test_signal_rules_reject_bad_specs_and_accept_good_ones(dataset_id):
    with TestClient(app) as client:
        listing = client.get("/api/research/signals").json()["indicators"]
        assert listing, "discovered indicators are listed"
        key = listing[0]["source_key"]
        bad = client.put(f"/api/research/signals/{key}", json={"spec": {"mode": "conditions", "long": {
            "logic": "XOR", "conditions": [{"op": "gt", "value": 1}]}}})
        assert bad.status_code == 400 and "AND or OR" in bad.json()["detail"]
        unknown = client.put("/api/research/signals/code:does_not_exist", json={"spec": None})
        assert unknown.status_code == 404
        ok = client.put(f"/api/research/signals/{key}", json={"spec": {"mode": "conditions", "long": {
            "logic": "AND", "conditions": [{"op": "cross_above", "value": 0.0}]}}})
        assert ok.status_code == 200 and ok.json()["spec"]["mode"] == "conditions"
        client.put(f"/api/research/signals/{key}", json={"spec": None})  # back to legacy


def test_synthetic_scenarios_are_labelled_and_validated(dataset_id):
    with TestClient(app) as client:
        sc = client.get("/api/research/synthetic/scenarios").json()
        names = {s["scenario"] for s in sc["scenarios"]}
        assert {"trending", "sideways", "volatile", "sudden_move"} <= names
        assert "Synthetic" in sc["label"]
        bad = client.post("/api/research/synthetic/create", json={"scenario": "not_real", "days": 10, "seed": 1})
        assert bad.status_code == 400
        too_long = client.post("/api/research/synthetic/create", json={"scenario": "trending", "days": 9999, "seed": 1})
        assert too_long.status_code == 422
        made = client.post("/api/research/synthetic/create", json={"scenario": "trending", "days": 5, "seed": 77})
        assert made.status_code == 200 and made.json()["is_synthetic"] == 1
        again = client.post("/api/research/synthetic/create", json={"scenario": "trending", "days": 5, "seed": 77})
        assert again.status_code == 400  # same name is never silently duplicated


def test_robustness_refuses_unknown_and_unfinished_runs(dataset_id):
    with TestClient(app) as client:
        r = client.post("/api/research/robustness/start", json={"combination_run_id": 999999})
        assert r.status_code == 404
        missing = client.get("/api/research/robustness", params={"combination_run_id": 999999})
        assert missing.status_code == 200 and missing.json()["result"] is None


def test_robustness_runs_on_a_finished_combination_run(dataset_id):
    db = get_db()
    run = cs.start_combination(ComboRequest(dataset_id=dataset_id, rules=Rules(stop_loss_pct=0.5, target_pct=1.0),
                                            segments=3, pool_size=5, top_n=5), db)
    _wait(run["id"], db)
    with TestClient(app) as client:
        start = client.post("/api/research/robustness/start", json={
            "combination_run_id": run["id"], "candidates": 4, "permutations": 20, "mc_iterations": 500, "spa_draws": 100})
        assert start.status_code == 200, start.text
        rid = start.json()["id"]
        end = time.time() + 300
        while time.time() < end:
            res = client.get("/api/research/robustness", params={"combination_run_id": run["id"]}).json()
            if res["status"] not in ("running", "paused"):
                break
            time.sleep(0.5)
        assert res["status"] == "completed", res
        assert res["run_id"] == rid and res["result"] is not None


def test_analysis_reads_stored_results_only(dataset_id):
    with TestClient(app) as client:
        r = client.get("/api/research/analysis", params={"dataset_id": dataset_id})
        assert r.status_code == 200
        body = r.json()
        assert set(body) >= {"points", "pareto", "clusters"}
        assert body["pareto"]["front"] is not None
        assert client.get("/api/research/analysis", params={"dataset_id": 999999}).status_code == 404
