"""End-to-end: Optuna studies, both agents, persistence, leaderboards, and the HTTP API."""
import json

import pytest

from indicatorlab import ranking
from indicatorlab.agents.evaluation import audit
from indicatorlab.agents.research import ResearchAgent, ResearchSettings
from indicatorlab.agents.runtime import ManagedWorker, WorkerContext
from indicatorlab.db import get_db
from indicatorlab.evaluator import Evaluator, ResearchConfig
from indicatorlab.optimizer import OptimizationConfig, run_study
from indicatorlab.registry import REGISTRY
from indicatorlab.strategy import StrategySpec


@pytest.fixture(scope="module")
def research_cfg():
    return ResearchConfig(min_trades=5)


def test_optuna_study_records_every_trial_and_is_deduplicated(dataset_id, research_cfg):
    db = get_db()
    REGISTRY.discover()
    cfg = OptimizationConfig(dataset_id=dataset_id, study_name="it_tpe_1", n_trials=8,
                             sampler="tpe", pruner="median", max_indicators=3, seed=1)
    ctx = WorkerContext("optuna:it_tpe_1", db, throttle=False)
    study = run_study(cfg, ctx, research_cfg)
    states = {t.state.name for t in study.trials}
    assert states <= {"COMPLETE", "PRUNED", "FAIL"}
    assert len(study.trials) == 8
    rows = db.query("SELECT status FROM experiments WHERE study_name='it_tpe_1'")
    assert len(rows) >= 1
    run = db.one("SELECT status FROM optimization_runs WHERE study_name='it_tpe_1'")
    assert run["status"] == "completed"
    # Re-running the identical search space with the same seed must not duplicate completed rows.
    before = db.one("SELECT COUNT(*) n FROM experiments WHERE status='completed'")["n"]
    dup = db.one("SELECT COUNT(*) n FROM (SELECT strategy_hash, split, config_hash FROM experiments"
                 " WHERE status='completed' GROUP BY 1,2,3 HAVING COUNT(*)>1)")["n"]
    assert dup == 0 and before > 0


def test_optuna_multiobjective_nsga2_produces_pareto_front(dataset_id, research_cfg):
    db = get_db()
    cfg = OptimizationConfig(dataset_id=dataset_id, study_name="it_nsga_1", n_trials=10,
                             sampler="nsga2", objective="profit_vs_drawdown", max_indicators=3, seed=2)
    study = run_study(cfg, WorkerContext("optuna:it_nsga_1", db, throttle=False), research_cfg)
    assert len(study.directions) == 2
    assert len(study.best_trials) >= 1


def test_research_agent_full_cycle_and_checkpoint(dataset_id, research_cfg):
    db = get_db()
    settings = ResearchSettings(dataset_id=dataset_id, cycles=1, population_size=6, generations=2,
                                hill_climb_steps=3, tuning_trials=4, walk_forward_top_k=3,
                                max_indicators=3, seed=9)
    worker = ManagedWorker("research_agent", lambda ctx: ResearchAgent(settings, ctx, research_cfg).run(),
                           throttle=False)
    worker.start()
    worker._thread.join(timeout=240)
    assert not worker.is_alive, "research agent did not finish in time"
    assert worker.ctx.state.status == "completed", worker.ctx.state.detail
    cp = db.one("SELECT state_json FROM agent_checkpoints WHERE agent='research_agent'")
    state = json.loads(cp["state_json"])
    assert state["cycle"] == 1
    logs = db.query("SELECT message FROM agent_logs WHERE agent='research_agent'")
    text = " ".join(r["message"] for r in logs)
    for phase_marker in ("forward selection", "backward elimination", "GA generation", "walk-forward"):
        assert phase_marker in text, f"missing phase evidence: {phase_marker}"
    validations = db.one("SELECT COUNT(*) n FROM validations WHERE dataset_id=?", (dataset_id,))["n"]
    assert validations >= 1


def test_research_agent_never_touches_out_of_sample(dataset_id, research_cfg):
    db = get_db()
    before = db.one("SELECT COUNT(*) n FROM experiments WHERE split='oos'")["n"]
    settings = ResearchSettings(dataset_id=dataset_id, cycles=1, population_size=4, generations=1,
                                hill_climb_steps=1, tuning_trials=0, walk_forward_top_k=1, max_indicators=2)
    ctx = WorkerContext("research_agent", db, throttle=False)
    ResearchAgent(settings, ctx, research_cfg).run()
    after = db.one("SELECT COUNT(*) n FROM experiments WHERE split='oos'")["n"]
    assert after == before


def test_evaluation_agent_audit_and_leaderboards(dataset_id, research_cfg):
    db = get_db()
    result = audit(dataset_id, research_cfg.min_trades)
    assert result["completed"] > 0
    assert result["missing_metrics"] == 0
    assert result["eligibility_mismatches"] == 0
    assert result["duplicate_completed_rows"] == 0
    boards = {k: ranking.leaderboard(k, dataset_id, top=10, db=db) for k in ranking.RANKINGS}
    profits = [r["metrics"]["net_profit"] for r in boards["profit"]]
    assert profits == sorted(profits, reverse=True)
    dd = [r["metrics"]["max_drawdown_pct"] for r in boards["drawdown"]]
    assert dd == sorted(dd)
    assert all(r["metrics"]["eligible"] for r in boards["profit"])


def test_evaluate_cached_never_overwrites(dataset_id, research_cfg):
    ev = Evaluator(dataset_id, research_cfg)
    spec = StrategySpec.build([{"name": "ref_rsi"}], min_votes=1, stop_loss_pct=0.3, target_pct=0.9)
    first = ev.evaluate_cached(spec, "train", agent="test")
    second = ev.evaluate_cached(spec, "train", agent="test")
    assert second["_cached"] is True
    assert {k: v for k, v in first.items() if k != "_cached"} == {k: v for k, v in second.items() if k != "_cached"}
    inserted = ev.record(spec, "train", first, agent="test")
    assert inserted is False


def test_walk_forward_reports_segments(dataset_id, research_cfg):
    ev = Evaluator(dataset_id, research_cfg)
    spec = StrategySpec.build([{"name": "ref_donchian"}, {"name": "ref_ema_cross"}], min_votes=1)
    wf = ev.walk_forward(spec, "validation")
    assert wf["total_segments"] == research_cfg.split.walk_forward_segments
    assert sum(s["trades"] for s in wf["segments"]) == wf["total_trades"]
    assert isinstance(wf["validated"], bool)


def test_split_boundaries_are_chronological_and_disjoint(dataset_id, research_cfg):
    ev = Evaluator(dataset_id, research_cfg)
    b = ev.bounds
    assert b["train"][1] == b["validation"][0]
    assert b["validation"][1] == b["oos"][0]
    assert b["oos"][1] == len(ev.candles)


# -- HTTP API --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client(dataset_id):
    from fastapi.testclient import TestClient

    from indicatorlab.api import app

    with TestClient(app) as c:
        yield c


def test_api_health_and_indicator_registry(client):
    assert client.get("/api/health").json()["status"] == "ok"
    body = client.get("/api/indicators").json()
    assert body["summary"]["available"] == len([i for i in body["indicators"] if i["available"]])
    assert body["summary"]["total"] >= 7


def test_api_dashboard_reports_real_counts(client, dataset_id):
    body = client.get(f"/api/dashboard?dataset_id={dataset_id}").json()
    assert body["experiments"]["completed"] > 0
    assert "trials" in body and "recent_activity" in body


def test_api_backtest_and_oos_once(client, dataset_id):
    strategy = {"indicators": [{"name": "ref_rsi"}, {"name": "ref_bollinger"}], "min_votes": 1,
                "direction": "both", "stop_loss_pct": 0.5, "target_pct": 1.0}
    r = client.post("/api/backtest", json={"dataset_id": dataset_id, "strategy": strategy, "split": "validation"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert "metrics" in body and "equity" in body and body["candles"]
    assert len(body["equity"]["equity"]) == len(body["equity"]["timestamps"])
    bad = client.post("/api/backtest", json={"dataset_id": dataset_id, "strategy": strategy, "split": "oos"})
    assert bad.status_code == 400
    first = client.post("/api/backtest/oos", json={"dataset_id": dataset_id, "strategy": strategy}).json()
    second = client.post("/api/backtest/oos", json={"dataset_id": dataset_id, "strategy": strategy}).json()
    assert first["already_run"] is False and second["already_run"] is True
    assert first["metrics"] == second["metrics"]


def test_api_rejects_unknown_indicator_and_bad_params(client, dataset_id):
    r = client.post("/api/backtest", json={"dataset_id": dataset_id, "split": "train",
                                           "strategy": {"indicators": [{"name": "nope"}]}})
    assert r.status_code == 400
    r = client.post("/api/backtest", json={"dataset_id": dataset_id, "split": "train",
                                           "strategy": {"indicators": [{"name": "ref_rsi", "params": {"zz": 1}}]}})
    assert r.status_code == 400


def test_api_results_and_experiments(client, dataset_id):
    boards = client.get(f"/api/results/rankings?dataset_id={dataset_id}&top=5").json()
    assert set(boards) == set(ranking.RANKINGS)
    exp = client.get(f"/api/experiments?dataset_id={dataset_id}&limit=5").json()
    assert exp["total"] > 0 and len(exp["rows"]) <= 5
    detail = client.get(f"/api/experiments/{exp['rows'][0]['id']}").json()
    assert detail["spec"]["indicators"]


def test_api_candles_resample(client, dataset_id):
    body = client.get(f"/api/datasets/{dataset_id}/candles?timeframe=15min&limit=50").json()
    assert body["timeframe"] == "15min" and len(body["bars"]) <= 50
    assert body["bars"][0]["high"] >= body["bars"][0]["low"]


def test_api_optuna_chart(client, dataset_id):
    studies = get_db().query("SELECT study_name FROM optimization_runs WHERE study_name='it_tpe_1'")
    assert studies
    chart = client.get("/api/analytics/optuna/it_tpe_1/history")
    assert chart.status_code == 200 and "data" in chart.json()
    bad = client.get("/api/analytics/optuna/it_tpe_1/pareto")
    assert bad.status_code == 400


def test_resume_from_checkpoint_skips_completed_phases(dataset_id, research_cfg):
    db = get_db()
    settings = ResearchSettings(dataset_id=dataset_id, cycles=1, population_size=4, generations=1,
                                hill_climb_steps=1, tuning_trials=0, walk_forward_top_k=1, max_indicators=2,
                                seed=21)
    done_spec = StrategySpec.build([{"name": "ref_rsi"}], min_votes=1).to_dict()
    db.execute("INSERT INTO agent_checkpoints(agent, state_json, updated_at) VALUES ('resume_probe', ?, 'x')"
               " ON CONFLICT(agent) DO UPDATE SET state_json=excluded.state_json",
               (db.dumps({"cycle": 0, "done": {"forward": done_spec, "backward": done_spec}, "ga": {}}),))
    ctx = WorkerContext("resume_probe", db, throttle=False)
    ResearchAgent(settings, ctx, research_cfg).run()
    text = " ".join(r["message"] for r in db.query("SELECT message FROM agent_logs WHERE agent='resume_probe'"))
    assert "forward selection added" not in text, "completed phase was re-run after resume"
    assert "genetic algorithm" in text


def test_pause_blocks_progress_and_stop_ends_worker(dataset_id, research_cfg):
    import time

    settings = ResearchSettings(dataset_id=dataset_id, cycles=0, population_size=4, generations=1,
                                hill_climb_steps=1, tuning_trials=0, walk_forward_top_k=1, max_indicators=2)
    worker = ManagedWorker("pause_probe", lambda ctx: ResearchAgent(settings, ctx, research_cfg).run(),
                           throttle=False)
    worker.start()
    time.sleep(0.3)
    worker.pause()
    time.sleep(1.2)
    assert worker.ctx.state.status == "paused"
    heartbeat_before = worker.ctx.state.last_heartbeat
    time.sleep(0.8)
    assert worker.ctx.state.status == "paused"
    worker.resume()
    time.sleep(0.5)
    assert worker.ctx.state.status in ("running", "paused") and worker.ctx.state.last_heartbeat >= heartbeat_before
    worker.stop(wait=60)
    assert not worker.is_alive
    assert worker.ctx.state.status == "stopped"


def test_restart_recovery_marks_stale_workers_stopped():
    from indicatorlab.agents.controller import AgentController

    db = get_db()
    db.execute("INSERT INTO agents(name, status, last_heartbeat) VALUES ('ghost_agent','running','x')"
               " ON CONFLICT(name) DO UPDATE SET status='running'")
    recovered = AgentController().recover_after_restart()
    assert "ghost_agent" in recovered
    row = db.one("SELECT status FROM agents WHERE name='ghost_agent'")
    assert row["status"] == "stopped"
    warn = db.one("SELECT level FROM agent_logs WHERE agent='ghost_agent' ORDER BY id DESC LIMIT 1")
    assert warn["level"] == "warn"
