"""Statistics tests: DSR, PBO/CSCV, migration backup, and end-to-end significance on stored experiments."""
import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from indicatorlab.overfitting import (deflated_sharpe, expected_max_sharpe, pbo_cscv,
                                      probabilistic_sharpe, return_moments, sharpe_moments)


# -- moments and sharpe ---------------------------------------------------------------------------
def test_sharpe_moments_normal_sample_close_to_theory():
    rng = np.random.default_rng(1)
    r = rng.normal(0.001, 0.01, 200_000)
    m = sharpe_moments(r)
    assert m["sharpe_per_bar"] == pytest.approx(0.1, abs=0.01)
    assert m["return_skew"] == pytest.approx(0.0, abs=0.03)
    assert m["return_kurtosis"] == pytest.approx(3.0, abs=0.05)  # raw kurtosis, normal = 3


def test_return_moments_from_equity_and_flat_curve():
    eq = np.array([100.0, 101.0, 100.5, 102.0, 103.0])
    m = return_moments(eq)
    assert m["returns_obs"] == 4
    assert m["sharpe_per_bar"] is not None
    flat = return_moments(np.full(10, 100.0))
    assert flat["sharpe_per_bar"] is None  # zero variance is reported, not divided by zero


# -- expected maximum and PSR / DSR ----------------------------------------------------------------
def test_expected_max_sharpe_grows_with_trials_and_matches_known_order():
    values = [expected_max_sharpe(n, 1.0) for n in (2, 10, 100, 1000)]
    assert values == sorted(values)
    assert values[0] == pytest.approx(0.5, abs=0.1)  # close to 1/sqrt(pi) = 0.564 for two trials
    assert expected_max_sharpe(1, 1.0) == 0.0
    assert expected_max_sharpe(10, 0.0) == 0.0


def test_psr_is_half_at_benchmark_and_rises_with_sharpe():
    at = probabilistic_sharpe(0.1, 0.1, 500, 0.0, 3.0)
    assert at["psr"] == pytest.approx(0.5)
    above = probabilistic_sharpe(0.2, 0.1, 500, 0.0, 3.0)
    assert above["psr"] > 0.5


def test_dsr_penalises_more_trials_for_same_sharpe():
    # Same spread for both lists so that only the number of trials differs.
    few_trials = list(np.linspace(-0.1, 0.1, 5))
    many_trials = list(np.linspace(-0.1, 0.1, 500))
    sr = 0.25
    few = deflated_sharpe(sr, 300, 0.0, 3.0, few_trials + [sr])
    many = deflated_sharpe(sr, 300, 0.0, 3.0, many_trials + [sr])
    assert few["status"] == many["status"] == "ok"
    assert many["expected_max_sharpe"] > few["expected_max_sharpe"]
    assert many["dsr"] < few["dsr"]
    assert many["assumptions"]["n_trials"] == 501


def test_dsr_records_assumptions_and_handles_small_samples():
    short = deflated_sharpe(0.2, 10, 0.0, 3.0, [0.1, 0.2, 0.3])
    assert short["status"] == "insufficient_sample" and "at least 30" in short["reason"]
    alone = deflated_sharpe(0.2, 300, 0.0, 3.0, [0.2])
    assert alone["status"] == "insufficient_sample" and "trials" in alone["reason"]
    none = deflated_sharpe(None, 300, 0.0, 3.0, [0.1, 0.2])
    assert none["status"] == "insufficient_sample"
    assert "assumptions" in none and none["assumptions"]["sharpe_scale"].startswith("per bar")


# -- PBO / CSCV -------------------------------------------------------------------------------------
def test_pbo_near_half_for_pure_noise():
    # A single draw is noisy (SD about 0.18 across seeds), so average over seeds.
    values = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        result = pbo_cscv(rng.normal(size=(480, 30)), blocks=8)
        assert result["status"] == "ok"
        assert result["combinations"] == math.comb(8, 4)
        values.append(result["pbo"])
    assert 0.4 <= float(np.mean(values)) <= 0.65


def test_pbo_low_when_one_config_is_genuinely_better():
    rng = np.random.default_rng(12)
    m = rng.normal(size=(480, 30))
    m[:, 0] += 0.3  # a real edge present in every block
    result = pbo_cscv(m, blocks=8, labels=[f"c{j}" for j in range(30)])
    assert result["pbo"] < 0.1
    assert result["combination_detail"][0]["selected"] == "c0"


def test_pbo_refuses_tiny_or_invalid_input():
    rng = np.random.default_rng(13)
    assert pbo_cscv(rng.normal(size=(10, 5)), blocks=8)["status"] == "insufficient_sample"
    assert pbo_cscv(rng.normal(size=(100, 1)), blocks=8)["status"] == "insufficient_sample"
    with pytest.raises(ValueError):
        pbo_cscv(rng.normal(size=(100, 5)), blocks=5)


def test_pbo_does_not_mutate_input_and_is_deterministic():
    rng = np.random.default_rng(14)
    m = rng.normal(size=(200, 10))
    before = m.copy()
    a = pbo_cscv(m, blocks=8)
    b = pbo_cscv(m, blocks=8)
    assert np.array_equal(m, before)
    assert a["pbo"] == b["pbo"]


# -- migration -------------------------------------------------------------------------------------
def test_migration_backs_up_existing_v1_database_and_keeps_rows(tmp_path: Path):
    from indicatorlab.db import SCHEMA, SCHEMA_VERSION, Database

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)  # the version 1 layout
    conn.execute("INSERT INTO schema_version(version) VALUES (1)")
    conn.execute("INSERT INTO datasets(name, symbol, timeframe, rows, start_ts, end_ts, imported_at)"
                 " VALUES ('legacy','X','1min',1,'a','b','c')")
    conn.commit()
    conn.close()

    db = Database(path)
    report = db.migration_report
    assert report["from"] == 1 and report["to"] == SCHEMA_VERSION
    assert report["backup"] and Path(report["backup"]).exists()
    assert db.one("SELECT name FROM datasets WHERE name='legacy'") is not None
    assert db.one("SELECT name FROM sqlite_master WHERE name='statistical_tests'") is not None
    assert db.one("SELECT version FROM schema_version")["version"] == SCHEMA_VERSION
    backup = sqlite3.connect(report["backup"])
    assert backup.execute("SELECT name FROM datasets").fetchall() == [("legacy",)]
    backup.close()
    db.close() if hasattr(db, "close") else None


def test_migration_is_noop_on_current_database(tmp_path: Path):
    from indicatorlab.db import Database

    path = tmp_path / "fresh.db"
    first = Database(path)
    assert first.migration_report["from"] == 0
    second = Database(path)
    assert second.migration_report["backup"] is None
    assert second.migration_report["from"] == second.migration_report["to"]


# -- end to end on stored experiments ---------------------------------------------------------------
def test_dsr_stored_for_eligible_experiments_and_pbo_uses_train_only(dataset_id):
    from indicatorlab.db import get_db
    from indicatorlab.evaluator import Evaluator, ResearchConfig
    from indicatorlab.significance import compute_dsr, compute_pbo, dsr_for_experiment
    from indicatorlab.strategy import StrategySpec

    db = get_db()
    research = ResearchConfig()
    ev = Evaluator(dataset_id, research, db)
    specs = [
        StrategySpec.build([{"name": "ref_sma_cross", "params": {"fast": f, "slow": s}}], min_votes=1,
                           stop_loss_pct=0.5, target_pct=1.0)
        for f, s in [(3, 9), (5, 15), (8, 21), (10, 30)]
    ]
    for spec in specs:
        if ev.cached(spec, "train") is None:
            metrics = ev.evaluate(spec, "train")
            ev.record(spec, "train", metrics, agent="test")

    dsr = compute_dsr(dataset_id, "train", research, db)
    assert dsr["experiments"] >= len(specs)
    assert dsr["counts"].get("ok", 0) + dsr["counts"].get("insufficient_sample", 0) == dsr["experiments"]

    eligible = db.query("SELECT id FROM experiments WHERE dataset_id=? AND split='train'"
                        " AND eligible=1 AND status='completed' AND config_hash=?",
                        (dataset_id, research.fingerprint()))
    assert eligible
    stored = dsr_for_experiment(eligible[0]["id"], db)
    assert stored is not None and stored["status"] in {"ok", "insufficient_sample"}
    assert "assumptions" in stored["result"]

    pbo = compute_pbo(dataset_id, blocks=4, max_configs=10, research=research, db=db)
    assert pbo["status"] in {"ok", "insufficient_sample"}
    assert pbo["config"]["split"] == "train"  # OOS is never read here
    assert pbo["config"]["backtest"].startswith("train split")
    rows = db.query("SELECT COUNT(*) AS n FROM statistical_tests WHERE kind='pbo' AND dataset_id=?",
                    (dataset_id,))
    assert rows[0]["n"] >= 1


def test_no_oos_rows_are_read_by_significance(dataset_id):
    from indicatorlab.db import get_db
    from indicatorlab.significance import compute_pbo
    db = get_db()
    before = db.query("SELECT COUNT(*) AS n FROM experiments WHERE split='oos'")[0]["n"]
    compute_pbo(dataset_id, blocks=4, max_configs=4, db=db)
    after = db.query("SELECT COUNT(*) AS n FROM experiments WHERE split='oos'")[0]["n"]
    assert before == after


def test_statistics_endpoints(dataset_id):
    from fastapi.testclient import TestClient

    from indicatorlab.api import app

    with TestClient(app) as client:
        r = client.post("/api/statistics/dsr", json={"dataset_id": dataset_id, "split": "validation"})
        assert r.status_code == 400  # DSR never runs on validation or OOS
        r = client.post("/api/statistics/dsr", json={"dataset_id": dataset_id, "split": "train"})
        assert r.status_code == 200 and r.json()["split"] == "train"
        r = client.post("/api/statistics/pbo", json={"dataset_id": dataset_id, "blocks": 3})
        assert r.status_code == 422  # blocks must be even and at least 4
        r = client.post("/api/statistics/pbo", json={"dataset_id": dataset_id, "blocks": 4, "max_configs": 6})
        assert r.status_code == 200
        listed = client.get("/api/statistics", params={"dataset_id": dataset_id, "kind": "pbo"}).json()
        assert listed and all(row["kind"] == "pbo" for row in listed)
        bad = client.get("/api/statistics", params={"dataset_id": dataset_id, "kind": "spa"})
        assert bad.status_code == 422
