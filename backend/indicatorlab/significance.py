"""Database-backed significance tests: Deflated Sharpe Ratio and PBO (CSCV).

Both tests read only completed experiments and the train split, so the OOS period and the
final test data are never used to select or score anything here.

Results are written to ``statistical_tests``. DSR rows are derived data and are recomputed when
the trial population changes (one row per experiment, replaced atomically). PBO rows are
append-only snapshots: every run is kept with its configurations and blocks.
"""
from __future__ import annotations

import json

import numpy as np

from .db import Database, get_db, utcnow
from .evaluator import Evaluator, ResearchConfig
from .overfitting import deflated_sharpe, pbo_cscv
from .strategy import spec_from_dict


def _eligible_rows(db: Database, dataset_id: int, split: str, config_hash: str) -> list[dict]:
    rows = db.query(
        "SELECT id, strategy_hash, spec_json, metrics_json FROM experiments"
        " WHERE dataset_id=? AND split=? AND config_hash=? AND status='completed' AND eligible=1"
        " ORDER BY id", (dataset_id, split, config_hash))
    for r in rows:
        r["metrics"] = json.loads(r.pop("metrics_json"))
    return rows


def compute_dsr(dataset_id: int, split: str = "train", research: ResearchConfig | None = None,
                db: Database | None = None) -> dict:
    """Compute DSR for every eligible completed experiment on this dataset and split."""
    db = db or get_db()
    research = research or ResearchConfig()
    config_hash = research.fingerprint()
    rows = _eligible_rows(db, dataset_id, split, config_hash)
    trial_sr = [r["metrics"].get("sharpe_per_bar") for r in rows]
    legacy = sum(1 for s in trial_sr if s is None)
    now = utcnow()
    counts = {"ok": 0, "insufficient_sample": 0}
    with db.transaction() as cur:
        cur.execute("DELETE FROM statistical_tests WHERE kind='dsr' AND dataset_id=? AND split=?"
                    " AND experiment_id IN (SELECT id FROM experiments WHERE config_hash=?)",
                    (dataset_id, split, config_hash))
        for r in rows:
            m = r["metrics"]
            res = deflated_sharpe(m.get("sharpe_per_bar"), int(m.get("returns_obs") or 0),
                                  float(m.get("return_skew") or 0.0), float(m.get("return_kurtosis") or 3.0),
                                  [s for s in trial_sr if s is not None])
            if m.get("sharpe_per_bar") is None:
                res = {"status": "insufficient_sample", "reason": "experiment has no return statistics"
                       " (recorded before the statistics migration, or flat equity)",
                       "assumptions": res.get("assumptions", {})}
            res["legacy_excluded_trials"] = legacy
            counts[res["status"]] = counts.get(res["status"], 0) + 1
            cur.execute(
                "INSERT INTO statistical_tests(created_at, kind, dataset_id, split, experiment_id,"
                " strategy_hash, config_json, seed, status, result_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (now, "dsr", dataset_id, split, r["id"], r["strategy_hash"],
                 json.dumps({"config_hash": config_hash}), None, res["status"], json.dumps(res, default=float)))
    return {"kind": "dsr", "dataset_id": dataset_id, "split": split, "experiments": len(rows),
            "trials_with_statistics": len(rows) - legacy, "legacy_excluded": legacy, "counts": counts}


def compute_pbo(dataset_id: int, *, blocks: int = 8, max_configs: int = 40,
                research: ResearchConfig | None = None, db: Database | None = None) -> dict:
    """PBO over the top eligible train configurations. Uses the train split only."""
    db = db or get_db()
    research = research or ResearchConfig()
    if max_configs < 2:
        raise ValueError("max_configs must be at least 2")
    rows = _eligible_rows(db, dataset_id, "train", research.fingerprint())
    rows.sort(key=lambda r: float(r["metrics"]["net_profit"]), reverse=True)
    seen, chosen = set(), []
    for r in rows:  # one entry per strategy
        if r["strategy_hash"] not in seen:
            seen.add(r["strategy_hash"])
            chosen.append(r)
        if len(chosen) == max_configs:
            break
    ev = Evaluator(dataset_id, research, db)
    columns, labels = [], []
    for r in chosen:
        spec = spec_from_dict(json.loads(r["spec_json"]))
        res, _ = ev.simulate(spec, "train")
        eq = np.asarray(res.equity, dtype=float)
        prev = eq[:-1]
        ok = prev > 0
        ret = np.zeros(max(len(eq) - 1, 0))
        ret[ok] = np.diff(eq)[ok] / prev[ok]
        columns.append(ret)
        labels.append(r["strategy_hash"])
    if not columns:
        result = {"status": "insufficient_sample", "reason": "no eligible train experiments",
                  "blocks": blocks, "configs": 0}
    else:
        length = min(len(c) for c in columns)
        matrix = np.column_stack([c[:length] for c in columns]) if length else np.zeros((0, 0))
        result = pbo_cscv(matrix, blocks=blocks, labels=labels) if length else \
            {"status": "insufficient_sample", "reason": "empty equity curves", "blocks": blocks,
             "configs": len(columns)}
    config = {"dataset_id": dataset_id, "split": "train", "blocks": blocks, "max_configs": max_configs,
              "config_hash": research.fingerprint(), "strategy_hashes": labels,
              "backtest": "train split, chronological blocks, OOS not used"}
    db.execute("INSERT INTO statistical_tests(created_at, kind, dataset_id, split, experiment_id,"
               " strategy_hash, config_json, seed, status, result_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
               (utcnow(), "pbo", dataset_id, "train", None, None, json.dumps(config), None,
                result["status"], json.dumps(result, default=float)))
    return {"kind": "pbo", "dataset_id": dataset_id, "status": result["status"], "result": result,
            "config": config}


def list_tests(dataset_id: int, kind: str | None = None, db: Database | None = None) -> list[dict]:
    db = db or get_db()
    sql = "SELECT * FROM statistical_tests WHERE dataset_id=?"
    params: list = [dataset_id]
    if kind:
        sql += " AND kind=?"
        params.append(kind)
    sql += " ORDER BY id DESC LIMIT 500"
    out = []
    for r in db.query(sql, params):
        r["config"] = json.loads(r.pop("config_json"))
        r["result"] = json.loads(r.pop("result_json"))
        out.append(r)
    return out


def dsr_for_experiment(experiment_id: int, db: Database | None = None) -> dict | None:
    db = db or get_db()
    row = db.one("SELECT * FROM statistical_tests WHERE kind='dsr' AND experiment_id=?"
                 " ORDER BY id DESC LIMIT 1", (experiment_id,))
    if not row:
        return None
    return {"status": row["status"], "result": json.loads(row["result_json"]), "created_at": row["created_at"]}

