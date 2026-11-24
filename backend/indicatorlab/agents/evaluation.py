"""Agent 2 - Evaluation & Ranking Agent.

Runs alongside Agent 1 and works only from stored experiments. On each pass it:
  * processes experiments recorded since its last pass (a watermark on experiments.id),
  * audits the ledger (failures, pruned trials, missing metrics, eligibility mismatches,
    duplicate completed rows), and
  * recomputes every leaderboard and publishes the snapshot to its checkpoint.

It never edits or deletes an experiment. The database enforces one completed row per
strategy, dataset, split and config, so results cannot be overwritten.
"""
from __future__ import annotations

import json

from ..db import get_db
from ..ranking import RANKINGS, leaderboard, summary
from .runtime import WorkerContext

AGENT_NAME = "evaluation_agent"


def audit(dataset_id: int, min_trades: int) -> dict:
    db = get_db()
    counts = {r["status"]: r["n"] for r in db.query(
        "SELECT status, COUNT(*) n FROM experiments WHERE dataset_id=? GROUP BY status", (dataset_id,))}
    rows = db.query(
        "SELECT metrics_json, eligible FROM experiments WHERE dataset_id=? AND status='completed'",
        (dataset_id,))
    missing_metrics = 0
    eligibility_mismatch = 0
    for r in rows:
        if not r["metrics_json"]:
            missing_metrics += 1
            continue
        metrics = json.loads(r["metrics_json"])
        if bool(r["eligible"]) != (metrics.get("trades", 0) >= min_trades):
            eligibility_mismatch += 1
    dup = db.one(
        "SELECT COUNT(*) n FROM (SELECT strategy_hash, split, config_hash FROM experiments"
        " WHERE dataset_id=? AND status='completed' GROUP BY strategy_hash, split, config_hash"
        " HAVING COUNT(*) > 1)", (dataset_id,))
    return {
        "completed": counts.get("completed", 0),
        "failed": counts.get("failed", 0),
        "pruned": counts.get("pruned", 0),
        "missing_metrics": missing_metrics,
        "eligibility_mismatches": eligibility_mismatch,
        "duplicate_completed_rows": int(dup["n"]) if dup else 0,
        "status": "ok" if missing_metrics == 0 and eligibility_mismatch == 0
                  and (dup is None or dup["n"] == 0) else "attention",
    }


def evaluation_loop(ctx: WorkerContext, dataset_id: int, min_trades: int, interval: float = 3.0) -> None:
    db = get_db()
    ctx.set_task("waiting for experiments", algorithm="Ledger audit + leaderboards", progress=0.0)
    watermark = int((db.one("SELECT COALESCE(MAX(id),0) m FROM experiments") or {"m": 0})["m"])
    while True:
        ctx.checkpoint()
        new_rows = db.one("SELECT COUNT(*) n FROM experiments WHERE id > ? AND dataset_id = ?",
                          (watermark, dataset_id))
        processed = int(new_rows["n"]) if new_rows else 0
        if processed:
            ctx.count_trial(processed)
            top = db.one("SELECT COALESCE(MAX(id),0) m FROM experiments")
            watermark = int(top["m"])
        boards = {}
        for key in RANKINGS:
            boards[key] = [
                {"rank": r["rank"], "strategy_hash": r["strategy_hash"],
                 "description": _describe(r["spec"]), "metrics": r["metrics"],
                 "validated": r["validated"]}
                for r in leaderboard(key, dataset_id, top=10, db=db)]
        snapshot = {
            "updated_at": _now(), "summary": summary(dataset_id, db=db),
            "audit": audit(dataset_id, min_trades), "leaderboards": boards,
            "new_experiments_last_pass": processed,
        }
        ctx.save_checkpoint(snapshot)
        ctx.set_task("ranking experiments", progress=1.0 if snapshot["summary"]["eligible_strategies"] else 0.0)
        ctx.state.detail["audit_status"] = snapshot["audit"]["status"]
        ctx.heartbeat()
        ctx.sleep(interval)


def _describe(spec: dict) -> str:
    parts = [f"{i['name']}({', '.join(f'{k}={v}' for k, v in sorted(i['params'].items()))})"
             for i in spec["indicators"]]
    return (f"{' + '.join(parts)} | votes>={spec['min_votes']} | {spec['direction']} | "
            f"SL {spec['stop_loss_pct']}% TP {spec['target_pct']}%")


def _now() -> str:
    from ..db import utcnow

    return utcnow()
