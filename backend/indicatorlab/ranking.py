"""Leaderboards computed from stored experiments only (Agent 2's source of truth)."""
from __future__ import annotations

import json

from .db import Database, get_db

# key -> (metric, higher_is_better, label)
RANKINGS: dict[str, tuple[str | None, bool, str]] = {
    "profit": ("net_profit", True, "Highest net profit"),
    "win_rate": ("win_rate", True, "Highest win rate"),
    "profit_factor": ("profit_factor", True, "Highest profit factor"),
    "drawdown": ("max_drawdown_pct", False, "Lowest maximum drawdown"),
    "risk_adjusted": ("recovery_factor", True, "Best risk-adjusted (net profit / max drawdown)"),
    "walk_forward": (None, True, "Best walk-forward validated strategies"),
}


def _candidates(db: Database, dataset_id: int, split: str) -> list[dict]:
    rows = db.query(
        "SELECT e.id, e.strategy_hash, e.spec_json, e.metrics_json, e.created_at, e.study_name,"
        " e.trial_number, e.agent FROM experiments e"
        " WHERE e.dataset_id=? AND e.split=? AND e.status='completed' AND e.eligible=1 ORDER BY e.id",
        (dataset_id, split))
    latest: dict[str, dict] = {}
    for r in rows:  # later rows win: the most recent evaluation of each strategy is shown
        latest[r["strategy_hash"]] = {
            "id": r["id"], "strategy_hash": r["strategy_hash"],
            "spec": json.loads(r["spec_json"]), "metrics": json.loads(r["metrics_json"]),
            "created_at": r["created_at"], "study_name": r["study_name"],
            "trial_number": r["trial_number"], "agent": r["agent"],
        }
    return list(latest.values())


def _validations(db: Database, dataset_id: int) -> dict[str, dict]:
    rows = db.query(
        "SELECT v.* FROM validations v JOIN (SELECT strategy_hash h, MAX(id) mid FROM validations"
        " WHERE dataset_id=? GROUP BY strategy_hash) m ON v.id = m.mid", (dataset_id,))
    return {r["strategy_hash"]: {
        "validated": bool(r["validated"]), "profitable_segments": r["profitable_segments"],
        "total_segments": r["total_segments"], "validation_net_profit": r["validation_net_profit"],
        "checked_at": r["created_at"]} for r in rows}


def leaderboard(ranking: str, dataset_id: int, *, top: int = 10, split: str = "train",
                db: Database | None = None) -> list[dict]:
    if ranking not in RANKINGS:
        raise ValueError(f"ranking must be one of {sorted(RANKINGS)}")
    db = db or get_db()
    metric, higher, _ = RANKINGS[ranking]
    rows = _candidates(db, dataset_id, split)
    validations = _validations(db, dataset_id)
    for r in rows:
        r["validation"] = validations.get(r["strategy_hash"])
        r["validated"] = bool(r["validation"] and r["validation"]["validated"])
    if ranking == "walk_forward":
        rows = [r for r in rows if r["validation"] is not None]
        rows.sort(key=lambda r: (r["validated"], r["validation"]["validation_net_profit"]), reverse=True)
    else:
        rows.sort(key=lambda r: (r["metrics"][metric], r["metrics"]["trades"]), reverse=higher)
        if ranking == "drawdown":
            rows = [r for r in rows if r["metrics"]["net_profit"] > 0]
    out = []
    for position, r in enumerate(rows[:top], start=1):
        out.append({"rank": position, **r})
    return out


def summary(dataset_id: int, split: str = "train", db: Database | None = None) -> dict:
    db = db or get_db()
    candidates = _candidates(db, dataset_id, split)
    validations = _validations(db, dataset_id)
    return {
        "eligible_strategies": len(candidates),
        "validated_strategies": sum(1 for v in validations.values() if v["validated"]),
        "validation_checks": sum(1 for _ in validations),
        "best_net_profit": max((c["metrics"]["net_profit"] for c in candidates), default=None),
        "best_win_rate": max((c["metrics"]["win_rate"] for c in candidates), default=None),
    }
