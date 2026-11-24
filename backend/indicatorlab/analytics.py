"""Chart data for the Analytics page. Optuna's own visualizations are reused where they apply."""
from __future__ import annotations

import json

import numpy as np
import optuna
import optuna.visualization as ov

from .config import OPTUNA_STORAGE
from .db import get_db
from .evaluator import Evaluator, ResearchConfig
from .strategy import StrategySpec

OPTUNA_CHARTS = {
    "history": ov.plot_optimization_history,
    "importance": ov.plot_param_importances,
    "contour": ov.plot_contour,
    "slice": ov.plot_slice,
    "parallel": ov.plot_parallel_coordinate,
    "pareto": ov.plot_pareto_front,
}


def optuna_chart(study_name: str, chart: str) -> dict:
    if chart not in OPTUNA_CHARTS:
        raise ValueError(f"chart must be one of {sorted(OPTUNA_CHARTS)}")
    study = optuna.load_study(study_name=study_name, storage=OPTUNA_STORAGE)
    if chart == "pareto" and len(study.directions) < 2:
        raise ValueError("Pareto front needs a multi-objective study")
    if chart != "pareto" and len(study.directions) > 1:
        raise ValueError("this chart needs a single-objective study")
    fig = OPTUNA_CHARTS[chart](study)
    return json.loads(fig.to_json())


def equity_payload(res, index, metrics: dict, max_points: int = 3000) -> dict:
    """Downsampled equity and drawdown curves for charting. Values come from the simulation."""
    eq = np.asarray(res.equity, dtype=float)
    if eq.size == 0:
        return {"timestamps": [], "equity": [], "drawdown": [], "metrics": metrics}
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    idx = np.linspace(0, len(eq) - 1, min(max_points, len(eq))).astype(int)
    return {
        "timestamps": [index[i].isoformat() for i in idx],
        "equity": [round(float(eq[i]), 2) for i in idx],
        "drawdown": [round(float(dd[i]), 2) for i in idx],
        "metrics": metrics,
    }


def equity_series(dataset_id: int, spec: StrategySpec, split: str,
                  research: ResearchConfig | None = None) -> dict:
    ev = Evaluator(dataset_id, research)
    res, metrics = ev.simulate(spec, split)
    payload = equity_payload(res, ev.split_frame(split).index, metrics)
    return {"split": split, **payload, "trades": res.trades}


def walk_forward_board(dataset_id: int) -> list[dict]:
    db = get_db()
    rows = db.query(
        "SELECT v.strategy_hash, v.created_at, v.profitable_segments, v.total_segments,"
        " v.validation_net_profit, v.validated, v.segments_json FROM validations v"
        " WHERE v.dataset_id=? ORDER BY v.id DESC", (dataset_id,))
    seen, out = set(), []
    for r in rows:
        if r["strategy_hash"] in seen:
            continue
        seen.add(r["strategy_hash"])
        out.append({**r, "validated": bool(r["validated"]),
                    "segments": json.loads(r.pop("segments_json"))})
    return out


def train_vs_validation(dataset_id: int) -> list[dict]:
    db = get_db()
    rows = db.query(
        "SELECT e.strategy_hash, e.metrics_json, v.validation_net_profit, v.validated FROM experiments e"
        " JOIN (SELECT strategy_hash h, MAX(id) mid FROM validations WHERE dataset_id=? GROUP BY strategy_hash) m"
        "   ON m.h = e.strategy_hash"
        " JOIN validations v ON v.id = m.mid"
        " WHERE e.dataset_id=? AND e.split='train' AND e.status='completed'"
        " AND e.id = (SELECT MAX(id) FROM experiments x WHERE x.strategy_hash=e.strategy_hash"
        "             AND x.dataset_id=e.dataset_id AND x.split='train' AND x.status='completed')",
        (dataset_id, dataset_id))
    return [{"strategy_hash": r["strategy_hash"],
             "train_net_profit": json.loads(r["metrics_json"])["net_profit"],
             "validation_net_profit": r["validation_net_profit"],
             "validated": bool(r["validated"])} for r in rows]
