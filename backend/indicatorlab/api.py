"""FastAPI application. Every metric returned here is computed from stored data or live workers."""
from __future__ import annotations

import json
import math
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import analytics, data, ranking, significance, studio
from .legacy import import_express_store
from .research import combo_service
from .research import service as research_service
from .research.individual import Rules
from .agents.controller import AgentController
from .agents.research import ResearchSettings
from .agents.runtime import system_metrics
from .config import DATA_DIR, DEFAULT_BACKTEST, DEFAULT_SPLIT, MIN_TRADES, BacktestConfig, CostModel
from .db import get_db
from .evaluator import Evaluator, ResearchConfig
from .optimizer import OBJECTIVES, PRUNERS, SAMPLERS, OptimizationConfig
from .registry import REGISTRY
from .strategy import (DIRECTIONS, MAX_INDICATORS, STOP_LOSS_CHOICES, TARGET_CHOICES, StrategySpec,
                       spec_from_dict)

controller = AgentController()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db = get_db()
    REGISTRY.discover()
    recovered = controller.recover_after_restart()
    app.state.recovered = recovered
    # One-time, idempotent import of the old Express JSON store (datasets and Studio items).
    app.state.legacy_report = import_express_store(DATA_DIR, db, DATA_DIR / "backups")
    app.state.research_interrupted = research_service.recover_after_restart(db)
    yield
    controller.stop_all(wait=5.0)


app = FastAPI(title="Indicator Lab API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])


# -- request models -------------------------------------------------------------------------
class IndicatorIn(BaseModel):
    name: str
    params: dict[str, float] = Field(default_factory=dict)


class StrategyIn(BaseModel):
    indicators: list[IndicatorIn] = Field(min_length=1, max_length=MAX_INDICATORS)
    min_votes: int = 1
    direction: str = "both"
    stop_loss_pct: float = 0.5
    target_pct: float = 1.0


class ImportIn(BaseModel):
    path: str
    name: str
    symbol: str
    timeframe: str = "1min"
    replace: bool = False


class SyntheticIn(BaseModel):
    name: str = "SYNTH_1m_demo"
    days: int = Field(60, ge=5, le=500)
    seed: int = 42


class BacktestIn(BaseModel):
    dataset_id: int
    strategy: StrategyIn
    split: str = "validation"
    initial_capital: float | None = None
    position_pct: float | None = None
    brokerage_per_order: float | None = None
    charges_pct: float | None = None
    slippage_points: float | None = None
    chart_timeframe: str = "5min"


class AgentsStartIn(BaseModel):
    dataset_id: int
    cycles: int = 1
    population_size: int = 16
    generations: int = 6
    hill_climb_steps: int = 20
    tuning_trials: int = 30
    walk_forward_top_k: int = 5
    max_indicators: int = 4
    seed: int = 42


class OptimizationIn(BaseModel):
    dataset_id: int
    study_name: str = Field(min_length=1, max_length=80)
    n_trials: int = 100
    sampler: str = "tpe"
    pruner: str = "median"
    n_jobs: int = 1
    objective: str = "net_profit"
    max_indicators: int = 4
    choices_per_param: int = 5
    seed: int | None = 42


# -- helpers -------------------------------------------------------------------------------
def _research_config(min_trades: int = MIN_TRADES) -> ResearchConfig:
    return ResearchConfig(backtest=DEFAULT_BACKTEST, split=DEFAULT_SPLIT, min_trades=min_trades)


def _spec(body: StrategyIn) -> StrategySpec:
    try:
        return StrategySpec.build([i.model_dump() for i in body.indicators], min_votes=body.min_votes,
                                  direction=body.direction, stop_loss_pct=body.stop_loss_pct,
                                  target_pct=body.target_pct)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc


def _backtest_config(body: BacktestIn) -> ResearchConfig:
    costs = CostModel(
        brokerage_per_order=body.brokerage_per_order if body.brokerage_per_order is not None
        else DEFAULT_BACKTEST.costs.brokerage_per_order,
        charges_pct=body.charges_pct if body.charges_pct is not None else DEFAULT_BACKTEST.costs.charges_pct,
        slippage_points=body.slippage_points if body.slippage_points is not None
        else DEFAULT_BACKTEST.costs.slippage_points)
    bt = BacktestConfig(
        initial_capital=body.initial_capital or DEFAULT_BACKTEST.initial_capital,
        position_pct=body.position_pct or DEFAULT_BACKTEST.position_pct,
        costs=costs, no_entry_after_minute=DEFAULT_BACKTEST.no_entry_after_minute,
        square_off_minute=DEFAULT_BACKTEST.square_off_minute)
    return ResearchConfig(backtest=bt, split=DEFAULT_SPLIT, min_trades=MIN_TRADES)


def _ds_or_404(dataset_id: int) -> dict:
    try:
        return data.get_dataset(dataset_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# -- system and dashboard ----------------------------------------------------------------------
@app.get("/api/health")
def health():
    return {"status": "ok", "recovered_agents": getattr(app.state, "recovered", [])}


@app.get("/api/system")
def system():
    return system_metrics()


@app.get("/api/dashboard")
def dashboard(dataset_id: int | None = None):
    db = get_db()
    exp = db.query("SELECT status, COUNT(*) n FROM experiments GROUP BY status")
    by_status = {r["status"]: r["n"] for r in exp}
    trial_states = {"completed": 0, "pruned": 0, "failed": 0, "running": 0}
    for run in db.query("SELECT study_name FROM optimization_runs"):
        try:
            import optuna

            from .config import OPTUNA_STORAGE
            study = optuna.load_study(study_name=run["study_name"], storage=OPTUNA_STORAGE)
            for t in study.trials:
                key = {"COMPLETE": "completed", "PRUNED": "pruned", "FAIL": "failed",
                       "RUNNING": "running"}.get(t.state.name)
                if key:
                    trial_states[key] += 1
        except Exception:  # noqa: BLE001 - a missing study is skipped, not invented
            continue
    strategies = db.one("SELECT COUNT(DISTINCT strategy_hash) n FROM experiments")
    ds = dataset_id or (db.one("SELECT id FROM datasets ORDER BY id DESC LIMIT 1") or {}).get("id")
    summary = ranking.summary(ds, db=db) if ds else {}
    recent = db.query("SELECT ts, agent, level, message FROM agent_logs ORDER BY id DESC LIMIT 15")
    return {
        "system": system_metrics(),
        "indicators": REGISTRY.summary()["available"],
        "strategies_discovered": int(strategies["n"]) if strategies else 0,
        "trials": trial_states,
        "experiments": {"completed": by_status.get("completed", 0), "failed": by_status.get("failed", 0),
                        "pruned": by_status.get("pruned", 0)},
        "best_net_profit": summary.get("best_net_profit"),
        "best_win_rate": summary.get("best_win_rate"),
        "eligible_strategies": summary.get("eligible_strategies", 0),
        "validated_strategies": summary.get("validated_strategies", 0),
        "agents": controller.agents_status()["agents"],
        "recent_activity": recent,
    }


# -- datasets ---------------------------------------------------------------------------------
@app.get("/api/datasets")
def list_datasets():
    return data.list_datasets()


@app.post("/api/datasets/import")
def import_dataset(body: ImportIn):
    try:
        return data.import_csv(Path(body.path), name=body.name, symbol=body.symbol,
                               timeframe=body.timeframe, replace=body.replace)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/datasets/synthetic")
def create_synthetic(body: SyntheticIn):
    """Generates SYNTHETIC test data (random walk). Not market data; for pipeline testing only."""
    path = data.generate_synthetic(DATA_DIR / "samples" / f"{body.name}.csv", days=body.days, seed=body.seed)
    return data.import_csv(path, name=body.name, symbol="SYNTHETIC", timeframe="1min", replace=True)


MARKET_TIMEFRAMES = {"1min": "1min", "5min": "5min", "10min": "10min", "1h": "1h", "4h": "4h"}
MAX_CHART_BARS = 5000


@app.get("/api/datasets/{dataset_id}/candles")
def candles(dataset_id: int, timeframe: str = "5min", limit: int = Query(500, ge=1, le=MAX_CHART_BARS),
            start: str | None = None, end: str | None = None):
    """OHLC bars for charts. Optional ``start``/``end`` (ISO timestamps, inclusive) select a date range.

    Without a range the most recent ``limit`` bars are returned (previous behaviour). Responses report
    ``total_bars`` in the selected range and ``truncated`` when the range holds more than ``limit`` bars.
    """
    _ds_or_404(dataset_id)
    if timeframe not in MARKET_TIMEFRAMES and not timeframe.endswith("min") and not timeframe.endswith("h"):
        raise HTTPException(400, f"unsupported timeframe '{timeframe}'")
    df = data.load_candles(dataset_id)
    try:
        if start:
            df = df.loc[pd.Timestamp(start):]
        if end:
            df = df.loc[:pd.Timestamp(end)]
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, f"invalid date: {exc}") from exc
    if timeframe != "1min":
        df = df.resample(timeframe, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    total = int(len(df))
    ranged = bool(start or end)
    per_candle = 1
    if ranged and total > limit:
        # The whole range must fit the display limit: group consecutive bars into exact OHLC candles.
        per_candle = int(math.ceil(total / limit))
        first_ts = df.index[np.arange(0, total, per_candle)]
        groups = np.arange(total) // per_candle
        df = df.groupby(groups).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                     "volume": "sum"})
        df.index = first_ts
    elif not ranged:
        df = df.tail(limit)
    return {"timeframe": timeframe, "total_bars": total, "truncated": False,
            "aggregated": per_candle > 1, "source_bars_per_candle": per_candle,
            "start": df.index[0].isoformat() if len(df) else None,
            "end": df.index[-1].isoformat() if len(df) else None,
            "bars": [{"time": t.isoformat(), "open": float(r.open), "high": float(r.high), "low": float(r.low),
                      "close": float(r.close)} for t, r in zip(df.index, df.itertuples())]}



# -- indicators --------------------------------------------------------------------------------
@app.get("/api/indicators")
def indicators():
    return {"summary": REGISTRY.summary(), "indicators": [d.to_dict() for d in REGISTRY.all()]}


@app.get("/api/indicators/{name}")
def indicator(name: str):
    try:
        return REGISTRY.get(name).to_dict()
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/indicators/reload")
def reload_indicators():
    return REGISTRY.discover()


@app.get("/api/strategy/templates")
def templates():
    avail = [d for d in REGISTRY.available()]
    if not avail:
        return []
    names = [{"name": d.name} for d in avail]
    out = [
        {"id": "single", "label": "Single indicator", "description": "One indicator, one vote",
         "spec": {"indicators": names[:1], "min_votes": 1, "direction": "both",
                  "stop_loss_pct": 0.5, "target_pct": 1.0}},
        {"id": "pair", "label": "Two-indicator consensus", "description": "Both indicators must agree",
         "spec": {"indicators": names[:2], "min_votes": min(2, len(names)), "direction": "both",
                  "stop_loss_pct": 0.5, "target_pct": 1.0}},
        {"id": "majority", "label": "Majority vote (all available)",
         "description": "Trade when most indicators agree",
         "spec": {"indicators": names[:MAX_INDICATORS], "min_votes": max(1, min(len(names), MAX_INDICATORS) // 2 + 1),
                  "direction": "both", "stop_loss_pct": 0.3, "target_pct": 0.8}},
    ]
    return out


@app.post("/api/strategy/describe")
def describe(body: StrategyIn):
    spec = _spec(body)
    return {"hash": spec.strategy_hash, "description": spec.describe(), "spec": spec.to_dict()}


@app.get("/api/strategy/options")
def strategy_options():
    return {"directions": list(DIRECTIONS), "stop_loss_choices": list(STOP_LOSS_CHOICES),
            "target_choices": list(TARGET_CHOICES), "max_indicators": MAX_INDICATORS,
            "defaults": {"initial_capital": DEFAULT_BACKTEST.initial_capital,
                         "position_pct": DEFAULT_BACKTEST.position_pct,
                         "brokerage_per_order": DEFAULT_BACKTEST.costs.brokerage_per_order,
                         "charges_pct": DEFAULT_BACKTEST.costs.charges_pct,
                         "slippage_points": DEFAULT_BACKTEST.costs.slippage_points,
                         "min_trades": MIN_TRADES}}


# -- backtest ---------------------------------------------------------------------------------
@app.post("/api/backtest")
def backtest(body: BacktestIn):
    _ds_or_404(body.dataset_id)
    if body.split not in ("train", "validation"):
        raise HTTPException(400, "use /api/backtest/oos for the out-of-sample split")
    spec = _spec(body.strategy)
    ev = Evaluator(body.dataset_id, _backtest_config(body))
    res, metrics = ev.simulate(spec, body.split)
    frame = ev.split_frame(body.split)
    chart = data.load_candles(body.dataset_id).loc[frame.index[0]:frame.index[-1]]
    if body.chart_timeframe != "1min":
        chart = chart.resample(body.chart_timeframe, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    chart = chart.tail(1500)
    return _to_jsonable({
        "strategy": {"hash": spec.strategy_hash, "description": spec.describe(), "spec": spec.to_dict()},
        "split": body.split, "metrics": metrics,
        "trades": res.trades, "equity": analytics.equity_payload(res, frame.index, metrics),
        "candles": [{"time": t.isoformat(), "open": float(r.open), "high": float(r.high),
                     "low": float(r.low), "close": float(r.close)}
                    for t, r in zip(chart.index, chart.itertuples())],
    })


@app.post("/api/backtest/oos")
def backtest_oos(body: BacktestIn):
    """The only route that reads the out-of-sample tail. Runs once per strategy and is then stored."""
    _ds_or_404(body.dataset_id)
    spec = _spec(body.strategy)
    db = get_db()
    existing = db.one("SELECT metrics_json, created_at FROM oos_tests WHERE strategy_hash=? AND dataset_id=?",
                      (spec.strategy_hash, body.dataset_id))
    if existing:
        return {"already_run": True, "created_at": existing["created_at"],
                "metrics": json.loads(existing["metrics_json"])}
    ev = Evaluator(body.dataset_id, _backtest_config(body))
    _, metrics = ev.simulate(spec, "oos")
    from .db import utcnow

    db.execute("INSERT INTO oos_tests(strategy_hash, dataset_id, created_at, metrics_json) VALUES (?,?,?,?)",
               (spec.strategy_hash, body.dataset_id, utcnow(), db.dumps(metrics)))
    return {"already_run": False, "metrics": _to_jsonable(metrics)}


# -- agents -----------------------------------------------------------------------------------
@app.get("/api/agents")
def agents():
    return controller.agents_status()


@app.post("/api/agents/start")
def start_agents(body: AgentsStartIn):
    _ds_or_404(body.dataset_id)
    try:
        settings = ResearchSettings(**body.model_dump())
        return controller.start_agents(settings, _research_config())
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/agents/{name}/{action}")
def agent_control(name: str, action: str):
    try:
        return controller.control(name, action)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/agents/{name}/logs")
def agent_logs(name: str, after: int = 0, limit: int = Query(200, le=1000)):
    rows = get_db().query("SELECT id, ts, level, message FROM agent_logs WHERE agent=? AND id>?"
                          " ORDER BY id DESC LIMIT ?", (name, after, limit))
    return list(reversed(rows))


@app.get("/api/agents/evaluation/snapshot")
def evaluation_snapshot():
    return controller.evaluation_snapshot() or {}


# -- optimization -----------------------------------------------------------------------------
@app.get("/api/optimization/options")
def optimization_options():
    return {"samplers": list(SAMPLERS), "pruners": list(PRUNERS), "objectives": list(OBJECTIVES),
            "max_indicators": MAX_INDICATORS,
            "notes": {"grid": "exhaustive over discretised choices; size is capped",
                      "cmaes": "continuous-only sampler; categorical choices fall back to independent sampling",
                      "nsga2": "multi-objective: net profit and drawdown; pruning is disabled"}}


@app.get("/api/optimization/runs")
def optimization_runs():
    return controller.optimization_runs()


@app.post("/api/optimization/start")
def start_optimization(body: OptimizationIn):
    _ds_or_404(body.dataset_id)
    try:
        cfg = OptimizationConfig(**body.model_dump())
        return controller.start_optimization(cfg, _research_config())
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.post("/api/optimization/{study_name}/{action}")
def optimization_control(study_name: str, action: str):
    try:
        return controller.control_optimization(f"optuna:{study_name}", action)
    except (ValueError, RuntimeError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/optimization/studies/{study_name}")
def study(study_name: str):
    try:
        return controller.optimization_study(study_name)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


# -- results ---------------------------------------------------------------------------------
@app.get("/api/results/rankings")
def rankings(dataset_id: int, top: int = Query(10, ge=1, le=50), split: str = "train"):
    _ds_or_404(dataset_id)
    return {key: ranking.leaderboard(key, dataset_id, top=top, split=split) for key in ranking.RANKINGS}


@app.get("/api/results/rankings/{key}")
def ranking_one(key: str, dataset_id: int, top: int = Query(10, ge=1, le=50), split: str = "train"):
    _ds_or_404(dataset_id)
    try:
        return ranking.leaderboard(key, dataset_id, top=top, split=split)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/results/strategies/{strategy_hash}")
def strategy_detail(strategy_hash: str, dataset_id: int):
    db = get_db()
    rows = db.query("SELECT split, spec_json, metrics_json, created_at, agent, study_name FROM experiments"
                    " WHERE strategy_hash=? AND dataset_id=? AND status='completed' ORDER BY id",
                    (strategy_hash, dataset_id))
    if not rows:
        raise HTTPException(404, "strategy not found")
    validations = db.query("SELECT * FROM validations WHERE strategy_hash=? AND dataset_id=? ORDER BY id DESC",
                           (strategy_hash, dataset_id))
    oos = db.one("SELECT metrics_json, created_at FROM oos_tests WHERE strategy_hash=? AND dataset_id=?",
                 (strategy_hash, dataset_id))
    return _to_jsonable({
        "hash": strategy_hash,
        "spec": json.loads(rows[0]["spec_json"]),
        "experiments": [{"split": r["split"], "metrics": json.loads(r["metrics_json"]),
                         "created_at": r["created_at"], "agent": r["agent"], "study": r["study_name"]}
                        for r in rows],
        "validations": [{**v, "segments": json.loads(v.pop("segments_json")),
                         "validated": bool(v["validated"])} for v in validations],
        "out_of_sample": {"metrics": json.loads(oos["metrics_json"]), "created_at": oos["created_at"]} if oos else None,
    })


# -- experiments ----------------------------------------------------------------------------
@app.get("/api/experiments")
def experiments(dataset_id: int | None = None, study: str | None = None, status: str | None = None,
                limit: int = Query(100, le=1000), offset: int = 0):
    where, params = [], []
    if dataset_id is not None:
        where.append("dataset_id=?")
        params.append(dataset_id)
    if study:
        where.append("study_name=?")
        params.append(study)
    if status:
        where.append("status=?")
        params.append(status)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    rows = get_db().query(
        f"SELECT id, created_at, agent, study_name, trial_number, strategy_hash, split, spec_json,"
        f" metrics_json, eligible, status, duration_s, error FROM experiments {clause}"
        f" ORDER BY id DESC LIMIT ? OFFSET ?", (*params, limit, offset))
    total = get_db().one(f"SELECT COUNT(*) n FROM experiments {clause}", params)
    out = []
    for r in rows:
        spec = json.loads(r.pop("spec_json"))
        r["description"] = spec_from_dict(spec).describe()
        r["spec"] = spec
        r["metrics"] = json.loads(r.pop("metrics_json")) if r["metrics_json"] else None
        r["eligible"] = bool(r["eligible"])
        out.append(r)
    return {"total": total["n"] if total else 0, "rows": out}


@app.get("/api/experiments/{experiment_id}")
def experiment(experiment_id: int):
    row = get_db().one("SELECT * FROM experiments WHERE id=?", (experiment_id,))
    if not row:
        raise HTTPException(404, "experiment not found")
    row["spec"] = json.loads(row.pop("spec_json"))
    row["metrics"] = json.loads(row.pop("metrics_json")) if row["metrics_json"] else None
    return row


# -- analytics ------------------------------------------------------------------------------
@app.get("/api/analytics/optuna/{study_name}/{chart}")
def optuna_chart(study_name: str, chart: str):
    try:
        return analytics.optuna_chart(study_name, chart)
    except (ValueError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - Optuna raises ValueError for empty studies
        raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc


@app.get("/api/analytics/equity")
def equity(dataset_id: int, strategy_hash: str, split: str = "validation"):
    rows = get_db().query("SELECT spec_json FROM experiments WHERE strategy_hash=? AND dataset_id=? LIMIT 1",
                          (strategy_hash, dataset_id))
    if not rows:
        raise HTTPException(404, "strategy not found")
    if split not in ("train", "validation"):
        raise HTTPException(400, "equity analytics supports train and validation only")
    spec = spec_from_dict(json.loads(rows[0]["spec_json"]))
    return _to_jsonable(analytics.equity_series(dataset_id, spec, split, _research_config()))


@app.get("/api/analytics/walk_forward")
def walk_forward(dataset_id: int):
    return analytics.walk_forward_board(dataset_id)


@app.get("/api/analytics/train_vs_validation")
def train_vs_validation(dataset_id: int):
    return analytics.train_vs_validation(dataset_id)


# -- statistics: deflated Sharpe ratio and PBO (train split only) --------------------------------
class DsrIn(BaseModel):
    dataset_id: int
    split: str = "train"


class PboIn(BaseModel):
    dataset_id: int
    blocks: int = Field(8, ge=4, le=16)
    max_configs: int = Field(40, ge=2, le=200)


@app.post("/api/statistics/dsr")
def run_dsr(body: DsrIn):
    _ds_or_404(body.dataset_id)
    if body.split != "train":
        raise HTTPException(400, "DSR is computed on the train split only; OOS is never used for selection")
    return significance.compute_dsr(body.dataset_id, body.split, _research_config())


@app.post("/api/statistics/pbo")
def run_pbo(body: PboIn):
    _ds_or_404(body.dataset_id)
    try:
        return significance.compute_pbo(body.dataset_id, blocks=body.blocks, max_configs=body.max_configs,
                                        research=_research_config())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/statistics")
def list_statistics(dataset_id: int, kind: str | None = Query(None, pattern="^(dsr|pbo)$")):
    _ds_or_404(dataset_id)
    return _to_jsonable(significance.list_tests(dataset_id, kind))


@app.get("/api/statistics/experiments/{experiment_id}/dsr")
def experiment_dsr(experiment_id: int):
    found = significance.dsr_for_experiment(experiment_id)
    if found is None:
        return {"status": "not_performed", "reason": "no DSR has been computed for this experiment"}
    return found


# -- Indicator Studio and Strategy Studio (ported from the Express routes) ---------------------
def _studio_error(exc: Exception):
    if isinstance(exc, LookupError):
        raise HTTPException(404, str(exc)) from exc
    raise HTTPException(400, str(exc)) from exc


def _validation_failed(exc: studio.ValidationFailed, key: str):
    """Same body as the Express routes: top-level detail, validation and the stored item."""
    return JSONResponse(status_code=400, content=_to_jsonable({
        "detail": str(exc), "validation": exc.validation, key: exc.item}))


@app.get("/api/studio/indicators")
def studio_indicators():
    items = studio.list_items("indicator")
    return {"total": len(items), "registered_count": len(studio.registered_items("indicator")), "items": items}


@app.post("/api/studio/indicators")
def create_studio_indicator(body: dict):
    try:
        return studio.create_indicator(body)
    except ValueError as exc:
        _studio_error(exc)


@app.put("/api/studio/indicators/{item_id}")
def update_studio_indicator_meta(item_id: int, body: dict):
    try:
        return studio.update_indicator_meta(item_id, body)
    except (ValueError, LookupError) as exc:
        _studio_error(exc)


@app.delete("/api/studio/indicators/{item_id}")
def delete_studio_indicator(item_id: int):
    try:
        return studio.delete_item("indicator", item_id)
    except LookupError as exc:
        _studio_error(exc)


@app.post("/api/studio/indicators/validate")
def validate_studio_indicator(body: dict):
    try:
        return _to_jsonable(studio.validate_indicator(body))
    except ValueError as exc:
        _studio_error(exc)


@app.post("/api/studio/indicators/{item_id}/update")
def update_studio_indicator(item_id: int, body: dict):
    try:
        item, validation = studio.update_indicator(item_id, body)
    except studio.ValidationFailed as exc:
        return _validation_failed(exc, "indicator")
    except (ValueError, LookupError) as exc:
        _studio_error(exc)
    return _to_jsonable({
        "indicator": item, "validation": validation,
        "catalogue_size": len(studio.registered_items("indicator")), "auto_started_optimization": False,
        "message": f"Indicator '{item['name']}' v{item['version']} validated, saved, and registered in the "
                   "Research Agent catalogue. No optimization run was started automatically.",
    })


@app.get("/api/studio/strategies")
def studio_strategies():
    items = studio.list_items("strategy")
    return {"total": len(items), "registered_count": len(studio.registered_items("strategy")), "items": items}


@app.post("/api/studio/strategies")
def create_studio_strategy(body: dict):
    try:
        return studio.create_strategy(body)
    except ValueError as exc:
        _studio_error(exc)


@app.put("/api/studio/strategies/{item_id}")
def update_studio_strategy_meta(item_id: int, body: dict):
    try:
        return studio.update_strategy_meta(item_id, body)
    except (ValueError, LookupError) as exc:
        _studio_error(exc)


@app.delete("/api/studio/strategies/{item_id}")
def delete_studio_strategy(item_id: int):
    try:
        return studio.delete_item("strategy", item_id)
    except LookupError as exc:
        _studio_error(exc)


@app.post("/api/studio/strategies/validate")
def validate_studio_strategy(body: dict):
    try:
        return _to_jsonable(studio.execute_strategy_validation(body))
    except ValueError as exc:
        _studio_error(exc)


@app.post("/api/studio/strategies/{item_id}/update")
def update_studio_strategy(item_id: int, body: dict):
    try:
        item, validation = studio.update_strategy(item_id, body)
    except studio.ValidationFailed as exc:
        return _validation_failed(exc, "strategy")
    except (ValueError, LookupError) as exc:
        _studio_error(exc)
    return _to_jsonable({
        "strategy": item, "validation": validation,
        "catalogue_size": len(studio.registered_items("strategy")), "auto_started_optimization": False,
        "message": f"Strategy '{item['name']}' v{item['version']} validated, saved, and registered in the "
                   "Research Agent catalogue. No optimization run was started automatically.",
    })


class UploadIn(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    symbol: str = Field(min_length=1, max_length=40)
    timeframe: str = "1min"
    csv_text: str = Field(min_length=1, max_length=25_000_000)
    replace: bool = True


@app.post("/api/datasets/upload")
def upload_dataset(body: UploadIn):
    """Import candles pasted or uploaded as CSV text. Same validation as /api/datasets/import."""
    import re as _re
    safe = _re.sub(r"[^A-Za-z0-9_.-]+", "_", body.name)[:100] or "upload"
    target = DATA_DIR / "uploads" / f"{safe}.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body.csv_text, encoding="utf-8")
    try:
        return data.import_csv(target, name=body.name, symbol=body.symbol,
                               timeframe=body.timeframe, replace=body.replace)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc



# -- individual indicator research ----------------------------------------------------------------
class RulesIn(BaseModel):
    direction: str = "both"
    stop_loss_pct: float = 0.5
    target_pct: float = 1.0
    max_hold_bars: int | None = None
    brokerage_per_order: float = 20.0
    charges_pct: float = 0.0005
    slippage_points: float = 0.5


class IndividualIn(BaseModel):
    dataset_id: int
    split: str = "train"
    timeframe: str = "1min"
    start: str | None = None
    end: str | None = None
    rules: RulesIn = Field(default_factory=RulesIn)


def _rules(r: RulesIn) -> Rules:
    return Rules(**r.model_dump())


@app.post("/api/research/individual/start")
def start_individual(body: IndividualIn):
    _ds_or_404(body.dataset_id)
    try:
        return _to_jsonable(research_service.start_run(research_service.RunRequest(
            dataset_id=body.dataset_id, split=body.split, timeframe=body.timeframe, start=body.start,
            end=body.end, rules=_rules(body.rules))))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/research/individual")
def individual_results(dataset_id: int, split: str = Query("train", pattern="^(train|validation)$"),
                       run_id: int | None = None):
    _ds_or_404(dataset_id)
    return _to_jsonable(research_service.individual_table(dataset_id, split, run_id))


@app.get("/api/research/runs/{run_id}")
def research_run(run_id: int):
    try:
        return _to_jsonable(research_service.run_status(run_id))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@app.post("/api/research/runs/{run_id}/{action}")
def research_run_control(run_id: int, action: str):
    try:
        return _to_jsonable(research_service.control_run(run_id, action))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

class ComboIn(BaseModel):
    dataset_id: int
    timeframe: str = "1min"
    start: str | None = None
    end: str | None = None
    rules: RulesIn = Field(default_factory=RulesIn)
    segments: int = 4
    pool_size: int = 12
    corr_threshold: float = 0.80
    purge_bars: int = 60
    max_evaluations: int = 4000
    top_n: int = 10


@app.post("/api/research/combination/start")
def start_combination(body: ComboIn):
    _ds_or_404(body.dataset_id)
    try:
        req = combo_service.ComboRequest(
            dataset_id=body.dataset_id, timeframe=body.timeframe, start=body.start, end=body.end,
            rules=_rules(body.rules), segments=body.segments, pool_size=body.pool_size,
            corr_threshold=body.corr_threshold, purge_bars=body.purge_bars,
            max_evaluations=body.max_evaluations, top_n=body.top_n)
        return _to_jsonable(combo_service.start_combination(req))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/research/combination")
def combination_latest(dataset_id: int, run_id: int | None = None):
    _ds_or_404(dataset_id)
    rid = run_id or combo_service.latest_combination_run(dataset_id)
    if rid is None:
        return {"run": None}
    try:
        return _to_jsonable(combo_service.combination_payload(rid))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


class SignalSpecIn(BaseModel):
    spec: dict | None = None


@app.get("/api/research/signals")
def signal_rules_list():
    from .research.individual import discover_sources
    from .research.signals import get_spec
    db = get_db()
    available, unavailable = discover_sources(db)
    rows = []
    for src in available + unavailable:
        rows.append({"source_key": src.key, "name": src.name, "source": src.source, "category": src.category,
                     "available": src.available, "status_reason": src.status_reason,
                     "spec": get_spec(db, src.key)})
    return {"indicators": rows}


@app.put("/api/research/signals/{source_key}")
def signal_rules_put(source_key: str, body: SignalSpecIn):
    from .research.individual import discover_sources
    from .research.signals import SpecError, set_spec
    db = get_db()
    available, unavailable = discover_sources(db)
    known = {s.key for s in available + unavailable}
    try:
        spec = set_spec(db, source_key, body.spec, known)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except SpecError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"source_key": source_key, "spec": spec}


class RobustIn(BaseModel):
    combination_run_id: int
    candidates: int = 10
    permutations: int = 200
    mc_iterations: int = 5000
    spa_draws: int = 500


@app.post("/api/research/robustness/start")
def robustness_start(body: RobustIn):
    from .research import robust_service
    try:
        return _to_jsonable(robust_service.start_robustness(
            body.combination_run_id, candidates=body.candidates, permutations=body.permutations,
            mc_iterations=body.mc_iterations, spa_draws=body.spa_draws))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/research/robustness")
def robustness_get(combination_run_id: int):
    from .research import robust_service
    return _to_jsonable(robust_service.latest_robustness(combination_run_id) or {"run_id": None, "result": None})


@app.get("/api/research/synthetic/scenarios")
def synthetic_scenarios():
    from .research import synthetic
    return {"scenarios": synthetic.scenario_list(), "label": "Synthetic data. Never market performance."}


class SynthIn(BaseModel):
    scenario: str
    days: int = Field(60, ge=5, le=365)
    seed: int = Field(1, ge=0, le=1_000_000)


@app.post("/api/research/synthetic/create")
def synthetic_create(body: SynthIn):
    from .research import synthetic
    try:
        return _to_jsonable(synthetic.create_synthetic_dataset(body.scenario, body.days, body.seed, get_db()))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/research/analysis")
def research_analysis(dataset_id: int):
    """Optional analysis on stored results: Pareto front over evaluated indicators, and clusters of the
    correlation matrix of the latest combination run. Reads stored metrics only; runs no backtests."""
    from .research import analysis
    _ds_or_404(dataset_id)
    db = get_db()
    from .research import service as individual_service
    ind = individual_service.individual_table(dataset_id, "train", None, db)
    points = [{"name": r["name"], "source_key": r["source_key"], "net_return_pct": r["net_return_pct"],
               "raw_win_rate": r["raw_win_rate"], "max_drawdown_pct": r["max_drawdown_pct"],
               "wilson_lb": r["wilson_lb"]} for r in ind["rows"] if r["status"] == "evaluated"]
    pareto = analysis.pareto_front(points) if points else {"front": [], "ranks": []}
    clusters = None
    run_id = combo_service.latest_combination_run(dataset_id, db)
    if run_id is not None:
        corr = combo_service.combination_payload(run_id, db).get("correlation")
        if corr and corr.get("names"):
            clusters = analysis.cluster_indicators(corr["names"], corr["matrix"], corr.get("threshold", 0.8))
    return _to_jsonable({"points": points, "pareto": pareto, "clusters": clusters,
                         "clusters_from_run": run_id})


@app.get("/api/research/dashboard")
def research_dashboard(dataset_id: int):
    _ds_or_404(dataset_id)
    return _to_jsonable(combo_service.dashboard_payload(dataset_id))


@app.get("/api/research/oos/{run_id}")
def oos_get(run_id: int):
    return _to_jsonable(combo_service.oos_for_run(run_id))


@app.post("/api/research/oos/{run_id}")
def oos_run(run_id: int):
    """One-shot out-of-sample test of a run's frozen combination. A second call is refused."""
    try:
        return _to_jsonable(combo_service.run_oos(run_id))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# -- built frontend (production). API routes above always take precedence. --------------------
from fastapi.responses import FileResponse  # noqa: E402

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


@app.get("/{full_path:path}", include_in_schema=False)
def spa(full_path: str):
    if full_path.startswith("api/"):
        raise HTTPException(404, "unknown API route")
    if not FRONTEND_DIST.exists():
        raise HTTPException(404, "frontend not built: run npm run build in frontend/")
    candidate = (FRONTEND_DIST / full_path).resolve()
    if full_path and candidate.is_file() and FRONTEND_DIST.resolve() in candidate.parents:
        return FileResponse(candidate)
    return FileResponse(FRONTEND_DIST / "index.html")
