"""Optuna-backed strategy search.

Public Optuna APIs used: ``create_study`` with RDB storage, samplers (TPE, Random, Grid, NSGA-II,
CMA-ES), the MedianPruner with intermediate reports, ``study.stop()`` and ``optimize`` callbacks.

Search space
------------
Every available indicator gets a boolean ``use__<name>``. Each parameter is discretised to a
short list of choices so that every sampler (including GridSampler) sees the same space.
Continuous parameters therefore get ``choices_per_param`` evenly spaced values. CMA-ES works
on continuous spaces only, so with discrete choices it falls back to independent sampling;
it is offered for completeness and labelled as such in the UI.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import optuna
from optuna.pruners import MedianPruner, NopPruner
from optuna.samplers import CmaEsSampler, GridSampler, NSGAIISampler, RandomSampler, TPESampler

from .config import OPTUNA_STORAGE
from .db import get_db, utcnow
from .evaluator import Evaluator, ResearchConfig
from .agents.runtime import StopRequested, WorkerContext
from .registry import REGISTRY, IndicatorDef, ParamSpec
from .strategy import DIRECTIONS, MAX_INDICATORS, STOP_LOSS_CHOICES, TARGET_CHOICES, StrategySpec

optuna.logging.set_verbosity(optuna.logging.WARNING)

SAMPLERS = ("tpe", "random", "grid", "nsga2", "cmaes")
PRUNERS = ("median", "none")
OBJECTIVES = ("net_profit", "profit_vs_drawdown")
PENALTY = -1e12


@dataclass(frozen=True)
class OptimizationConfig:
    dataset_id: int
    study_name: str
    n_trials: int = 100
    sampler: str = "tpe"
    pruner: str = "median"
    n_jobs: int = 1
    objective: str = "net_profit"
    max_indicators: int = 4
    choices_per_param: int = 5
    seed: int | None = 42

    def validate(self) -> None:
        if self.sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}")
        if self.pruner not in PRUNERS:
            raise ValueError(f"pruner must be one of {PRUNERS}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        if not 1 <= self.max_indicators <= MAX_INDICATORS:
            raise ValueError(f"max_indicators must be between 1 and {MAX_INDICATORS}")
        if self.n_trials < 1 or self.n_jobs < 1:
            raise ValueError("n_trials and n_jobs must be at least 1")
        if not self.study_name or len(self.study_name) > 80:
            raise ValueError("study_name must be 1 to 80 characters")

    def to_dict(self) -> dict:
        return asdict(self)


def discrete_choices(p: ParamSpec, k: int) -> list:
    if p.kind == "int":
        values = np.linspace(p.low, p.high, min(k, p.high - p.low + 1))
        return sorted({int(round(v)) for v in values})
    return sorted({round(float(v), 4) for v in np.linspace(p.low, p.high, k)})


def search_choices(defs: list[IndicatorDef], cfg: OptimizationConfig) -> dict[str, list]:
    space: dict[str, list] = {}
    for d in defs:
        space[f"use__{d.name}"] = [True, False]
        for p in d.params:
            space[f"{d.name}__{p.name}"] = discrete_choices(p, cfg.choices_per_param)
    space["min_votes"] = list(range(1, cfg.max_indicators + 1))
    space["stop_loss_pct"] = list(STOP_LOSS_CHOICES)
    space["target_pct"] = list(TARGET_CHOICES)
    space["direction"] = list(DIRECTIONS)
    return space


def spec_from_trial(trial: optuna.Trial, defs: list[IndicatorDef], space: dict[str, list],
                    cfg: OptimizationConfig) -> StrategySpec | None:
    chosen = []
    for d in defs:
        if trial.suggest_categorical(f"use__{d.name}", space[f"use__{d.name}"]):
            params = {p.name: trial.suggest_categorical(f"{d.name}__{p.name}", space[f"{d.name}__{p.name}"])
                      for p in d.params}
            chosen.append({"name": d.name, "params": params})
    if not chosen or len(chosen) > cfg.max_indicators:
        return None
    min_votes = min(trial.suggest_categorical("min_votes", space["min_votes"]), len(chosen))
    return StrategySpec.build(
        chosen, min_votes=min_votes,
        direction=trial.suggest_categorical("direction", space["direction"]),
        stop_loss_pct=trial.suggest_categorical("stop_loss_pct", space["stop_loss_pct"]),
        target_pct=trial.suggest_categorical("target_pct", space["target_pct"]))


def score_of(metrics: dict) -> float:
    return float(metrics["net_profit"]) if metrics["eligible"] else PENALTY


def dd_score_of(metrics: dict) -> float:
    return -float(metrics["max_drawdown_pct"]) if metrics["eligible"] else PENALTY


def make_sampler(name: str, seed: int | None, space: dict[str, list] | None = None):
    if name == "tpe":
        return TPESampler(seed=seed)
    if name == "random":
        return RandomSampler(seed=seed)
    if name == "grid":
        size = math.prod(len(v) for v in space.values())
        if size > 200_000:
            raise ValueError(f"grid search would need {size} points; reduce choices_per_param or indicators")
        return GridSampler(space, seed=seed)
    if name == "nsga2":
        return NSGAIISampler(seed=seed)
    if name == "cmaes":
        return CmaEsSampler(seed=seed)
    raise ValueError(name)


def run_study(cfg: OptimizationConfig, ctx: WorkerContext, research: ResearchConfig | None = None,
              *, agent: str = "optimization") -> optuna.Study:
    """Blocking. Intended to run inside a ManagedWorker thread."""
    cfg.validate()
    defs = REGISTRY.available()
    if not defs:
        raise RuntimeError("no available indicators are registered")
    space = search_choices(defs, cfg)
    multi = cfg.objective == "profit_vs_drawdown"
    use_pruner = cfg.pruner == "median" and not multi and cfg.sampler != "grid"
    evaluator = Evaluator(cfg.dataset_id, research)
    db = get_db()

    sampler = make_sampler(cfg.sampler, cfg.seed, space)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0) if use_pruner else NopPruner()
    study = optuna.create_study(
        study_name=cfg.study_name, storage=OPTUNA_STORAGE, sampler=sampler, pruner=pruner,
        directions=["maximize", "maximize"] if multi else None,
        direction=None if multi else "maximize", load_if_exists=False)
    db.execute("INSERT INTO optimization_runs(study_name, created_at, status, config_json)"
               " VALUES (?,?,?,?)", (cfg.study_name, utcnow(), "running", db.dumps(cfg.to_dict())))
    ctx.state.detail.update({"study_name": cfg.study_name, "sampler": cfg.sampler,
                             "pruner": cfg.pruner if use_pruner else "none", "objective": cfg.objective})
    ctx.set_task(f"Optuna study {cfg.study_name}", algorithm=f"Optuna {cfg.sampler.upper()}", progress=0.0)

    def objective(trial: optuna.Trial):
        if not _gate(ctx, trial):
            raise optuna.TrialPruned("stop requested")
        spec = spec_from_trial(trial, defs, space, cfg)
        if spec is None:
            trial.set_user_attr("rejected", "indicator count outside limits")
            return (PENALTY, PENALTY) if multi else PENALTY
        trial.set_user_attr("strategy_hash", spec.strategy_hash)
        trial.set_user_attr("description", spec.describe())
        cached = evaluator.cached(spec, "train")
        if cached is not None:
            ctx.duplicates_prevented += 1
            metrics = cached
        else:
            report = use_pruner

            def on_chunk(k: int, cumulative: float) -> None:
                if report:
                    trial.report(cumulative, k)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            started = time.monotonic()
            try:
                metrics = evaluator.evaluate(spec, "train", on_chunk=on_chunk)
            except optuna.TrialPruned:
                evaluator.record_failure(spec, "train", agent=agent, error="pruned by MedianPruner",
                                         study_name=cfg.study_name, trial_number=trial.number,
                                         status="pruned")
                raise
            except Exception as exc:  # noqa: BLE001 - recorded, then the trial is marked FAIL
                evaluator.record_failure(spec, "train", agent=agent, error=f"{type(exc).__name__}: {exc}",
                                         study_name=cfg.study_name, trial_number=trial.number)
                ctx.count_error()
                raise
            evaluator.record(spec, "train", metrics, agent=agent, study_name=cfg.study_name,
                             trial_number=trial.number, duration_s=round(time.monotonic() - started, 3))
        ctx.count_trial()
        for key in ("trades", "net_profit", "win_rate", "max_drawdown_pct", "profit_factor"):
            trial.set_user_attr(key, metrics[key])
        trial.set_user_attr("eligible", metrics["eligible"])
        if multi:
            return score_of(metrics), dd_score_of(metrics)
        return score_of(metrics)

    def progress(study_: optuna.Study, trial_: optuna.trial.FrozenTrial) -> None:
        done = sum(1 for t in study_.trials if t.state.is_finished())
        ctx.set_task(f"Optuna study {cfg.study_name}: {done}/{cfg.n_trials} trials finished",
                     progress=done / cfg.n_trials)
        if ctx.stop_event.is_set():
            study_.stop()

    status, error = "completed", None
    try:
        study.optimize(objective, n_trials=cfg.n_trials, n_jobs=cfg.n_jobs,
                       catch=(Exception,), callbacks=[progress])
        if ctx.stop_event.is_set():
            status = "stopped"
    except Exception as exc:  # noqa: BLE001 - recorded, then re-raised to the worker
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        db.execute("UPDATE optimization_runs SET status=?, error=? WHERE study_name=?",
                   (status, error, cfg.study_name))
    return study


def _gate(ctx: WorkerContext, trial: optuna.Trial) -> bool:
    """Honour pause (blocking) and stop. Returns False when the study should end."""
    try:
        ctx.checkpoint()
    except StopRequested:
        trial.study.stop()
        return False
    return not ctx.stop_event.is_set()


def study_summary(study: optuna.Study) -> dict:
    counts = {"complete": 0, "pruned": 0, "failed": 0, "running": 0, "waiting": 0}
    for t in study.trials:
        counts[t.state.name.lower()] = counts.get(t.state.name.lower(), 0) + 1
    best = None
    if len(study.directions) == 1 and any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        b = study.best_trial
        best = {"trial": b.number, "value": b.value, "params": b.user_attrs.get("description")}
    return {"study_name": study.study_name, "directions": [d.name.lower() for d in study.directions],
            "trials": len(study.trials), "states": counts, "best": best}
