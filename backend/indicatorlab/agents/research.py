"""Agent 1 - Strategy Research Agent.

Each cycle runs these phases in order, and they feed each other:

  1. Forward selection     add the indicator that most improves train profit, one at a time
  2. Backward elimination  remove indicators whose removal does not reduce train profit
  3. Genetic algorithm     population search with tournament selection, crossover and mutation
  4. Hill climbing         step to the best neighbour (parameters, votes, risk) until no gain
  5. Optuna tuning         TPE search over the parameters of the best structure (with pruning)
  6. Walk-forward          validate the top candidates on the validation split

Only the train and validation splits are used here. The out-of-sample tail is never read.
Progress is checkpointed after every phase and every GA generation, so a restart resumes.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from ..config import OPTUNA_STORAGE
from ..db import get_db, utcnow
from ..evaluator import Evaluator, ResearchConfig
from ..optimizer import PENALTY, _gate, score_of
from ..ranking import leaderboard
from ..registry import REGISTRY
from ..strategy import (DIRECTIONS, STOP_LOSS_CHOICES, TARGET_CHOICES, StrategySpec, crossover,
                        mutate, random_spec)
from .runtime import WorkerContext

AGENT_NAME = "research_agent"
PHASES = ("forward", "backward", "ga", "hill", "tune", "walk_forward")


@dataclass(frozen=True)
class ResearchSettings:
    dataset_id: int
    cycles: int = 1  # 0 means run until stopped
    population_size: int = 16
    generations: int = 6
    hill_climb_steps: int = 20
    tuning_trials: int = 30
    walk_forward_top_k: int = 5
    max_indicators: int = 4
    seed: int = 42
    auto_resume: bool = False

    def validate(self) -> None:
        if self.cycles < 0 or self.population_size < 4 or self.generations < 1:
            raise ValueError("cycles >= 0, population_size >= 4 and generations >= 1 are required")
        if not 1 <= self.max_indicators <= 6:
            raise ValueError("max_indicators must be between 1 and 6")
        if self.tuning_trials < 0 or self.walk_forward_top_k < 1:
            raise ValueError("tuning_trials >= 0 and walk_forward_top_k >= 1 are required")

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _spec_dict(spec: StrategySpec) -> dict:
    return spec.to_dict()


def _from_dict(data: dict) -> StrategySpec:
    return StrategySpec.build(data["indicators"], min_votes=data["min_votes"],
                              direction=data["direction"], stop_loss_pct=data["stop_loss_pct"],
                              target_pct=data["target_pct"])


class ResearchAgent:
    def __init__(self, settings: ResearchSettings, ctx: WorkerContext,
                 research: ResearchConfig | None = None):
        settings.validate()
        self.s = settings
        self.ctx = ctx
        self.ev = Evaluator(settings.dataset_id, research)
        self.db = get_db()
        self.defs = REGISTRY.available()
        if not self.defs:
            raise RuntimeError("no available indicators are registered")
        self.pool = [d.name for d in self.defs]

    # -- evaluation gate ----------------------------------------------------------------------
    def score(self, spec: StrategySpec) -> float:
        self.ctx.checkpoint()
        metrics = self.ev.evaluate_cached(spec, "train", agent=AGENT_NAME)
        if metrics.pop("_cached", False):
            self.ctx.duplicates_prevented += 1
        self.ctx.count_trial()
        return score_of(metrics)

    def _spec(self, indicators: list[dict], *, min_votes: int | None = None, direction: str = "both",
              sl: float = 0.5, tp: float = 1.0) -> StrategySpec:
        votes = min_votes if min_votes is not None else max(1, (len(indicators) + 1) // 2)
        return StrategySpec.build(indicators, min_votes=min(votes, len(indicators)),
                                  direction=direction, stop_loss_pct=sl, target_pct=tp)

    # -- phases -------------------------------------------------------------------------------
    def forward_selection(self) -> StrategySpec:
        self.ctx.set_task("forward selection", algorithm="Forward Selection", progress=0.0)
        selected: list[dict] = []
        remaining = list(self.pool)
        best_spec, best_score = None, float("-inf")
        while remaining and len(selected) < self.s.max_indicators:
            options = []
            for name in remaining:
                spec = self._spec(selected + [{"name": name, "params": {}}])
                options.append((self.score(spec), name, spec))
            top_score, top_name, top_spec = max(options, key=lambda x: x[0])
            if best_spec is not None and top_score <= best_score:
                break
            selected.append({"name": top_name, "params": {}})
            remaining.remove(top_name)
            best_spec, best_score = top_spec, top_score
            self.ctx.log(f"forward selection added {top_name}; train score {top_score:,.2f}")
        self.ctx.log(f"forward selection finished with {len(selected)} indicator(s): {best_spec.describe()}")
        return best_spec

    def backward_elimination(self, spec: StrategySpec) -> StrategySpec:
        self.ctx.set_task("backward elimination", algorithm="Backward Elimination", progress=0.0)
        current, current_score = spec, self.score(spec)
        while len(current.indicators) > 1:
            options = []
            for drop in current.indicators:
                rest = [{"name": c.name, "params": c.params_dict} for c in current.indicators if c.name != drop.name]
                cand = self._spec(rest, min_votes=current.min_votes, direction=current.direction,
                                  sl=current.stop_loss_pct, tp=current.target_pct)
                options.append((self.score(cand), drop.name, cand))
            top_score, dropped, cand = max(options, key=lambda x: x[0])
            if top_score < current_score:
                break  # every removal hurts: all remaining indicators contribute
            self.ctx.log(f"backward elimination removed {dropped}; train score {top_score:,.2f}")
            current, current_score = cand, top_score
        self.ctx.log(f"backward elimination finished with {len(current.indicators)} indicator(s)")
        return current

    def genetic_algorithm(self, seeds: list[StrategySpec], save_state) -> StrategySpec:
        P = self.s.population_size
        rng = self._rng
        self.ctx.set_task("genetic algorithm", algorithm="Genetic Algorithm", progress=0.0)
        resume = save_state(None)
        start_gen = 0
        population: list[StrategySpec] = []
        best_spec, best_score = None, float("-inf")
        if resume and resume.get("population"):
            population = [_from_dict(d) for d in resume["population"]]
            start_gen = int(resume["generation"])
            best_spec = _from_dict(resume["best"]) if resume.get("best") else None
            best_score = float(resume.get("best_score", float("-inf")))
            self.ctx.log(f"genetic algorithm resumed at generation {start_gen + 1}")
        else:
            population = list(seeds)[:P]
            while len(population) < P:
                population.append(random_spec(rng, max_indicators=self.s.max_indicators, pool=self.pool))
        for gen in range(start_gen, self.s.generations):
            scored = []
            for spec in population:
                scored.append((self.score(spec), spec))
            scored.sort(key=lambda x: x[0], reverse=True)
            if scored[0][0] > best_score or best_spec is None:
                best_score, best_spec = scored[0][0], scored[0][1]
            self.ctx.set_task(f"genetic algorithm generation {gen + 1}/{self.s.generations}",
                              progress=(gen + 1) / self.s.generations)
            self.ctx.log(f"GA generation {gen + 1}: best train score {best_score:,.2f}")
            elite = [spec for _, spec in scored[:2]]
            children: list[StrategySpec] = list(elite)
            while len(children) < P:
                p1 = self._tournament(scored)
                p2 = self._tournament(scored)
                try:
                    child = crossover(p1, p2, rng, registry=REGISTRY) if rng.random() < 0.7 else p1
                    if rng.random() < 0.3:
                        child = mutate(child, rng, pool=self.pool)
                except ValueError:
                    child = p1
                children.append(child)
            population = children
            save_state({"population": [_spec_dict(s) for s in population], "generation": gen + 1,
                        "best": _spec_dict(best_spec), "best_score": best_score})
        self.ctx.log(f"genetic algorithm finished; best train score {best_score:,.2f}")
        return best_spec

    def _tournament(self, scored: list[tuple[float, StrategySpec]], k: int = 3) -> StrategySpec:
        idx = self._rng.choice(len(scored), size=min(k, len(scored)), replace=False)
        return max((scored[int(i)] for i in idx), key=lambda x: x[0])[1]

    def hill_climb(self, spec: StrategySpec) -> StrategySpec:
        self.ctx.set_task("hill climbing", algorithm="Hill Climbing", progress=0.0)
        best, best_score = spec, self.score(spec)
        for step in range(self.s.hill_climb_steps):
            neighbours = self._neighbours(best)
            if not neighbours:
                break
            scored = [(self.score(n), n) for n in neighbours]
            top_score, top = max(scored, key=lambda x: x[0])
            self.ctx.set_task(f"hill climbing step {step + 1}/{self.s.hill_climb_steps}",
                              progress=(step + 1) / self.s.hill_climb_steps)
            if top_score <= best_score:
                self.ctx.log(f"hill climbing reached a local optimum after {step} improving steps")
                break
            best, best_score = top, top_score
        self.ctx.log(f"hill climbing finished; train score {best_score:,.2f}")
        return best

    def _neighbours(self, spec: StrategySpec) -> list[StrategySpec]:
        out: list[StrategySpec] = []
        base = [{"name": c.name, "params": c.params_dict} for c in spec.indicators]
        for i, item in enumerate(base):
            definition = REGISTRY.get(item["name"])
            for p in definition.params:
                for value in p.neighbours(item["params"][p.name]):
                    changed = [dict(x, params=dict(x["params"])) for x in base]
                    changed[i]["params"][p.name] = value
                    out.append(self._safe(changed, spec, min_votes=spec.min_votes))
        for v in (spec.min_votes - 1, spec.min_votes + 1):
            if 1 <= v <= len(base):
                out.append(self._safe(base, spec, min_votes=v))
        for choices, attr in ((STOP_LOSS_CHOICES, "stop_loss_pct"), (TARGET_CHOICES, "target_pct")):
            current = getattr(spec, attr)
            if current in choices:
                j = choices.index(current)
                for k in (j - 1, j + 1):
                    if 0 <= k < len(choices):
                        kwargs = {"sl": spec.stop_loss_pct, "tp": spec.target_pct}
                        kwargs["sl" if attr == "stop_loss_pct" else "tp"] = choices[k]
                        out.append(self._safe(base, spec, min_votes=spec.min_votes, **kwargs))
        for d in DIRECTIONS:
            if d != spec.direction:
                out.append(self._safe(base, spec, min_votes=spec.min_votes, direction=d))
        unique: dict[str, StrategySpec] = {}
        for cand in out:
            if cand is not None and cand.strategy_hash != spec.strategy_hash:
                unique.setdefault(cand.strategy_hash, cand)
        return list(unique.values())

    def _safe(self, indicators: list[dict], base: StrategySpec, **kwargs) -> StrategySpec | None:
        try:
            return self._spec(indicators, direction=kwargs.get("direction", base.direction),
                              min_votes=kwargs.get("min_votes"),
                              sl=kwargs.get("sl", base.stop_loss_pct), tp=kwargs.get("tp", base.target_pct))
        except ValueError:
            return None

    def optuna_tuning(self, spec: StrategySpec) -> StrategySpec:
        if self.s.tuning_trials == 0:
            return spec
        self.ctx.set_task("Optuna parameter tuning", algorithm="Optuna TPE + MedianPruner", progress=0.0)
        study = optuna.create_study(
            study_name=f"agent_tune_{uuid.uuid4().hex[:10]}", storage=OPTUNA_STORAGE,
            direction="maximize", sampler=TPESampler(seed=self.s.seed),
            pruner=MedianPruner(n_startup_trials=5), load_if_exists=False)
        ctx, ev = self.ctx, self.ev

        def objective(trial: optuna.Trial) -> float:
            if not _gate(ctx, trial):
                raise optuna.TrialPruned("stop requested")
            indicators = []
            for c in spec.indicators:
                params = {}
                for p in REGISTRY.get(c.name).params:
                    if p.kind == "int":
                        params[p.name] = trial.suggest_int(f"{c.name}__{p.name}", p.low, p.high)
                    else:
                        params[p.name] = trial.suggest_float(f"{c.name}__{p.name}", p.low, p.high)
                indicators.append({"name": c.name, "params": params})
            cand = self._safe(indicators, spec, min_votes=trial.suggest_int("min_votes", 1, len(indicators)),
                              sl=trial.suggest_float("stop_loss_pct", 0.1, 1.0),
                              tp=trial.suggest_float("target_pct", 0.2, 2.0))
            if cand is None:
                return PENALTY
            cached = ev.cached(cand, "train")
            if cached is not None:
                ctx.duplicates_prevented += 1
                metrics = cached
            else:
                def on_chunk(k: int, cumulative: float) -> None:
                    trial.report(cumulative, k)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                try:
                    metrics = ev.evaluate(cand, "train", on_chunk=on_chunk)
                except optuna.TrialPruned:
                    ev.record_failure(cand, "train", agent=AGENT_NAME, error="pruned",
                                      study_name=study.study_name, trial_number=trial.number, status="pruned")
                    raise
                ev.record(cand, "train", metrics, agent=AGENT_NAME, study_name=study.study_name,
                          trial_number=trial.number)
            ctx.count_trial()
            trial.set_user_attr("strategy_hash", cand.strategy_hash)
            trial.set_user_attr("spec", cand.to_dict())
            return score_of(metrics)

        def progress(st: optuna.Study, _t) -> None:
            done = sum(1 for t in st.trials if t.state.is_finished())
            ctx.set_task(f"Optuna tuning {done}/{self.s.tuning_trials}", progress=done / self.s.tuning_trials)
            if ctx.stop_event.is_set():
                st.stop()

        study.optimize(objective, n_trials=self.s.tuning_trials, catch=(Exception,), callbacks=[progress])
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value > PENALTY]
        if not complete:
            self.ctx.log("Optuna tuning found no feasible parameters; keeping the hill-climbed strategy")
            return spec
        best = max(complete, key=lambda t: t.value)
        self.ctx.log(f"Optuna tuning best trial {best.number}: train score {best.value:,.2f}")
        return _from_dict(best.user_attrs["spec"])

    def walk_forward(self, candidates: list[StrategySpec]) -> None:
        self.ctx.set_task("walk-forward validation", algorithm="Walk-forward on validation split", progress=0.0)
        seen: set[str] = set()
        unique = []
        for c in candidates:
            if c.strategy_hash not in seen:
                seen.add(c.strategy_hash)
                unique.append(c)
        for k, spec in enumerate(unique[: self.s.walk_forward_top_k]):
            self.ctx.checkpoint()
            result = self.ev.walk_forward(spec, "validation")
            self.db.execute(
                "INSERT INTO validations(created_at, strategy_hash, dataset_id, segments_json,"
                " profitable_segments, total_segments, validation_net_profit, validated)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (utcnow(), spec.strategy_hash, self.s.dataset_id, self.db.dumps(result["segments"]),
                 result["profitable_segments"], result["total_segments"],
                 result["validation_net_profit"], int(result["validated"])))
            self.ctx.count_trial()
            self.ctx.set_task("walk-forward validation", progress=(k + 1) / min(len(unique), self.s.walk_forward_top_k))
            self.ctx.log(f"walk-forward {spec.strategy_hash}: {result['profitable_segments']}/"
                         f"{result['total_segments']} profitable segments, validated={result['validated']}")

    # -- orchestration ------------------------------------------------------------------------
    def run_cycle(self, cycle: int, state: dict) -> None:
        self._rng = np.random.default_rng(self.s.seed + cycle * 1009)
        done: dict[str, dict] = dict(state.get("done", {}))
        ga_state: dict = state.get("ga") or {}

        def persist() -> None:
            self.ctx.save_checkpoint({"cycle": cycle, "done": done, "ga": ga_state})

        def ga_save(payload):
            if payload is None:
                return ga_state if ga_state else None
            ga_state.clear()
            ga_state.update(payload)
            persist()
            return None

        def stage(name: str):
            return _from_dict(done[name]) if name in done else None

        best = stage("forward")
        if best is None:
            best = self.forward_selection()
            done["forward"] = _spec_dict(best)
            persist()
        if "backward" not in done:
            best = self.backward_elimination(best)
            done["backward"] = _spec_dict(best)
            persist()
        else:
            best = stage("backward")
        if "ga" not in done:
            seeds = [best]
            ranked = leaderboard("profit", self.s.dataset_id, top=5, db=self.db)
            seeds += [_from_dict(r["spec"]) for r in ranked]
            best = self.genetic_algorithm(seeds, ga_save)
            done["ga"] = _spec_dict(best)
            ga_state.clear()
            persist()
        else:
            best = stage("ga")
        if "hill" not in done:
            best = self.hill_climb(best)
            done["hill"] = _spec_dict(best)
            persist()
        else:
            best = stage("hill")
        if "tune" not in done:
            best = self.optuna_tuning(best)
            done["tune"] = _spec_dict(best)
            persist()
        else:
            best = stage("tune")
        if "walk_forward" not in done:
            candidates = [best] + [_from_dict(r["spec"]) for r in
                                   leaderboard("profit", self.s.dataset_id, top=self.s.walk_forward_top_k,
                                               db=self.db)]
            self.walk_forward(candidates)
            done["walk_forward"] = _spec_dict(best)
            persist()
        self.ctx.log(f"cycle {cycle + 1} finished; cycle best: {best.describe()}")

    def run(self) -> None:
        state = self.ctx.load_checkpoint() or {}
        cycle = int(state.get("cycle", 0))
        target = "continuous" if self.s.cycles == 0 else f"{self.s.cycles} cycle(s)"
        if state:
            self.ctx.log(f"continuing from checkpoint: {cycle} cycle(s) complete, "
                         f"phases done: {sorted(state.get('done', {})) or 'none'}")
        if self.s.cycles and cycle >= self.s.cycles:
            self.ctx.log(f"nothing to do: {cycle} cycle(s) already complete, target is {self.s.cycles}")
        self.ctx.log(f"research started on dataset {self.s.dataset_id} ({target})")
        while self.s.cycles == 0 or cycle < self.s.cycles:
            self.ctx.checkpoint()
            self.run_cycle(cycle, state if state.get("cycle") == cycle else {})
            state = {}
            cycle += 1
            self.ctx.save_checkpoint({"cycle": cycle, "done": {}, "ga": {}})
            if self.s.cycles == 0:
                self.ctx.sleep(5.0)
        self.ctx.set_task("finished", progress=1.0)
