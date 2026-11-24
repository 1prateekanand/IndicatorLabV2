"""Strategy specification, canonical hashing, vote computation and genetic operators.

A strategy is a set of indicators (each with parameters). Each indicator votes +1, -1 or 0 per bar.
The strategy enters long when the vote total reaches ``min_votes`` and short when it falls to
``-min_votes``. This simple, explicit rule keeps every combination comparable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .registry import REGISTRY, IndicatorRegistry

MAX_INDICATORS = 6
STOP_LOSS_CHOICES = (0.0, 0.2, 0.3, 0.5, 0.8, 1.0)
TARGET_CHOICES = (0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0)
DIRECTIONS = ("both", "long", "short")


def spec_from_dict(data: dict, registry: IndicatorRegistry = REGISTRY) -> "StrategySpec":
    """Rebuild a spec from its stored dict (``StrategySpec.to_dict`` output)."""
    return StrategySpec.build(data["indicators"], min_votes=data["min_votes"], direction=data["direction"],
                              stop_loss_pct=data["stop_loss_pct"], target_pct=data["target_pct"],
                              registry=registry)


@dataclass(frozen=True)
class IndicatorConfig:
    name: str
    params: tuple[tuple[str, float | int], ...] = ()

    @property
    def params_dict(self) -> dict:
        return dict(self.params)


@dataclass(frozen=True)
class StrategySpec:
    indicators: tuple[IndicatorConfig, ...]
    min_votes: int = 1
    direction: str = "both"
    stop_loss_pct: float = 0.5
    target_pct: float = 1.0

    @staticmethod
    def build(indicators: list[dict | IndicatorConfig], *, min_votes: int = 1, direction: str = "both",
              stop_loss_pct: float = 0.5, target_pct: float = 1.0,
              registry: IndicatorRegistry = REGISTRY) -> "StrategySpec":
        """Normalise user input: fill defaults, clamp params, sort indicators by name."""
        configs = []
        for item in indicators:
            if isinstance(item, IndicatorConfig):
                name, params = item.name, item.params_dict
            else:
                name, params = item["name"], item.get("params") or {}
            try:
                definition = registry.get(name)
            except KeyError as exc:
                raise ValueError(str(exc)) from exc
            if not definition.available:
                raise ValueError(f"indicator '{name}' is unavailable: {definition.status_reason}")
            clean = definition.clean_params(params)
            configs.append(IndicatorConfig(name, tuple(sorted(clean.items()))))
        configs.sort(key=lambda c: c.name)
        spec = StrategySpec(tuple(configs), int(min_votes), direction,
                            round(float(stop_loss_pct), 4), round(float(target_pct), 4))
        spec.validate()
        return spec

    def validate(self) -> None:
        if not 1 <= len(self.indicators) <= MAX_INDICATORS:
            raise ValueError(f"a strategy needs 1 to {MAX_INDICATORS} indicators")
        names = [c.name for c in self.indicators]
        if len(set(names)) != len(names):
            raise ValueError("an indicator may appear only once per strategy")
        if not 1 <= self.min_votes <= len(self.indicators):
            raise ValueError("min_votes must be between 1 and the number of indicators")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        if self.stop_loss_pct < 0 or self.target_pct < 0 or self.stop_loss_pct > 10 or self.target_pct > 20:
            raise ValueError("stop-loss and target percentages are out of range")

    def to_dict(self) -> dict:
        return {
            "indicators": [{"name": c.name, "params": c.params_dict} for c in self.indicators],
            "min_votes": self.min_votes,
            "direction": self.direction,
            "stop_loss_pct": self.stop_loss_pct,
            "target_pct": self.target_pct,
        }

    @property
    def strategy_hash(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]

    def describe(self) -> str:
        parts = [f"{c.name}({', '.join(f'{k}={v}' for k, v in c.params)})" for c in self.indicators]
        return f"{' + '.join(parts)} | votes>={self.min_votes} | {self.direction} | " \
               f"SL {self.stop_loss_pct}% TP {self.target_pct}%"


# -- vote computation with a process-wide cache of indicator signal arrays -------------------
_SIGNAL_CACHE: dict[tuple, np.ndarray] = {}
_SIGNAL_CACHE_LIMIT = 4000


def indicator_signal(dataset_id: int, df: pd.DataFrame, name: str,
                     params: tuple[tuple[str, float | int], ...]) -> np.ndarray:
    key = (dataset_id, name, params)
    cached = _SIGNAL_CACHE.get(key)
    if cached is not None:
        return cached
    definition = REGISTRY.get(name)
    series = definition.compute(df, **dict(params))
    values = series.fillna(0).to_numpy(dtype=np.int8)
    if len(_SIGNAL_CACHE) >= _SIGNAL_CACHE_LIMIT:
        _SIGNAL_CACHE.pop(next(iter(_SIGNAL_CACHE)))
    _SIGNAL_CACHE[key] = values
    return values


def compute_votes(dataset_id: int, df: pd.DataFrame, spec: StrategySpec) -> np.ndarray:
    total = np.zeros(len(df), dtype=int)
    for config in spec.indicators:
        total += indicator_signal(dataset_id, df, config.name, config.params).astype(int)
    return total


def clear_signal_cache() -> None:
    _SIGNAL_CACHE.clear()


# -- random generation and genetic operators ----------------------------------------------------
def random_spec(rng: np.random.Generator, *, registry: IndicatorRegistry = REGISTRY,
                max_indicators: int = 4, pool: list[str] | None = None) -> StrategySpec:
    names = pool or [d.name for d in registry.available()]
    k = int(rng.integers(1, min(max_indicators, len(names)) + 1))
    chosen = rng.choice(names, size=k, replace=False).tolist()
    indicators = []
    for name in chosen:
        definition = registry.get(name)
        params = {p.name: p.sample(rng) for p in definition.params}
        indicators.append({"name": name, "params": params})
    return StrategySpec.build(
        indicators, min_votes=int(rng.integers(1, k + 1)),
        direction=str(rng.choice(DIRECTIONS)),
        stop_loss_pct=float(rng.choice(STOP_LOSS_CHOICES)),
        target_pct=float(rng.choice(TARGET_CHOICES)), registry=registry)


def mutate(spec: StrategySpec, rng: np.random.Generator, *, registry: IndicatorRegistry = REGISTRY,
           pool: list[str] | None = None) -> StrategySpec:
    data = spec.to_dict()
    ops = ["param", "param", "add", "remove", "votes", "risk"]
    op = str(rng.choice(ops))
    names = pool or [d.name for d in registry.available()]
    if op == "param" and data["indicators"]:
        item = data["indicators"][int(rng.integers(len(data["indicators"])))]
        definition = registry.get(item["name"])
        if definition.params:
            p = definition.params[int(rng.integers(len(definition.params)))]
            item["params"][p.name] = p.sample(rng)
    elif op == "add" and len(data["indicators"]) < MAX_INDICATORS:
        used = {i["name"] for i in data["indicators"]}
        free = [n for n in names if n not in used]
        if free:
            name = str(rng.choice(free))
            definition = registry.get(name)
            data["indicators"].append({"name": name, "params": {p.name: p.sample(rng) for p in definition.params}})
    elif op == "remove" and len(data["indicators"]) > 1:
        data["indicators"].pop(int(rng.integers(len(data["indicators"]))))
    elif op == "votes":
        data["min_votes"] = int(rng.integers(1, len(data["indicators"]) + 1))
    elif op == "risk":
        if rng.random() < 0.5:
            data["stop_loss_pct"] = float(rng.choice(STOP_LOSS_CHOICES))
        else:
            data["target_pct"] = float(rng.choice(TARGET_CHOICES))
    data["min_votes"] = min(data["min_votes"], len(data["indicators"]))
    return StrategySpec.build(data["indicators"], min_votes=data["min_votes"],
                              direction=data["direction"], stop_loss_pct=data["stop_loss_pct"],
                              target_pct=data["target_pct"], registry=registry)


def crossover(a: StrategySpec, b: StrategySpec, rng: np.random.Generator, *,
              registry: IndicatorRegistry = REGISTRY) -> StrategySpec:
    by_name = {c.name: c for c in a.indicators}
    for c in b.indicators:
        by_name.setdefault(c.name, c)
    chosen = [c for c in by_name.values() if rng.random() < 0.5] or [list(by_name.values())[0]]
    chosen = chosen[:MAX_INDICATORS]
    parent = a if rng.random() < 0.5 else b
    votes = min(parent.min_votes, len(chosen))
    return StrategySpec.build(
        [{"name": c.name, "params": c.params_dict} for c in chosen],
        min_votes=max(1, votes), direction=parent.direction,
        stop_loss_pct=a.stop_loss_pct if rng.random() < 0.5 else b.stop_loss_pct,
        target_pct=a.target_pct if rng.random() < 0.5 else b.target_pct, registry=registry)
