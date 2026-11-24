"""Combination research on the selected NIFTY50 candles.

Pipeline (run inside one research window, always the train split unless stated otherwise):

1. Singles: every available indicator (studio or code registry) is backtested on the window.
   Only indicators with at least ``MIN_TRADES`` trades are eligible for ranking.
2. Spearman filter: eligible indicators are visited best-first (Wilson lower bound). An indicator is
   dropped when its signal series has |Spearman rho| >= ``corr_threshold`` with an already kept one.
   The correlation is computed on the window's own bars only.
3. Candidate pool: the first ``pool_size`` survivors. Every pair and triple of the pool is evaluated.
4. SFFS: sequential forward floating selection over the pool, sizes 1..``MAX_SIZE`` (5).
5. Ranking: Wilson lower bound, then Beta-Binomial shrunk score, then net profit.

Combination logic is AND: a bar votes long only when every member votes long, short only when every
member votes short, and neutral otherwise. The result is an ordinary vote series, so the same
backtest engine, costs and exit rules apply.

Walk-forward: the train span is cut into ``segments`` equal chunks. For fold f the selection window is
all chunks before chunk f, ending ``purge_bars`` before the test chunk; the test chunk is never used to
select. Every evaluation is cached by member code hashes, segment and rule set, so unchanged work is
not repeated and edited indicators are never reused.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import DEFAULT_SPLIT
from ..db import Database, get_db, utcnow
from .individual import (Rules, Source, Window, config_hash, discover_sources, evaluate_votes, load_window,
                         signal_votes, simulate)

MAX_SIZE = 5
DEFAULT_POOL = 12
DEFAULT_CORR = 0.80
DEFAULT_PURGE = 60           # bars removed between a selection window and its test chunk
DEFAULT_BUDGET = 4000        # uncached backtests per run (coverage report says when it is reached)
CURVE_POINTS = 1000          # downsampled points stored for equity and drawdown charts


class BudgetExceeded(Exception):
    pass


@dataclass(frozen=True)
class ComboRequest:
    dataset_id: int
    timeframe: str = "1min"
    start: str | None = None
    end: str | None = None
    rules: Rules = field(default_factory=Rules)
    segments: int = DEFAULT_SPLIT.walk_forward_segments
    pool_size: int = DEFAULT_POOL
    corr_threshold: float = DEFAULT_CORR
    purge_bars: int = DEFAULT_PURGE
    max_evaluations: int = DEFAULT_BUDGET
    top_n: int = 10

    def validate(self) -> None:
        self.rules.validate()
        if not 2 <= self.segments <= 10:
            raise ValueError("segments must be between 2 and 10")
        if not 3 <= self.pool_size <= 25:
            raise ValueError("pool_size must be between 3 and 25")
        if not 0.5 <= self.corr_threshold < 1.0:
            raise ValueError("corr_threshold must be between 0.5 and 1 (exclusive)")
        if not 0 <= self.purge_bars <= 5000:
            raise ValueError("purge_bars must be between 0 and 5000")
        if not 100 <= self.max_evaluations <= 200000:
            raise ValueError("max_evaluations must be between 100 and 200000")
        if not 1 <= self.top_n <= 50:
            raise ValueError("top_n must be between 1 and 50")


# -- combination logic --------------------------------------------------------------------------
def combo_key(members: list[Source]) -> str:
    raw = "|".join(sorted(f"{m.key}@{m.code_hash}" for m in members))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def and_votes(vote_list: list[np.ndarray]) -> np.ndarray:
    """AND logic: +1 only where every member is +1, -1 only where every member is -1, else 0."""
    stacked = np.sign(np.vstack([np.asarray(v, dtype=np.int16) for v in vote_list]))
    out = np.zeros(stacked.shape[1], dtype=np.int8)
    out[(stacked > 0).all(axis=0)] = 1
    out[(stacked < 0).all(axis=0)] = -1
    return out


def _member_dict(m: Source) -> dict:
    return {"key": m.key, "name": m.name, "source": m.source, "category": m.category, "code_hash": m.code_hash}


def _score_key(metrics: dict) -> tuple:
    return (metrics["wilson_lb"], metrics["shrunk_score"], metrics["net_profit"])


def _ranks(values: np.ndarray) -> np.ndarray:
    return pd.Series(values.astype(float)).rank(method="average").to_numpy()


def _rho(r1: np.ndarray, r2: np.ndarray) -> float:
    s1, s2 = r1.std(), r2.std()
    if s1 == 0 or s2 == 0:
        return float("nan")  # a constant signal has no rank correlation
    return float(np.corrcoef(r1, r2)[0, 1])


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


# -- evaluator with cache and budget ------------------------------------------------------------
class Evaluator:
    """Evaluates singles and combinations on bar ranges, using the database cache and a run budget."""

    def __init__(self, db: Database, win: Window, rules: Rules, cfg_hash: str, budget: int,
                 run_id: int | None, prefix_end: int | None = None, by_key: dict[str, Source] | None = None):
        self.db, self.win, self.rules, self.cfg_hash = db, win, rules, cfg_hash
        self.budget, self.run_id = budget, run_id
        self.fresh = 0
        self.cache_hits = 0
        self.by_key = by_key if by_key is not None else {s.key: s for s in discover_sources(db)[0]}
        self._votes: dict[str, np.ndarray] = {}
        self._prefix = win.full.iloc[:prefix_end if prefix_end is not None else win.eval_end]

    def votes(self, src: Source) -> np.ndarray:
        if src.key not in self._votes:
            self._votes[src.key] = signal_votes(src, self._prefix, self.by_key)
        return self._votes[src.key]

    def evaluate(self, members: list[Source], segment: str, a: int, b: int) -> dict:
        key = combo_key(members)
        row = self.db.one("SELECT metrics_json FROM combination_evaluations WHERE combo_key=? AND dataset_id=?"
                          " AND segment=? AND config_hash=? AND status='completed'",
                          (key, self.win.dataset_id, segment, self.cfg_hash))
        if row:
            self.cache_hits += 1
            return json.loads(row["metrics_json"])
        if self.fresh >= self.budget:
            raise BudgetExceeded(f"evaluation budget of {self.budget} reached")
        votes = self.votes(members[0]) if len(members) == 1 else and_votes([self.votes(m) for m in members])
        metrics = evaluate_votes(votes, self.win.full, a, b, self.rules)
        metrics["members"] = [_member_dict(m) for m in members]
        metrics["combo_key"] = key
        metrics["size"] = len(members)
        self.fresh += 1
        self.db.execute(
            "INSERT OR IGNORE INTO combination_evaluations(created_at, run_id, combo_key, size, members_json,"
            " dataset_id, segment, config_hash, status, metrics_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (utcnow(), self.run_id, key, len(members), json.dumps(metrics["members"]), self.win.dataset_id,
             segment, self.cfg_hash, "completed", json.dumps(metrics, default=str)))
        return metrics


# -- selection on one window --------------------------------------------------------------------
def select(ev: Evaluator, sources: list[Source], a: int, b: int, tag: str, *, corr_threshold: float,
           pool_size: int, top_n: int, checkpoint=None) -> dict:
    """Singles, Spearman filter, pairs, triples and SFFS on bars [a, b) of the window."""
    tick = checkpoint or (lambda: None)
    singles: dict[str, dict] = {}
    budget_hit = False
    try:
        for src in sources:
            tick()
            singles[src.key] = ev.evaluate([src], tag, a, b)
    except BudgetExceeded:
        budget_hit = True  # keep the singles evaluated so far; the coverage report says so
    eligible = [s for s in sources if s.key in singles and singles[s.key]["eligible_for_score"]]
    eligible.sort(key=lambda s: _score_key(singles[s.key]), reverse=True)

    # Spearman filter on the window's own signals, best-first
    ranks = {s.key: _ranks(ev.votes(s)[a:b]) for s in eligible}
    kept: list[Source] = []
    dropped: list[dict] = []
    for s in eligible:
        partner, worst = None, 0.0
        for k in kept:
            rho = _rho(ranks[s.key], ranks[k.key])
            if not math.isnan(rho) and abs(rho) >= corr_threshold and abs(rho) > abs(worst):
                partner, worst = k, rho
        if partner is None:
            kept.append(s)
        else:
            dropped.append({"name": s.name, "key": s.key, "kept_partner": partner.name,
                            "rho": round(worst, 4)})
    pool = kept[:pool_size]
    corr_keys = [s.key for s in pool]
    corr = []
    for i, s in enumerate(pool):
        row = []
        for t in pool:
            row.append(_clean(round(_rho(ranks[s.key], ranks[t.key]), 4)) if s.key != t.key else 1.0)
        corr.append(row)

    results: dict[str, dict] = {}
    for s in eligible:
        results[singles[s.key]["combo_key"]] = singles[s.key]
    pairs_total = math.comb(len(pool), 2)
    triples_total = math.comb(len(pool), 3)
    pairs_done = triples_done = sffs_done = 0
    try:
        for combo in itertools.combinations(pool, 2):
            tick()
            m = ev.evaluate(list(combo), tag, a, b)
            pairs_done += 1
            results[combo_key(list(combo))] = m
        for combo in itertools.combinations(pool, 3):
            tick()
            m = ev.evaluate(list(combo), tag, a, b)
            triples_done += 1
            results[combo_key(list(combo))] = m
        sffs_done = _sffs(ev, pool, a, b, tag, results, tick)
    except BudgetExceeded:
        budget_hit = True

    ranked_all = sorted((m for m in results.values() if m["eligible_for_score"]), key=_score_key, reverse=True)
    ranked = [{"rank": i + 1, **_ranked_row(m)} for i, m in enumerate(ranked_all[:top_n])]
    coverage = {
        "indicators_available": len(sources),
        "eligible": len(eligible),
        "dropped_by_correlation": len(dropped),
        "pool_size": len(pool),
        "singles_evaluated": len(singles),
        "pairs_possible": pairs_total, "pairs_evaluated": pairs_done,
        "triples_possible": triples_total, "triples_evaluated": triples_done,
        "sffs_evaluations": sffs_done,
        "combinations_ranked": len(ranked_all),
        "budget_reached": budget_hit,
        "max_size": MAX_SIZE,
    }
    return {"ranked": ranked, "best": ranked[0] if ranked else None, "dropped": dropped,
            "pool": [{"key": s.key, "name": s.name} for s in pool], "corr_keys": corr_keys, "corr": corr,
            "singles": [{"key": s.key, "name": s.name, **_ranked_row(singles[s.key])} for s in sources if s.key in singles],
            "coverage": coverage}


def _ranked_row(m: dict) -> dict:
    return {"combo_key": m["combo_key"], "size": m["size"], "members": m["members"],
            "metrics": {k: m[k] for k in m if k not in ("members", "combo_key", "size")}}


def _sffs(ev: Evaluator, pool: list[Source], a: int, b: int, tag: str, results: dict, tick) -> int:
    """Sequential forward floating selection up to MAX_SIZE members. Returns evaluations requested."""
    if not pool:
        return 0
    count = 0

    def score(members: list[Source]) -> dict:
        nonlocal count
        count += 1
        tick()
        m = ev.evaluate(members, tag, a, b)
        results[combo_key(members)] = m
        return m

    singles = [(score([s]), s) for s in pool]
    current = [max(singles, key=lambda t: _score_key(t[0]))[1]]
    best_at = {1: _score_key(score(current))}
    while len(current) < min(MAX_SIZE, len(pool)):
        trials = []
        for c in pool:
            if c in current:
                continue
            trials.append((_score_key(score(current + [c])), c))
        if not trials:
            break
        top_val, top_c = max(trials, key=lambda t: t[0])
        current = current + [top_c]
        size = len(current)
        best_at[size] = max(best_at.get(size, top_val), top_val)
        # floating step: drop members while a smaller subset beats the best seen at that size
        while len(current) > 2:
            drops = [(_score_key(score([m for m in current if m is not r])), r) for r in current]
            val, rem = max(drops, key=lambda t: t[0])
            if val > best_at.get(len(current) - 1, (-1e9,)):
                current = [m for m in current if m is not rem]
                best_at[len(current)] = max(best_at.get(len(current), val), val)
            else:
                break
    return count


# -- walk-forward folds -------------------------------------------------------------------------
def fold_plan(a: int, b: int, segments: int, purge: int) -> list[dict]:
    """Expanding-window folds over the train span [a, b). Fold f selects on chunks 0..f-1 (ending
    ``purge`` bars before chunk f) and tests on chunk f."""
    edges = np.linspace(a, b, segments + 1).astype(int)
    plan = []
    for f in range(1, segments):
        sel_a, sel_b = int(edges[0]), int(edges[f]) - purge
        test_a, test_b = int(edges[f]), int(edges[f + 1])
        plan.append({"fold": f, "sel": (sel_a, sel_b), "test": (test_a, test_b), "purge": purge})
    return plan


def _curve(res, df) -> dict:
    eq = np.asarray(res.equity, dtype=float)
    if len(eq) == 0:
        return {"time": [], "equity": [], "drawdown_pct": []}
    peak = np.maximum.accumulate(eq)
    dd = np.where(peak > 0, (eq - peak) / peak * 100.0, 0.0)
    step = max(1, math.ceil(len(eq) / CURVE_POINTS))
    idx = np.arange(0, len(eq), step)
    if idx[-1] != len(eq) - 1:
        idx = np.append(idx, len(eq) - 1)
    times = [df.index[i].isoformat() for i in idx]
    return {"time": times, "equity": [round(float(eq[i]), 2) for i in idx],
            "drawdown_pct": [round(float(dd[i]), 4) for i in idx]}


def combo_curve(ev: Evaluator, members: list[Source], a: int, b: int) -> dict:
    votes = ev.votes(members[0]) if len(members) == 1 else and_votes([ev.votes(m) for m in members])
    res, _cfg, df = simulate(votes, ev.win.full, a, b, ev.rules)
    return _curve(res, df)


def curve_for_frame(votes: np.ndarray, frame: pd.DataFrame, a: int, b: int, rules: Rules) -> dict:
    res, _cfg, df = simulate(votes, frame, a, b, rules)
    return _curve(res, df)


def prepare_window(req: ComboRequest, db: Database | None = None) -> tuple[Window, str]:
    req.validate()
    db = db or get_db()
    win = load_window(req.dataset_id, req.timeframe, "train", req.start, req.end, db)
    ds = db.one("SELECT rows, end_ts FROM datasets WHERE id=?", (req.dataset_id,))
    cfg = config_hash({"timeframe": req.timeframe, "start": req.start, "end": req.end, "split": "train",
                       "eval_start": win.start_ts, "eval_end": win.end_ts, "bars": win.bars_in_split,
                       "segments": req.segments, "pool": req.pool_size, "corr": req.corr_threshold,
                       "purge": req.purge_bars, "algo": "combination-v1"},
                      req.rules, ds["rows"], ds["end_ts"])
    return win, cfg
