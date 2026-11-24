"""Individual indicator evaluation on the selected NIFTY50 candles.

Discovery (nothing is hardcoded):
    * Indicator Studio items with status registered and available      source = "studio"
    * Code-registry indicators (built-in plugins) that are available   source = "code"
    * Anything else is listed as unavailable with its reason, so it is visible but not evaluated.

Signal conversion: an indicator's output is read as a vote per bar.
    +1 -> long signal, -1 -> short signal, 0 -> no signal (binary 0/1 outputs act as long-only).
A single indicator enters when its vote is non-zero (min_votes = 1) and reverses on the opposite vote.

Trade simulation: the same engine as the backtester. Entry at the next bar's open, stop-loss and target
checked intrabar, optional holding-period exit, brokerage, transaction charges and slippage from the
configured cost model. Win = trade with positive net P&L after costs.

Split: the train part of the selected range only. Validation and OOS are not used here.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import data
from ..config import DEFAULT_BACKTEST, DEFAULT_SPLIT, BacktestConfig, CostModel
from ..db import Database, get_db, utcnow
from ..engine import Exit, run_backtest
from ..metrics import compute_metrics
from ..registry import REGISTRY
from ..studio import list_items, run_sandbox
from .scoring import shrunk_scores

ALLOWED_SPLITS = ("train", "validation")  # OOS is deliberately not available to research
TIMEFRAMES = {"1min": "1min", "5min": "5min", "10min": "10min", "1h": "1h", "4h": "4h"}
DIRECTIONS = ("both", "long", "short")
SANDBOX_CPU_S = 900
SANDBOX_MEM_MB = 4096
SANDBOX_TIMEOUT_S = 900


# -- rules --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Rules:
    direction: str = "both"
    stop_loss_pct: float = 0.5
    target_pct: float = 1.0
    max_hold_bars: int | None = None
    brokerage_per_order: float = DEFAULT_BACKTEST.costs.brokerage_per_order
    charges_pct: float = DEFAULT_BACKTEST.costs.charges_pct
    slippage_points: float = DEFAULT_BACKTEST.costs.slippage_points

    def validate(self) -> None:
        if self.direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}")
        if not 0 <= self.stop_loss_pct <= 10 or not 0 <= self.target_pct <= 20:
            raise ValueError("stop-loss must be 0-10% and target 0-20%")
        if self.max_hold_bars is not None and self.max_hold_bars < 1:
            raise ValueError("max_hold_bars must be at least 1 or null")
        if min(self.brokerage_per_order, self.charges_pct, self.slippage_points) < 0:
            raise ValueError("costs cannot be negative")

    def backtest_config(self) -> BacktestConfig:
        costs = CostModel(brokerage_per_order=self.brokerage_per_order, charges_pct=self.charges_pct,
                          slippage_points=self.slippage_points)
        return dataclasses.replace(DEFAULT_BACKTEST, costs=costs)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


# -- sources ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    key: str            # stable id: "studio:<id>" or "code:<name>"
    name: str
    source: str         # "studio" | "code"
    category: str
    code_hash: str      # changes whenever the indicator's code or parameters change
    params: dict = field(default_factory=dict)
    code: str | None = None
    available: bool = True
    status_reason: str = "ok"
    spec: dict | None = None   # conditions-based signal rules; None means the indicator's own votes


def _hash(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:16]


def discover_sources(db: Database | None = None) -> tuple[list[Source], list[Source]]:
    """Return (available, unavailable). Discovery is live: nothing is cached between calls."""
    available: list[Source] = []
    unavailable: list[Source] = []
    for d in REGISTRY.all():
        params = {p.name: p.default for p in d.params}
        src = Source(key=f"code:{d.name}", name=d.name, source="code", category=d.category,
                     code_hash=_hash(d.name, d.module, json.dumps(params, sort_keys=True)),
                     params=params, available=d.available, status_reason=d.status_reason or "ok")
        (available if d.available else unavailable).append(src)
    for item in list_items("indicator", db):
        params = {p["name"]: p["default"] for p in item.get("params", [])}
        ok = bool(item.get("registered") and item.get("available"))
        src = Source(key=f"studio:{item['id']}", name=item["name"], source="studio",
                     category=item.get("category") or "Custom",
                     code_hash=_hash(item.get("code", ""), json.dumps(params, sort_keys=True), item.get("version", "")),
                     params=params, code=item.get("code"), available=ok,
                     status_reason=item.get("status_reason") or ("ok" if ok else "not registered"))
        (available if ok else unavailable).append(src)
    specs = {r["source_key"]: r["spec_json"] for r in (db or get_db()).query("SELECT source_key, spec_json FROM signal_specs")}
    available = [_with_spec(s, specs) for s in available]
    unavailable = [_with_spec(s, specs) for s in unavailable]
    available.sort(key=lambda s: (s.source, s.name))
    return available, unavailable


def _with_spec(src: Source, specs: dict[str, str]) -> Source:
    raw = specs.get(src.key)
    if raw is None:
        return src
    spec = json.loads(raw)
    if spec.get("mode") != "conditions":
        return src
    # the spec is part of the cache key: changing a rule never reuses an old result
    return dataclasses.replace(src, spec=spec, code_hash=_hash(src.code_hash, raw))


# -- signals ------------------------------------------------------------------------------------
def frame_to_candles(frame: pd.DataFrame) -> list[dict]:
    return [{"time": t.isoformat(), "open": float(o), "high": float(h), "low": float(l), "close": float(c),
             "volume": float(v)}
            for t, o, h, l, c, v in zip(frame.index, frame["open"], frame["high"], frame["low"],
                                        frame["close"], frame["volume"])]


def compute_votes(src: Source, prefix: pd.DataFrame) -> np.ndarray:
    """Vote per bar on the candles in ``prefix`` (which ends at the evaluation boundary, so the
    computation is causal and uses no later bars)."""
    if src.source == "code":
        definition = REGISTRY.get(src.name)
        series = definition.compute(prefix, **src.params)
        return series.fillna(0).to_numpy(dtype=np.int8)
    result = run_sandbox({"mode": "indicator", "code": src.code, "candles": frame_to_candles(prefix),
                          "params": src.params, "syntax_only": False, "return_signals": True},
                         cpu_limit_s=SANDBOX_CPU_S, mem_limit_mb=SANDBOX_MEM_MB, timeout_s=SANDBOX_TIMEOUT_S)
    if not result.get("ok"):
        err = result.get("error") or {}
        raise RuntimeError(f"{err.get('type', 'SandboxError')} (line {err.get('line', 1)}): {err.get('message')}")
    if result.get("continuous"):
        raise RuntimeError("this indicator returns continuous values; add a signal rule for it on the Indicator page")
    signals = result.get("signals")
    if signals is None or len(signals) != len(prefix):
        raise RuntimeError("indicator did not return one signal per candle")
    return np.asarray(signals, dtype=np.int8)


# -- signal votes (legacy votes or configurable conditions on raw output) -----------------------------
def raw_series(src: Source, prefix: pd.DataFrame) -> np.ndarray:
    """Continuous output of one indicator on the candles in ``prefix`` (NaN where undefined)."""
    if src.source == "code":
        definition = REGISTRY.get(src.name)
        return definition.compute(prefix, **src.params).astype(float).to_numpy()
    result = run_sandbox({"mode": "indicator", "code": src.code, "candles": frame_to_candles(prefix),
                          "params": src.params, "syntax_only": False, "return_signals": False, "return_raw": True},
                         cpu_limit_s=SANDBOX_CPU_S, mem_limit_mb=SANDBOX_MEM_MB, timeout_s=SANDBOX_TIMEOUT_S)
    if not result.get("ok"):
        err = result.get("error") or {}
        raise RuntimeError(f"{err.get('type', 'SandboxError')} (line {err.get('line', 1)}): {err.get('message')}")
    raw = result.get("raw")
    if raw is None or len(raw) != len(prefix):
        raise RuntimeError("indicator did not return one value per candle")
    return np.array([np.nan if v is None else v for v in raw], dtype=float)


def signal_votes(src: Source, prefix: pd.DataFrame, by_key: dict[str, Source] | None = None) -> np.ndarray:
    """Votes for one source. Legacy sources use their own votes. Sources with a conditions spec are
    evaluated from raw output: a comparison with another indicator uses that indicator's raw value,
    which does not depend on its own rules, so there are no chains and no cycles."""
    if src.spec is None:
        return compute_votes(src, prefix)
    from .signals import references, votes_from_raw
    by_key = by_key or {}
    cache: dict[str, np.ndarray] = {src.key: raw_series(src, prefix)}

    def raw_of(key: str) -> np.ndarray:
        if key not in cache:
            other = by_key.get(key)
            if other is None:
                raise ValueError(f"signal rule refers to '{key}', which is not an available indicator")
            cache[key] = raw_series(other, prefix)
        return cache[key]

    for ref in references(src.spec):
        raw_of(ref)
    return votes_from_raw(src.spec, cache[src.key], raw_of)


# -- dataset window -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Window:
    dataset_id: int
    dataset_name: str
    symbol: str
    timeframe: str
    full: pd.DataFrame        # resampled candles of the whole dataset (causal prefix source)
    split: str                # "train" or "validation"
    eval_start: int           # first bar of the evaluated split (positional)
    eval_end: int             # one past the last bar (positional)
    start_ts: str
    end_ts: str
    bars_in_split: int


def dataset_frame(dataset_id: int, timeframe: str, db: Database | None = None) -> pd.DataFrame:
    """Candles of the whole dataset at ``timeframe`` (OHLC resampled exactly, no range filter)."""
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {tuple(TIMEFRAMES)}")
    db = db or get_db()
    if db.one("SELECT id FROM datasets WHERE id=?", (int(dataset_id),)) is None:
        raise LookupError(f"dataset {dataset_id} not found")
    frame = data.load_candles(int(dataset_id))
    if timeframe != "1min":
        frame = frame.resample(TIMEFRAMES[timeframe], label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return frame


def load_window(dataset_id: int, timeframe: str, split: str, start: str | None = None,
                end: str | None = None, db: Database | None = None) -> Window:
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {ALLOWED_SPLITS}; OOS is not available to research")
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"timeframe must be one of {tuple(TIMEFRAMES)}")
    db = db or get_db()
    ds = db.one("SELECT * FROM datasets WHERE id=?", (int(dataset_id),))
    if ds is None:
        raise LookupError(f"dataset {dataset_id} not found")
    frame = dataset_frame(int(dataset_id), timeframe, db)
    lo, hi = 0, len(frame)
    try:
        if start:
            lo = int(frame.index.searchsorted(pd.Timestamp(start)))
        if end:
            hi = int(frame.index.searchsorted(pd.Timestamp(end), side="right"))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid date range: {exc}") from exc
    if hi - lo < 50:
        raise ValueError("the selected range holds fewer than 50 candles")
    span = hi - lo
    if split == "train":
        eval_start, eval_end = lo, lo + int(span * DEFAULT_SPLIT.train_end)
    else:
        eval_start = lo + int(span * DEFAULT_SPLIT.train_end)
        eval_end = lo + int(span * DEFAULT_SPLIT.validation_end)
    return Window(dataset_id=int(dataset_id), dataset_name=ds["name"], symbol=ds["symbol"], timeframe=timeframe,
                  full=frame, split=split, eval_start=eval_start, eval_end=eval_end,
                  start_ts=frame.index[eval_start].isoformat(), end_ts=frame.index[eval_end - 1].isoformat(),
                  bars_in_split=eval_end - eval_start)


def config_hash(window_spec: dict, rules: Rules, dataset_rows: int, dataset_end: str) -> str:
    raw = json.dumps({"window": window_spec, "rules": rules.to_dict(), "rows": dataset_rows,
                      "dataset_end": dataset_end}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# -- evaluation ---------------------------------------------------------------------------------
def simulate(votes_full: np.ndarray, full: pd.DataFrame, a: int, b: int, rules: Rules):
    """Run the backtest on bars [a, b). ``votes_full`` must be causal (computed on a prefix that ends
    at or after ``b``). The slice starts flat and is closed at its last bar."""
    df = full.iloc[a:b]
    votes = votes_full[a:b].astype(int)
    cfg = rules.backtest_config()
    res = run_backtest(df, votes, min_votes=1, direction=rules.direction,
                       exit_rules=Exit(stop_loss_pct=rules.stop_loss_pct, target_pct=rules.target_pct),
                       cfg=cfg, max_hold_bars=rules.max_hold_bars)
    return res, cfg, df


def evaluate_votes(votes_full: np.ndarray, full: pd.DataFrame, a: int, b: int, rules: Rules,
                   eval_start_ts: str | None = None, eval_end_ts: str | None = None) -> dict:
    """Metrics of ``simulate`` for bars [a, b)."""
    res, cfg, df = simulate(votes_full, full, a, b, rules)
    start = eval_start_ts or (df.index[0].isoformat() if len(df) else None)
    end = eval_end_ts or (df.index[-1].isoformat() if len(df) else None)
    return summarize(res, cfg, len(df), start, end)


def summarize(res, cfg, bars: int, start_ts: str | None, end_ts: str | None) -> dict:
    m = compute_metrics(res.trades, res.equity, cfg.initial_capital, res.final_capital)
    wins = int(sum(1 for t in res.trades if t["net_pnl"] > 0))
    trades = len(res.trades)
    losses = trades - wins
    scores = shrunk_scores(wins, trades)
    return {
        "trades": trades, "wins": wins, "losses": losses,
        "raw_win_rate": round(100.0 * wins / trades, 4) if trades else None,
        "shrunk_score": round(scores["shrunk_score"], 6), "wilson_lb": round(scores["wilson_lb"], 6),
        "eligible_for_score": scores["eligible_for_score"],
        "net_profit": round(float(m["net_profit"]), 2), "net_return_pct": round(float(m["return_pct"]), 4),
        "profit_factor": m["profit_factor"], "max_drawdown_pct": round(float(m["max_drawdown_pct"]), 4),
        "total_costs": round(float(m["total_costs"]), 2), "avg_trade": round(float(m["avg_trade"]), 2),
        "longs": int(m["longs"]), "shorts": int(m["shorts"]),
        "exit_reasons": m["exit_reasons"], "skipped_entries": int(res.skipped_entries),
        "bars_evaluated": int(bars), "eval_start": start_ts, "eval_end": end_ts,
    }


def evaluate_source(src: Source, win: Window, rules: Rules, by_key: dict[str, Source] | None = None) -> dict:
    """Trades, wins, losses and scores for one indicator on the window's evaluated split."""
    prefix = win.full.iloc[:win.eval_end]
    if src.spec is not None and by_key is None:
        by_key = {s.key: s for s in discover_sources()[0]}
    votes_full = signal_votes(src, prefix, by_key)
    return evaluate_votes(votes_full, win.full, win.eval_start, win.eval_end, rules,
                          win.start_ts, win.end_ts)


def store_result(db: Database, *, run_id: int | None, src: Source, win: Window, cfg_hash: str,
                 status: str, metrics: dict | None, error: str | None = None) -> None:
    db.execute(
        "INSERT OR IGNORE INTO indicator_evaluations(created_at, run_id, source_key, name, source, code_hash,"
        " dataset_id, split, config_hash, status, error, metrics_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (utcnow(), run_id, src.key, src.name, src.source, src.code_hash, win.dataset_id, win.split,
         cfg_hash, status, (error or None) and error[:500], json.dumps(metrics, default=str) if metrics else None))


def cached_result(db: Database, src: Source, dataset_id: int, split: str, cfg_hash: str) -> dict | None:
    row = db.one("SELECT metrics_json FROM indicator_evaluations WHERE source_key=? AND code_hash=? AND dataset_id=?"
                 " AND split=? AND config_hash=? AND status='completed'",
                 (src.key, src.code_hash, dataset_id, split, cfg_hash))
    return json.loads(row["metrics_json"]) if row else None


def latest_any_hash_results(db: Database, dataset_id: int, split: str, cfg_hash: str) -> dict[str, dict]:
    """Most recent completed result per source under this configuration, whatever its code hash.

    Used only to report that an indicator changed after its last result. It is never used as a score.
    """
    rows = db.query(
        "SELECT e.* FROM indicator_evaluations e WHERE e.dataset_id=? AND e.split=? AND e.config_hash=?"
        " AND e.status='completed' AND e.id = (SELECT MAX(id) FROM indicator_evaluations x WHERE x.source_key=e.source_key"
        " AND x.dataset_id=e.dataset_id AND x.split=e.split AND x.config_hash=e.config_hash AND x.status='completed')",
        (dataset_id, split, cfg_hash))
    return {r["source_key"]: {"id": r["id"], "code_hash": r["code_hash"], "created_at": r["created_at"]} for r in rows}


def latest_results(db: Database, dataset_id: int, split: str, cfg_hash: str) -> dict[str, dict]:
    """Completed results for the current code of each source under this exact configuration."""
    rows = db.query(
        "SELECT e.* FROM indicator_evaluations e WHERE e.dataset_id=? AND e.split=? AND e.config_hash=?"
        " AND e.status='completed' AND e.id = (SELECT MAX(id) FROM indicator_evaluations x WHERE x.source_key=e.source_key"
        " AND x.code_hash=e.code_hash AND x.dataset_id=e.dataset_id AND x.split=e.split AND x.config_hash=e.config_hash"
        " AND x.status='completed')",
        (dataset_id, split, cfg_hash))
    return {r["source_key"]: {"id": r["id"], "name": r["name"], "source": r["source"], "code_hash": r["code_hash"],
                              "created_at": r["created_at"], "metrics": json.loads(r["metrics_json"])}
            for r in rows}
