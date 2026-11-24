"""Strategy evaluation on chronological splits, plus experiment persistence with deduplication.

Splits (by bar position, chronological):
    train       [0, train_end)               used by Agent 1 and Optuna for search
    validation  [train_end, validation_end)  used for walk-forward validation
    oos         [validation_end, end)        untouched; only the explicit OOS endpoint reads it

Indicator signals are computed on the full series and then sliced. Every indicator is causal,
so values inside a slice never depend on bars that come later.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .config import DEFAULT_BACKTEST, DEFAULT_SPLIT, MIN_TRADES, BacktestConfig, SplitConfig
from .data import load_candles
from .db import Database, get_db, utcnow
from .engine import BacktestResult, Exit, run_backtest
from .metrics import compute_metrics
from .overfitting import return_moments
from .strategy import StrategySpec, compute_votes

SPLITS = ("train", "validation", "oos")


class TrialCallback(Exception):
    """Raised by an ``on_chunk`` hook to abort a chunked evaluation (used for pruning)."""


@dataclass(frozen=True)
class ResearchConfig:
    backtest: BacktestConfig = DEFAULT_BACKTEST
    split: SplitConfig = DEFAULT_SPLIT
    min_trades: int = MIN_TRADES

    def fingerprint(self) -> str:
        raw = json.dumps({"bt": asdict(self.backtest), "split": asdict(self.split),
                          "min_trades": self.min_trades}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class Evaluator:
    def __init__(self, dataset_id: int, research: ResearchConfig | None = None,
                 db: Database | None = None):
        self.dataset_id = dataset_id
        self.research = research or ResearchConfig()
        self.db = db or get_db()
        self.candles = load_candles(dataset_id)
        n = len(self.candles)
        s = self.research.split
        self.bounds = {
            "train": (0, int(n * s.train_end)),
            "validation": (int(n * s.train_end), int(n * s.validation_end)),
            "oos": (int(n * s.validation_end), n),
        }
        self.config_hash = self.research.fingerprint()

    # -- slicing ------------------------------------------------------------------------------
    def slice_bounds(self, split: str) -> tuple[int, int]:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}")
        return self.bounds[split]

    def split_frame(self, split: str) -> pd.DataFrame:
        a, b = self.slice_bounds(split)
        return self.candles.iloc[a:b]

    def _chunks(self, split: str) -> list[tuple[int, int]]:
        a, b = self.slice_bounds(split)
        edges = np.linspace(a, b, self.research.split.train_chunks + 1).astype(int)
        return [(int(edges[k]), int(edges[k + 1])) for k in range(len(edges) - 1)]

    # -- evaluation ---------------------------------------------------------------------------
    def _votes(self, spec: StrategySpec) -> np.ndarray:
        return compute_votes(self.dataset_id, self.candles, spec)

    def _exit(self, spec: StrategySpec) -> Exit:
        return Exit(stop_loss_pct=spec.stop_loss_pct, target_pct=spec.target_pct)

    def simulate(self, spec: StrategySpec, split: str) -> tuple[BacktestResult, dict]:
        """One continuous run across a whole split (used for display and walk-forward segments)."""
        a, b = self.slice_bounds(split)
        votes = self._votes(spec)[a:b]
        df = self.candles.iloc[a:b]
        res = run_backtest(df, votes, min_votes=spec.min_votes, direction=spec.direction,
                           exit_rules=self._exit(spec), cfg=self.research.backtest)
        return res, self._metrics(res, self.research.backtest.initial_capital)

    def evaluate(self, spec: StrategySpec, split: str,
                 on_chunk: Callable[[int, float], None] | None = None) -> dict:
        """Chunked evaluation used for search. Capital is carried between chunks.

        Each chunk is closed flat at its end, so the same procedure is applied to every strategy
        and results are comparable. ``on_chunk(index, cumulative_net)`` may raise to prune.
        """
        votes_full = self._votes(spec)
        exit_rules = self._exit(spec)
        capital = self.research.backtest.initial_capital
        all_trades: list[dict] = []
        equities: list[np.ndarray] = []
        skipped = 0
        for k, (a, b) in enumerate(self._chunks(split)):
            res = run_backtest(self.candles.iloc[a:b], votes_full[a:b], min_votes=spec.min_votes,
                               direction=spec.direction, exit_rules=exit_rules,
                               cfg=self.research.backtest, initial_capital=capital)
            capital = res.final_capital
            all_trades.extend(res.trades)
            equities.append(res.equity)
            skipped += res.skipped_entries
            if on_chunk is not None:
                on_chunk(k, float(sum(t["net_pnl"] for t in all_trades)))
        combined = BacktestResult(trades=all_trades,
                                  equity=np.concatenate(equities) if equities else np.zeros(0),
                                  final_capital=capital, skipped_entries=skipped)
        return self._metrics(combined, self.research.backtest.initial_capital)

    def walk_forward(self, spec: StrategySpec, split: str = "validation") -> dict:
        """Evaluate the strategy on sequential, non-overlapping segments of a split.

        Each segment starts with fresh capital, so segments are independent observations.
        """
        a, b = self.slice_bounds(split)
        votes = self._votes(spec)
        edges = np.linspace(a, b, self.research.split.walk_forward_segments + 1).astype(int)
        segments = []
        for k in range(len(edges) - 1):
            sa, sb = int(edges[k]), int(edges[k + 1])
            res = run_backtest(self.candles.iloc[sa:sb], votes[sa:sb], min_votes=spec.min_votes,
                               direction=spec.direction, exit_rules=self._exit(spec),
                               cfg=self.research.backtest)
            net = float(sum(t["net_pnl"] for t in res.trades))
            segments.append({
                "segment": k + 1,
                "start": self.candles.index[sa].isoformat(),
                "end": self.candles.index[sb - 1].isoformat(),
                "trades": len(res.trades),
                "net_profit": round(net, 2),
            })
        profitable = sum(1 for s in segments if s["net_profit"] > 0)
        total_net = round(sum(s["net_profit"] for s in segments), 2)
        total_trades = sum(s["trades"] for s in segments)
        validated = (profitable / len(segments) >= 0.5 and total_net > 0
                     and total_trades >= self.research.min_trades)
        return {"split": split, "segments": segments, "profitable_segments": profitable,
                "total_segments": len(segments), "validation_net_profit": total_net,
                "total_trades": total_trades, "validated": bool(validated)}

    def _metrics(self, res: BacktestResult, initial: float) -> dict:
        m = compute_metrics(res.trades, res.equity, initial, res.final_capital)
        m.update(return_moments(res.equity))  # additive: per-bar Sharpe, skew, kurtosis for DSR
        m["skipped_entries"] = int(res.skipped_entries)
        m["eligible"] = m["trades"] >= self.research.min_trades
        return m

    # -- persistence and deduplication ---------------------------------------------------------
    def cached(self, spec: StrategySpec, split: str) -> dict | None:
        row = self.db.one(
            "SELECT metrics_json FROM experiments WHERE strategy_hash=? AND dataset_id=? AND split=?"
            " AND config_hash=? AND status='completed'",
            (spec.strategy_hash, self.dataset_id, split, self.config_hash))
        return json.loads(row["metrics_json"]) if row else None

    def record(self, spec: StrategySpec, split: str, metrics: dict, *, agent: str,
               study_name: str | None = None, trial_number: int | None = None,
               duration_s: float | None = None) -> bool:
        """Insert a completed experiment. Returns False if an identical result already exists."""
        with self.db.transaction() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO experiments(created_at, agent, study_name, trial_number,"
                " strategy_hash, dataset_id, split, config_hash, spec_json, metrics_json, eligible,"
                " status, duration_s) VALUES (?,?,?,?,?,?,?,?,?,?,?,'completed',?)",
                (utcnow(), agent, study_name, trial_number, spec.strategy_hash, self.dataset_id,
                 split, self.config_hash, self.db.dumps(spec.to_dict()), self.db.dumps(metrics),
                 int(bool(metrics.get("eligible"))), duration_s))
            return cur.rowcount == 1

    def record_failure(self, spec: StrategySpec, split: str, *, agent: str, error: str,
                       study_name: str | None = None, trial_number: int | None = None,
                       status: str = "failed") -> None:
        self.db.execute(
            "INSERT INTO experiments(created_at, agent, study_name, trial_number, strategy_hash,"
            " dataset_id, split, config_hash, spec_json, metrics_json, eligible, status, error)"
            " VALUES (?,?,?,?,?,?,?,?,?,NULL,0,?,?)",
            (utcnow(), agent, study_name, trial_number, spec.strategy_hash, self.dataset_id, split,
             self.config_hash, self.db.dumps(spec.to_dict()), status, error[:500]))

    def evaluate_cached(self, spec: StrategySpec, split: str, *, agent: str,
                        study_name: str | None = None, trial_number: int | None = None) -> dict:
        """Return stored metrics if this exact experiment exists; otherwise run and record it."""
        hit = self.cached(spec, split)
        if hit is not None:
            return {**hit, "_cached": True}
        started = time.monotonic()
        metrics = self.evaluate(spec, split)
        self.record(spec, split, metrics, agent=agent, study_name=study_name,
                    trial_number=trial_number, duration_s=round(time.monotonic() - started, 3))
        return {**metrics, "_cached": False}
