"""Statistical robustness tests for research results. Pure functions; the background runner lives in
robust_service.py.

* Hansen SPA (2005, consistent version): does any candidate beat the benchmark out of sample of its
  own selection? Uses a stationary block bootstrap with the consistent recentering of Hansen (2005).
* Permutation test: how often does a randomly time-shifted version of the same signals (same
  frequency, different timing) make at least as much money?
* Monte Carlo: resample the observed trades with replacement to get a distribution of net profit,
  maximum drawdown and probability of loss.

All functions take numpy arrays and return plain dicts. Nothing here uses synthetic data unless the
caller passes it; the runner records which dataset each result came from.
"""
from __future__ import annotations

import math

import numpy as np


def bar_returns(equity) -> np.ndarray:
    """Simple per-bar returns of an equity curve (mark-to-market)."""
    eq = np.asarray(equity, dtype=float)
    if eq.size < 3:
        return np.zeros(0)
    prev = eq[:-1]
    ok = prev > 0
    out = np.zeros(prev.size)
    out[ok] = np.diff(eq)[ok] / prev[ok]
    return out


def stationary_bootstrap_indices(n: int, mean_block: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """B index sequences of length n from the stationary bootstrap (Politis & Romano, 1994)."""
    p = 1.0 / max(1, mean_block)
    idx = np.empty((B, n), dtype=np.int64)
    for b in range(B):
        cur = rng.integers(0, n)
        starts = rng.random(n) < p
        draws = rng.integers(0, n, size=n)
        for t in range(n):
            if starts[t]:
                cur = draws[t]
            else:
                cur = (cur + 1) % n
            idx[b, t] = cur
    return idx


def hansen_spa(strategy_returns: np.ndarray, benchmark: np.ndarray, B: int = 500, seed: int = 0) -> dict:
    """Hansen's consistent SPA test that no candidate beats the benchmark.

    ``strategy_returns`` is (T x K), one column per candidate. The performance differential is
    d_k,t = r_k,t - r_benchmark,t. H0: max_k E[d_k] <= 0. Small p-values reject H0.
    """
    R = np.asarray(strategy_returns, dtype=float)
    if R.ndim == 1:
        R = R[:, None]
    bench = np.asarray(benchmark, dtype=float)
    if R.shape[0] != bench.shape[0]:
        raise ValueError("strategy and benchmark returns must cover the same bars")
    n, K = R.shape
    if n < 50:
        raise ValueError("at least 50 bars are needed for the SPA bootstrap")
    if K < 1:
        raise ValueError("at least one candidate is needed")
    D = R - bench[:, None]
    dbar = D.mean(axis=0)
    omega = D.std(axis=0, ddof=1)
    omega = np.where(omega > 0, omega, np.nan)
    t_k = math.sqrt(n) * dbar / omega
    stat = float(np.nanmax(np.maximum(t_k, -np.inf)))
    stat = max(stat, 0.0) if np.isfinite(stat) else 0.0
    # consistent recentering: candidates clearly worse than the benchmark are excluded from the bootstrap
    threshold = -math.sqrt(2 * math.log(math.log(max(n, 16))))
    mu = np.where(t_k <= threshold, dbar, 0.0)
    mean_block = max(1, round(n ** (1.0 / 3.0)))
    rng = np.random.default_rng(seed)
    idx = stationary_bootstrap_indices(n, mean_block, B, rng)
    # bootstrap the demeaned differentials, then centre each candidate at its recentering value mu:
    # 0 for candidates that could be good, their own mean for clearly bad ones
    Dc = D - dbar
    boot = np.empty(B)
    for b in range(B):
        dstar = Dc[idx[b]].mean(axis=0) + mu
        z = math.sqrt(n) * dstar / omega
        boot[b] = np.nanmax(z)
    boot = np.where(np.isfinite(boot), boot, 0.0)
    p = float((1 + np.sum(boot >= stat)) / (B + 1))
    return {"statistic": round(stat, 6), "p_value": round(p, 6), "bootstrap_draws": int(B),
                "mean_block_bars": int(mean_block), "candidates": int(K), "bars": int(n),
                "best_candidate": int(np.nanargmax(t_k)) if np.isfinite(t_k).any() else None,
                "best_mean_excess_per_bar": float(dbar[int(np.nanargmax(t_k))]) if np.isfinite(t_k).any() else None}


def permutation_p_value(observed: float, null: np.ndarray) -> dict:
    """One-sided p-value with the standard +1 correction (the observed value is one draw)."""
    null = np.asarray(null, dtype=float)
    if null.size == 0:
        raise ValueError("the null distribution is empty")
    p = (1 + int(np.sum(null >= observed))) / (null.size + 1)
    return {"observed": round(float(observed), 2), "p_value": round(float(p), 6), "draws": int(null.size),
            "null_mean": round(float(null.mean()), 2), "null_p95": round(float(np.percentile(null, 95)), 2)}


def circular_shift(votes: np.ndarray, shift: int) -> np.ndarray:
    """Shift a signal series in time, keeping its frequency and its internal structure."""
    return np.roll(np.asarray(votes), shift)


def monte_carlo(trade_pnl: np.ndarray, capital: float, iterations: int = 5000, seed: int = 0) -> dict:
    """Bootstrap the observed trades (with replacement) to see how much the result depends on order and luck."""
    pnl = np.asarray(trade_pnl, dtype=float)
    n = pnl.size
    if n < 5:
        raise ValueError("at least 5 trades are needed for a Monte Carlo test")
    rng = np.random.default_rng(seed)
    nets = np.empty(iterations)
    dds = np.empty(iterations)
    chunk = max(1, 2_000_000 // n)
    done = 0
    while done < iterations:
        m = min(chunk, iterations - done)
        sample = pnl[rng.integers(0, n, size=(m, n))]
        cum = np.cumsum(sample, axis=1)
        peak = np.maximum.accumulate(np.concatenate([np.zeros((m, 1)), cum], axis=1), axis=1)[:, 1:]
        dd = (peak - cum).max(axis=1)
        nets[done:done + m] = cum[:, -1]
        dds[done:done + m] = dd
        done += m
    pct = lambda a, q: round(float(np.percentile(a, q)), 2)  # noqa: E731
    return {"iterations": int(iterations), "trades": int(n),
            "net_profit": {"p5": pct(nets, 5), "p50": pct(nets, 50), "p95": pct(nets, 95)},
            "max_drawdown": {"p5": pct(dds, 5), "p50": pct(dds, 50), "p95": pct(dds, 95)},
            "max_drawdown_pct_of_capital_p95": round(float(np.percentile(dds, 95)) / capital * 100.0, 4),
            "probability_of_loss": round(float(np.mean(nets < 0)), 4),
            "observed_net_profit": round(float(pnl.sum()), 2)}
