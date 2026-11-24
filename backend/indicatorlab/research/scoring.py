"""Custom conservative selection score: Beta-Binomial shrinkage and Wilson lower bound.

Definitions used throughout the research pipeline (see README, "Scoring"):

    p_hat = (wins + alpha) / (trades + alpha + beta)             Beta-Binomial posterior mean
    n     = trades + alpha + beta
    LB    = Wilson lower bound of p_hat with n and z             conservative win-rate estimate

Defaults alpha = beta = 3 and z = 1.28 (one-sided ~90%). A result with fewer than
``MIN_TRADES`` trades scores 0 for both measures, so tiny samples cannot rank high.
"""
from __future__ import annotations

import math

ALPHA = 3.0
BETA = 3.0
Z = 1.28
MIN_TRADES = 15


def beta_binomial_mean(wins: int, trades: int, alpha: float = ALPHA, beta: float = BETA) -> float:
    if trades < 0 or wins < 0 or wins > trades:
        raise ValueError("require 0 <= wins <= trades")
    return (wins + alpha) / (trades + alpha + beta)


def wilson_lower_bound(p: float, n: float, z: float = Z) -> float:
    """Lower bound of the Wilson score interval for proportion p with effective sample size n."""
    if n <= 0:
        return 0.0
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    z2 = z * z
    centre = p + z2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (centre - margin) / (1 + z2 / n)


def shrunk_scores(wins: int, trades: int, *, min_trades: int = MIN_TRADES,
                  alpha: float = ALPHA, beta: float = BETA, z: float = Z) -> dict:
    """Shrunk score and Wilson lower bound. Both are 0 below ``min_trades``."""
    if trades < min_trades:
        return {"shrunk_score": 0.0, "wilson_lb": 0.0, "eligible_for_score": False}
    p = beta_binomial_mean(wins, trades, alpha, beta)
    n = trades + alpha + beta
    return {"shrunk_score": p, "wilson_lb": wilson_lower_bound(p, n, z), "eligible_for_score": True}
