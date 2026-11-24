"""Overfitting and significance statistics. Pure NumPy plus ``statistics.NormalDist``.

Implemented here
----------------
* ``return_moments``        per-bar return observations, Sharpe ratio, skewness, raw kurtosis.
* ``expected_max_sharpe``   E[max SR] over N independent trials (Bailey & López de Prado, 2014).
* ``probabilistic_sharpe``  PSR(SR*): probability the true Sharpe exceeds a benchmark.
* ``deflated_sharpe``       DSR: PSR evaluated at the expected-maximum benchmark across trials.
* ``pbo_cscv``              Probability of Backtest Overfitting via Combinatorially Symmetric
                            Cross-Validation (Bailey, Borwein, López de Prado, Zhu, 2016).

Every function returns a dict with ``status`` set to ``"ok"`` or ``"insufficient_sample"``
(with a reason) instead of raising for small samples, so callers can store the outcome honestly.

Sharpe ratios here are per observation (per bar), not annualised. DSR and PBO depend on
ratios of Sharpe values, so the annualisation factor cancels out only for the trial-variance
term; both are therefore computed on the same per-bar scale for every trial.
"""
from __future__ import annotations

import itertools
import math
from statistics import NormalDist

import numpy as np

EULER_GAMMA = 0.5772156649015329
_N01 = NormalDist()

MIN_OBS_DSR = 30          # shorter return series are not evaluated for DSR
MIN_TRIALS_DSR = 2        # the trial variance needs at least two trials
MIN_BLOCK_ROWS_PBO = 4    # rows per CSCV block


def return_moments(equity: np.ndarray) -> dict:
    """Per-bar simple returns derived from a mark-to-market equity curve."""
    equity = np.asarray(equity, dtype=float)
    if equity.size < 3:
        return {"returns_obs": 0, "sharpe_per_bar": None, "return_skew": None, "return_kurtosis": None}
    prev = equity[:-1]
    ok = prev > 0
    r = np.diff(equity)[ok] / prev[ok]
    return {"returns_obs": int(r.size), **sharpe_moments(r)}


def sharpe_moments(r: np.ndarray) -> dict:
    r = np.asarray(r, dtype=float)
    if r.size < 2:
        return {"sharpe_per_bar": None, "return_skew": None, "return_kurtosis": None}
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    if sd <= 0 or not math.isfinite(sd):
        return {"sharpe_per_bar": None, "return_skew": None, "return_kurtosis": None}
    z = (r - mu) / float(r.std(ddof=0))
    pop_sd = float(r.std(ddof=0))
    if pop_sd <= 0:
        return {"sharpe_per_bar": mu / sd, "return_skew": 0.0, "return_kurtosis": 3.0}
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))  # raw kurtosis: 3 for a normal distribution
    return {"sharpe_per_bar": mu / sd, "return_skew": skew, "return_kurtosis": kurt}


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """Expected maximum of N Sharpe ratios whose true values are iid with variance ``var_sharpe``.

    E[max] ~= sqrt(V) * ((1-g) * Phi^-1(1 - 1/N) + g * Phi^-1(1 - 1/(N e)))
    """
    if n_trials < 2:
        return 0.0
    if var_sharpe <= 0:
        return 0.0
    a = _N01.inv_cdf(1.0 - 1.0 / n_trials)
    b = _N01.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(var_sharpe) * ((1.0 - EULER_GAMMA) * a + EULER_GAMMA * b)


def probabilistic_sharpe(sr: float, benchmark: float, n_obs: int, skew: float, kurt: float) -> dict:
    """PSR(SR*) = Phi((SR - SR*) sqrt(T-1) / sqrt(1 - g3 SR + (g4 - 1)/4 SR^2))."""
    if n_obs < MIN_OBS_DSR:
        return {"status": "insufficient_sample", "reason": f"needs at least {MIN_OBS_DSR} return observations, has {n_obs}"}
    denom_sq = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    if denom_sq <= 0 or not math.isfinite(denom_sq):
        return {"status": "insufficient_sample", "reason": "non-positive variance term in the PSR denominator"}
    z = (sr - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denom_sq)
    return {"status": "ok", "psr": _N01.cdf(z), "z": z}


def deflated_sharpe(sr: float, n_obs: int, skew: float, kurt: float,
                    trial_sharpes: list[float]) -> dict:
    """DSR for one strategy against all trials that were searched.

    ``trial_sharpes`` must contain the per-bar Sharpe of every eligible trial, including this one.
    The result records every assumption so it can be stored and displayed.
    """
    clean = [float(x) for x in trial_sharpes if x is not None and math.isfinite(float(x))]
    n = len(clean)
    assumptions = {
        "method": "Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)",
        "sharpe_scale": "per bar, not annualised",
        "n_trials": n,
        "min_obs": MIN_OBS_DSR,
        "min_trials": MIN_TRIALS_DSR,
        "moments": "raw kurtosis (3 = normal); skew and kurtosis of the strategy's own bar returns",
        "trial_population": "eligible completed experiments on the same dataset, split and config",
    }
    if sr is None or not math.isfinite(sr):
        return {"status": "insufficient_sample", "reason": "no Sharpe ratio (flat or too few returns)",
                "assumptions": assumptions}
    if n < MIN_TRIALS_DSR:
        return {"status": "insufficient_sample", "reason": f"needs at least {MIN_TRIALS_DSR} eligible trials, has {n}",
                "assumptions": assumptions}
    var_sr = float(np.var(clean, ddof=1))
    sr0 = expected_max_sharpe(n, var_sr)
    psr = probabilistic_sharpe(sr, sr0, n_obs, skew, kurt)
    if psr["status"] != "ok":
        return {**psr, "assumptions": assumptions}
    return {"status": "ok", "dsr": psr["psr"], "expected_max_sharpe": sr0, "trial_variance": var_sr,
            "sharpe": sr, "n_obs": n_obs, "skew": skew, "kurtosis": kurt, "assumptions": assumptions}


def _sharpe_cols(block: np.ndarray) -> np.ndarray:
    """Per-column Sharpe of a (rows x configs) matrix; zero-variance columns get -inf."""
    mu = block.mean(axis=0)
    sd = block.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(sd > 0, mu / sd, -np.inf)
    return out


def pbo_cscv(returns: np.ndarray, blocks: int = 8, labels: list[str] | None = None) -> dict:
    """Probability of Backtest Overfitting by CSCV.

    ``returns`` is a (T x N) matrix of per-bar returns: rows are time, columns are configurations.
    Rows are cut into ``blocks`` equal, contiguous, chronological blocks (even number). For every
    way of choosing half the blocks as in-sample (IS) and the rest as out-of-sample (OOS), the
    configuration with the best IS Sharpe is selected and its OOS rank is converted to a logit.
    PBO is the fraction of combinations whose logit is <= 0 (the selected config ranks at or
    below the median out of sample).
    """
    m = np.asarray(returns, dtype=float)
    if m.ndim != 2:
        raise ValueError("returns must be a 2-D matrix (rows = time, columns = configurations)")
    T, N = m.shape
    labels = labels or [f"config_{j}" for j in range(N)]
    if blocks < 4 or blocks % 2:
        raise ValueError("blocks must be an even number of at least 4")
    base = {"blocks": blocks, "configs": N, "rows": T, "method": "CSCV (Bailey et al. 2016)"}
    if N < 2:
        return {**base, "status": "insufficient_sample", "reason": "needs at least 2 configurations"}
    if T // blocks < MIN_BLOCK_ROWS_PBO:
        return {**base, "status": "insufficient_sample",
                "reason": f"each block needs at least {MIN_BLOCK_ROWS_PBO} rows; {T} rows / {blocks} blocks is too few"}
    if not np.all(np.isfinite(m)):
        return {**base, "status": "insufficient_sample", "reason": "non-finite returns in the matrix"}

    edges = np.linspace(0, T, blocks + 1).astype(int)
    parts = [m[edges[k]:edges[k + 1]] for k in range(blocks)]
    combos = list(itertools.combinations(range(blocks), blocks // 2))
    logits, details = [], []
    wins = 0
    for combo in combos:
        is_idx = set(combo)
        is_rows = np.vstack([parts[k] for k in combo])
        oos_rows = np.vstack([parts[k] for k in range(blocks) if k not in is_idx])
        is_sr = _sharpe_cols(is_rows)
        oos_sr = _sharpe_cols(oos_rows)
        best = int(np.argmax(is_sr))
        # rank of the selected config's OOS performance among all N (1 = worst, N = best)
        rank = 1 + int(np.sum(oos_sr < oos_sr[best])) + 0.5 * (int(np.sum(oos_sr == oos_sr[best])) - 1)
        omega = rank / (N + 1)
        lam = math.log(omega / (1.0 - omega))
        logits.append(lam)
        if lam <= 0:
            wins += 1
        details.append({"is_blocks": list(combo), "selected": labels[best],
                        "is_sharpe": None if not math.isfinite(is_sr[best]) else float(is_sr[best]),
                        "oos_rank": rank, "omega": omega, "logit": lam})
    logits_arr = np.asarray(logits)
    return {**base, "status": "ok", "pbo": wins / len(combos), "combinations": len(combos),
            "logit_mean": float(logits_arr.mean()), "logit_median": float(np.median(logits_arr)),
            "logits": [round(float(x), 6) for x in logits_arr],
            "combination_detail": details[:200], "labels": labels}
