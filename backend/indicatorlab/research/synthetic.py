"""Synthetic market scenarios for controlled testing.

Synthetic data is never presented as market data. Every dataset it creates has ``is_synthetic=1`` and
a name starting with ``SYNTH_``, and the dashboard labels every result computed on it.

Scenarios (1-minute bars, NSE session 09:15-15:29, 375 bars per day):

* trending      persistent drift in one direction
* sideways      mean reversion around a level, no drift
* volatile      volatility regimes that cluster (calm and turbulent periods)
* sudden_move   calm random walk with rare large jumps (gaps of 3-6%)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..data import import_csv
from ..db import get_db

SCENARIOS = {
    "trending": "Persistent drift, moderate volatility",
    "sideways": "Mean reversion around a level, no drift",
    "volatile": "Volatility regimes that cluster",
    "sudden_move": "Calm random walk with rare large jumps",
}
BARS_PER_DAY = 375
START_PRICE = 22000.0


def generate_path(scenario: str, n: int, seed: int) -> np.ndarray:
    """Close prices for ``n`` bars. Deterministic for a given seed."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {', '.join(SCENARIOS)}")
    rng = np.random.default_rng(seed)
    if scenario == "trending":
        drift = np.full(n, 0.00004)
        vol = np.full(n, 0.0006)
        rets = drift + vol * rng.standard_normal(n)
    elif scenario == "sideways":
        logp = np.zeros(n)
        vol = 0.0005
        for t in range(1, n):
            logp[t] = logp[t - 1] - 0.01 * logp[t - 1] + vol * rng.standard_normal()
        rets = np.diff(logp, prepend=0.0)
    elif scenario == "volatile":
        regime = np.zeros(n, dtype=int)
        for t in range(1, n):
            regime[t] = 1 - regime[t - 1] if rng.random() < 0.004 else regime[t - 1]
        vol = np.where(regime == 1, 0.0022, 0.0004)
        rets = vol * rng.standard_normal(n)
    else:  # sudden_move
        rets = 0.0003 * rng.standard_normal(n)
        jumps = rng.random(n) < 1.0 / 3000.0
        sizes = rng.uniform(0.03, 0.06, n) * rng.choice([-1.0, 1.0], n)
        rets = rets + np.where(jumps, sizes, 0.0)
    return START_PRICE * np.exp(np.cumsum(rets))


def generate_frame(scenario: str, days: int, seed: int) -> pd.DataFrame:
    if not 2 <= days <= 400:
        raise ValueError("days must be between 2 and 400")
    n = days * BARS_PER_DAY
    closes = generate_path(scenario, n, seed)
    rng = np.random.default_rng(seed + 1_000_003)
    opens = np.concatenate([[START_PRICE], closes[:-1]])
    spread = np.abs(rng.normal(0, 0.0004, n)) * closes
    highs = np.maximum(opens, closes) + spread * rng.random(n)
    lows = np.minimum(opens, closes) - spread * rng.random(n)
    volume = rng.integers(50_000, 500_000, n).astype(float)
    index = []
    for d in range(days):
        day = pd.Timestamp("2020-01-01") + pd.Timedelta(days=d)
        while day.weekday() >= 5:
            day += pd.Timedelta(days=1)
        index.extend(pd.date_range(day + pd.Timedelta(hours=9, minutes=15), periods=BARS_PER_DAY, freq="min"))
    frame = pd.DataFrame({"Timestamp": pd.DatetimeIndex(index[:n]), "Open": opens, "High": highs,
                          "Low": lows, "Close": closes, "Volume": volume})
    return frame


def create_synthetic_dataset(scenario: str, days: int = 60, seed: int = 1, db=None) -> dict:
    db = db or get_db()
    name = f"SYNTH_{scenario}_{days}d_seed{seed}"
    if db.one("SELECT id FROM datasets WHERE name=?", (name,)):
        raise ValueError(f"synthetic dataset '{name}' already exists")
    frame = generate_frame(scenario, days, seed)
    folder = Path(config.DATA_DIR) / "synthetic"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.csv"
    out = frame.copy()
    out["Timestamp"] = out["Timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    out.round({"Open": 2, "High": 2, "Low": 2, "Close": 2}).to_csv(path, index=False)
    ds = import_csv(path, name=name, symbol="SYNTH", timeframe="1min", replace=False)
    db.execute("UPDATE datasets SET is_synthetic=1 WHERE id=?", (ds["id"],))
    return {**ds, "is_synthetic": 1, "scenario": scenario, "description": SCENARIOS[scenario], "seed": seed}


def scenario_list() -> list[dict]:
    return [{"scenario": k, "description": v} for k, v in SCENARIOS.items()]
