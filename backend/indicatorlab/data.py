"""Candle data: CSV import into SQLite, loading, and a SYNTHETIC generator for testing.

Accepted CSV layouts (column names are case-insensitive):
  * one timestamp column named ``timestamp``, ``datetime``, ``date_time`` or ``time``
    (ISO-8601 or any format pandas can parse), or
  * separate ``date`` and ``time`` columns, plus ``open``, ``high``, ``low``, ``close``
    and optionally ``volume``.

Timestamps are stored as naive wall-clock times in the exchange's local timezone (IST for NSE/BSE).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .db import get_db, utcnow

REQUIRED = ("open", "high", "low", "close")
_TS_ALIASES = ("timestamp", "datetime", "date_time", "time")


def read_candle_csv(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    if "date" in raw.columns and "time" in raw.columns and "timestamp" not in raw.columns:
        ts = pd.to_datetime(raw["date"].astype(str).str.strip() + " " + raw["time"].astype(str).str.strip(),
                            errors="raise")
    else:
        ts_col = next((c for c in _TS_ALIASES if c in raw.columns), None)
        if ts_col is None:
            raise ValueError("CSV needs a timestamp column (timestamp/datetime/date_time/time) "
                             "or separate date and time columns")
        ts = pd.to_datetime(raw[ts_col], errors="raise")
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    df = pd.DataFrame({
        "open": pd.to_numeric(raw["open"], errors="coerce"),
        "high": pd.to_numeric(raw["high"], errors="coerce"),
        "low": pd.to_numeric(raw["low"], errors="coerce"),
        "close": pd.to_numeric(raw["close"], errors="coerce"),
        "volume": pd.to_numeric(raw["volume"], errors="coerce") if "volume" in raw.columns else 0.0,
    })
    df.index = pd.DatetimeIndex(ts, name="ts")
    return clean_candles(df)


def clean_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise. Rejects corrupt data instead of silently repairing prices."""
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    if df[list(REQUIRED)].isna().any().any():
        bad = int(df[list(REQUIRED)].isna().any(axis=1).sum())
        raise ValueError(f"{bad} rows contain missing or non-numeric OHLC values")
    if (df[list(REQUIRED)] <= 0).any().any():
        raise ValueError("prices must be positive")
    if (df["high"] < df[["open", "close", "low"]].max(axis=1)).any() or \
       (df["low"] > df[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("inconsistent OHLC: high/low do not bound open/close in some rows")
    df["volume"] = df["volume"].fillna(0.0)
    if len(df) < 500:
        raise ValueError("at least 500 candles are required for a meaningful backtest")
    return df


def import_csv(path: str | Path, *, name: str, symbol: str, timeframe: str = "1min",
               replace: bool = False) -> dict:
    df = read_candle_csv(path)
    db = get_db()
    existing = db.one("SELECT id FROM datasets WHERE name = ?", (name,))
    if existing and not replace:
        raise ValueError(f"dataset '{name}' already exists; pass replace=true to overwrite")
    with db.transaction() as cur:
        if existing:
            cur.execute("DELETE FROM datasets WHERE id = ?", (existing["id"],))
        cur.execute(
            "INSERT INTO datasets(name, symbol, timeframe, source_file, rows, start_ts, end_ts, imported_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (name, symbol, timeframe, str(path), len(df), df.index[0].isoformat(),
             df.index[-1].isoformat(), utcnow()))
        dataset_id = cur.lastrowid
        cur.executemany(
            "INSERT INTO candles(dataset_id, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
            [(dataset_id, ts.isoformat(), float(o), float(h), float(l), float(c), float(v))
             for ts, o, h, l, c, v in zip(df.index, df["open"], df["high"], df["low"], df["close"],
                                          df["volume"])])
    load_candles.cache_clear()
    return get_dataset(dataset_id)


def list_datasets() -> list[dict]:
    return get_db().query("SELECT * FROM datasets ORDER BY id")


def get_dataset(dataset_id: int) -> dict:
    row = get_db().one("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
    if row is None:
        raise KeyError(f"dataset {dataset_id} not found")
    return row


@lru_cache(maxsize=8)
def load_candles(dataset_id: int) -> pd.DataFrame:
    rows = get_db().query(
        "SELECT ts, open, high, low, close, volume FROM candles WHERE dataset_id = ? ORDER BY ts",
        (dataset_id,))
    if not rows:
        raise KeyError(f"dataset {dataset_id} has no candles")
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("ts")), name="ts")
    return df


def generate_synthetic(path: str | Path, *, days: int = 60, seed: int = 42,
                       start: str = "2025-01-01", base_price: float = 22000.0) -> Path:
    """Write SYNTHETIC 1-minute NSE-style sessions (09:15-15:29 IST, weekdays) for testing only.

    The output has no real market meaning. It exists so the pipeline can be exercised end to end
    without any market data. Replace it with real historical CSV files for research.
    """
    rng = np.random.default_rng(seed)
    bars = []
    day = datetime.fromisoformat(start)
    trading_days = 0
    price = base_price
    vol = 0.0006
    while trading_days < days:
        if day.weekday() < 5:
            stamp = datetime.combine(day.date(), time(9, 15))
            for _ in range(375):  # 09:15 .. 15:29 inclusive
                vol = max(0.0002, 0.97 * vol + 0.03 * 0.0006 + 0.0002 * abs(rng.normal()))
                ret = rng.normal(0.0, vol)
                open_ = price
                close = price * (1 + ret)
                high = max(open_, close) * (1 + abs(rng.normal(0, vol / 2)))
                low = min(open_, close) * (1 - abs(rng.normal(0, vol / 2)))
                bars.append((stamp, open_, high, low, close, float(rng.integers(100, 5000))))
                price = close
                stamp += timedelta(minutes=1)
            trading_days += 1
        day += timedelta(days=1)
    df = pd.DataFrame(bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, float_format="%.2f")
    return out
