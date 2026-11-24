"""Reference indicators (textbook examples). Every value at bar t uses only bars <= t."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..registry import ParamSpec, indicator


def _state_from_masks(index: pd.Index, bull: pd.Series, bear: pd.Series) -> pd.Series:
    out = np.zeros(len(index), dtype=np.int8)
    out[bull.fillna(False).to_numpy(bool)] = 1
    out[bear.fillna(False).to_numpy(bool)] = -1
    return pd.Series(out, index=index)


@indicator("ref_sma_cross", category="Trend",
           params=(ParamSpec("fast", 9, 3, 50), ParamSpec("slow", 21, 10, 200)),
           description="Reference: fast vs slow simple moving average.")
def ref_sma_cross(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> pd.Series:
    fast, slow = int(min(fast, slow - 1)), int(slow)
    f = df["close"].rolling(fast).mean()
    s = df["close"].rolling(slow).mean()
    return _state_from_masks(df.index, f > s, f < s)


@indicator("ref_ema_cross", category="Trend",
           params=(ParamSpec("fast", 9, 3, 50), ParamSpec("slow", 21, 10, 200)),
           description="Reference: fast vs slow exponential moving average.")
def ref_ema_cross(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> pd.Series:
    fast, slow = int(min(fast, slow - 1)), int(slow)
    f = df["close"].ewm(span=fast, adjust=False).mean()
    s = df["close"].ewm(span=slow, adjust=False).mean()
    return _state_from_masks(df.index, f > s, f < s)


@indicator("ref_rsi", category="Momentum",
           params=(ParamSpec("period", 14, 5, 30), ParamSpec("low", 30, 15, 45),
                   ParamSpec("high", 70, 55, 85)),
           description="Reference: Wilder RSI. Oversold is bullish, overbought is bearish.")
def ref_rsi(df: pd.DataFrame, period: int = 14, low: int = 30, high: int = 70) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        rsi = 100 - 100 / (1 + gain / loss)
    rsi = rsi.where(loss != 0, 100.0)
    return _state_from_masks(df.index, rsi < low, rsi > high)


@indicator("ref_macd", category="Momentum",
           params=(ParamSpec("fast", 12, 5, 20), ParamSpec("slow", 26, 20, 50),
                   ParamSpec("signal", 9, 3, 15)),
           description="Reference: MACD line versus its signal line.")
def ref_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    fast, slow = int(min(fast, slow - 1)), int(slow)
    line = (df["close"].ewm(span=fast, adjust=False).mean()
            - df["close"].ewm(span=slow, adjust=False).mean())
    sig = line.ewm(span=int(signal), adjust=False).mean()
    return _state_from_masks(df.index, line > sig, line < sig)


@indicator("ref_bollinger", category="Volatility",
           params=(ParamSpec("period", 20, 10, 50), ParamSpec("k", 2.0, 1.0, 3.0, kind="float", step=0.25)),
           description="Reference: mean reversion at Bollinger band extremes.")
def ref_bollinger(df: pd.DataFrame, period: int = 20, k: float = 2.0) -> pd.Series:
    mid = df["close"].rolling(period).mean()
    sd = df["close"].rolling(period).std(ddof=0)
    return _state_from_masks(df.index, df["close"] < mid - k * sd, df["close"] > mid + k * sd)


@indicator("ref_stochastic", category="Momentum",
           params=(ParamSpec("k_period", 14, 5, 30), ParamSpec("low", 20, 10, 35),
                   ParamSpec("high", 80, 65, 90)),
           description="Reference: fast stochastic %K zones.")
def ref_stochastic(df: pd.DataFrame, k_period: int = 14, low: int = 20, high: int = 80) -> pd.Series:
    lo = df["low"].rolling(k_period).min()
    hi = df["high"].rolling(k_period).max()
    k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return _state_from_masks(df.index, k < low, k > high)


@indicator("ref_donchian", category="Breakout",
           params=(ParamSpec("period", 20, 5, 100),),
           description="Reference: breakout beyond the prior N-bar high or low.")
def ref_donchian(df: pd.DataFrame, period: int = 20) -> pd.Series:
    prior_high = df["high"].rolling(period).max().shift(1)
    prior_low = df["low"].rolling(period).min().shift(1)
    return _state_from_masks(df.index, df["close"] > prior_high, df["close"] < prior_low)
