"""Runtime configuration. Every path can be overridden with an environment variable."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("IL_DATA_DIR", PROJECT_ROOT / "data"))
DB_PATH = Path(os.getenv("IL_DB_PATH", DATA_DIR / "indicator_lab.db"))
OPTUNA_STORAGE = os.getenv("IL_OPTUNA_STORAGE", f"sqlite:///{DATA_DIR / 'optuna.db'}")
PLUGIN_DIR = Path(os.getenv("IL_INDICATOR_PLUGIN_DIR", PROJECT_ROOT / "indicator_plugins"))
CPU_LIMIT_PCT = float(os.getenv("IL_CPU_LIMIT_PCT", 90))
RAM_LIMIT_PCT = float(os.getenv("IL_RAM_LIMIT_PCT", 85))


@dataclass(frozen=True)
class CostModel:
    """Trading costs. Defaults are placeholders: set them to your broker's real schedule."""

    brokerage_per_order: float = 20.0  # flat currency amount per executed order
    charges_pct: float = 0.0005  # fraction of order turnover (taxes, exchange and other charges)
    slippage_points: float = 0.5  # adverse price adjustment applied to every fill


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000_000.0  # placeholder; set your own virtual capital
    position_pct: float = 0.10  # fraction of equity committed per trade
    lot_size: int = 1
    costs: CostModel = field(default_factory=CostModel)
    no_entry_after_minute: int = 15 * 60  # minutes after midnight; no new entries at/after this
    square_off_minute: int = 15 * 60 + 15  # flat everything at this minute of day
    max_points: float = 1e12  # reserved guard; not used for sizing


@dataclass(frozen=True)
class SplitConfig:
    """Chronological split of the dataset. The out-of-sample tail is never used by agents."""

    train_end: float = 0.60
    validation_end: float = 0.80
    train_chunks: int = 3  # sequential chunks evaluated for pruning and reporting
    walk_forward_segments: int = 4


DEFAULT_BACKTEST = BacktestConfig()
DEFAULT_SPLIT = SplitConfig()
MIN_TRADES = int(os.getenv("IL_MIN_TRADES", 30))
