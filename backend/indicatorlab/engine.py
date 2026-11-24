"""Bar-by-bar backtest engine with explicit execution rules.

Execution rules (these prevent look-ahead bias):
  1. Signals are read at the CLOSE of bar t.
  2. Orders created at bar t's close execute at the OPEN of bar t+1, plus adverse slippage.
  3. Stop-loss and target are checked against the high/low of bars AFTER the entry bar.
     If a single bar touches both, the stop-loss is assumed to fill first (conservative).
  4. A gap through the stop fills at the bar's open, not at the stop price.
  5. Any position still open is closed at the square-off minute or at the last bar of the slice.

Accounting: cash is reduced by entry costs at entry and realised P&L plus exit costs at exit.
Equity = cash + mark-to-market of any open position at the bar close.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import BacktestConfig


@dataclass(frozen=True)
class Exit:
    stop_loss_pct: float = 0.0  # 0 disables
    target_pct: float = 0.0  # 0 disables


@dataclass
class BacktestResult:
    trades: list[dict] = field(default_factory=list)
    equity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    timestamps: list = field(default_factory=list)
    final_capital: float = 0.0
    skipped_entries: int = 0  # signals that could not be sized to at least one lot


def run_backtest(df: pd.DataFrame, votes: np.ndarray, *, min_votes: int, direction: str,
                 exit_rules: Exit, cfg: BacktestConfig, initial_capital: float | None = None,
                 exit_signals: np.ndarray | None = None, max_hold_bars: int | None = None) -> BacktestResult:
    """Simulate one slice. ``votes`` is the integer sum of indicator signals per bar.

    ``exit_signals`` (optional, used by custom Strategy Studio code): a bar value equal to the open
    position's side (or its opposite) closes the position at the next bar, reason ``strategy_exit``.
    This mirrors the rule used by the Express engine so that Studio validation results match.

    ``max_hold_bars`` (optional): close a position once it has been held this many bars
    (exit reason ``holding_period``). ``None`` disables the rule.
    """
    if direction not in ("both", "long", "short"):
        raise ValueError("direction must be 'both', 'long' or 'short'")
    n = len(df)
    if n == 0:
        return BacktestResult(final_capital=initial_capital or cfg.initial_capital)
    opens = df["open"].to_numpy(float)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    minutes = (df.index.hour * 60 + df.index.minute).to_numpy()
    votes = np.asarray(votes, dtype=int)
    costs = cfg.costs
    slip = costs.slippage_points
    # Exit percentages are stored as percent values (0.5 means 0.5%); convert to fractions.
    sl_pct = exit_rules.stop_loss_pct / 100.0
    tgt_pct = exit_rules.target_pct / 100.0

    cash = float(initial_capital if initial_capital is not None else cfg.initial_capital)
    equity = np.empty(n)
    trades: list[dict] = []
    pos = 0  # +1 long, -1 short, 0 flat
    qty = 0
    entry_px = entry_cost = 0.0
    entry_i = -1
    entry_time = None
    pending: tuple[str, int, str] | None = None  # (kind, direction, reason)
    skipped = 0

    def fill_cost(px: float, q: int) -> float:
        return costs.brokerage_per_order + costs.charges_pct * px * q

    def close_position(i: int, px: float, reason: str) -> None:
        nonlocal cash, pos, qty
        px = float(px)
        pnl = float((px - entry_px) * qty * pos)
        exit_cost = float(fill_cost(px, qty))
        cash += pnl - exit_cost
        trades.append({
            "side": "LONG" if pos > 0 else "SHORT",
            "entry_time": entry_time.isoformat(), "exit_time": df.index[i].isoformat(),
            "entry_price": round(float(entry_px), 4), "exit_price": round(px, 4), "quantity": int(qty),
            "gross_pnl": round(pnl, 4), "costs": round(entry_cost + exit_cost, 4),
            "net_pnl": round(pnl - entry_cost - exit_cost, 4), "reason": reason,
            "bars_held": int(i - entry_i),
        })
        pos, qty = 0, 0

    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        t_min = minutes[i]

        # 1) Execute an order decided at the previous close, at this bar's open.
        if pending is not None:
            kind, side, reason = pending
            pending = None
            if kind == "enter" and pos == 0 and t_min < cfg.no_entry_after_minute:
                px = o + side * slip
                q = int(np.floor((cash * cfg.position_pct) / (px * cfg.lot_size))) * cfg.lot_size
                if q >= cfg.lot_size:
                    pos, qty = side, q
                    entry_px = px
                    entry_cost = fill_cost(px, q)
                    cash -= entry_cost
                    entry_i = i
                    entry_time = df.index[i]
                else:
                    skipped += 1
            elif kind == "exit" and pos != 0:
                close_position(i, o - pos * slip, reason)

        # 2) Stop-loss and target, checked intrabar on the open position.
        if pos != 0 and i > entry_i:
            sl_px = entry_px * (1 - pos * sl_pct) if sl_pct > 0 else None
            tg_px = entry_px * (1 + pos * tgt_pct) if tgt_pct > 0 else None
            if pos > 0:
                hit_sl = sl_px is not None and l <= sl_px
                hit_tg = tg_px is not None and h >= tg_px
            else:
                hit_sl = sl_px is not None and h >= sl_px
                hit_tg = tg_px is not None and l <= tg_px
            if hit_sl:
                fill = o if (pos > 0 and o <= sl_px) or (pos < 0 and o >= sl_px) else sl_px
                close_position(i, fill - pos * slip, "stop_loss")
            elif hit_tg:
                fill = o if (pos > 0 and o >= tg_px) or (pos < 0 and o <= tg_px) else tg_px
                close_position(i, fill - pos * slip, "target")

        # 3) Decide at this bar's close (orders execute next bar).
        if pos != 0 and t_min >= cfg.square_off_minute:
            close_position(i, c - pos * slip, "square_off")
        elif pos != 0:
            custom_exit = exit_signals is not None and int(exit_signals[i]) != 0 and int(exit_signals[i]) in (pos, -pos)
            reversal = votes[i] * pos <= -min_votes
            held_out = max_hold_bars is not None and (i - entry_i + 1) >= max_hold_bars
            if i + 1 < n and (custom_exit or reversal or held_out):
                reason = "strategy_exit" if custom_exit else ("signal_reversal" if reversal else "holding_period")
                pending = ("exit", -pos, reason)
        if pos == 0 and pending is None and t_min < cfg.square_off_minute and i + 1 < n:
            if votes[i] >= min_votes and direction in ("both", "long"):
                pending = ("enter", 1, "signal_long")
            elif votes[i] <= -min_votes and direction in ("both", "short"):
                pending = ("enter", -1, "signal_short")

        # 4) Mark to market.
        equity[i] = cash + ((c - entry_px) * qty * pos if pos != 0 else 0.0)

    # 5) Flatten at the end of the slice.
    if pos != 0:
        close_position(n - 1, closes[-1] - pos * slip, "end_of_data")
        equity[-1] = cash

    return BacktestResult(trades=trades, equity=equity, timestamps=list(df.index),
                          final_capital=float(cash), skipped_entries=skipped)
