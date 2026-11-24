"""Performance metrics computed from trades and the equity curve."""
from __future__ import annotations

import numpy as np

PF_CAP = 999.0  # profit factor is reported as this value when there are no losing trades


def compute_metrics(trades: list[dict], equity: np.ndarray, initial_capital: float,
                    final_capital: float | None = None) -> dict:
    nets = np.array([t["net_pnl"] for t in trades], dtype=float)
    n = int(len(nets))
    wins = nets[nets > 0]
    losses = nets[nets < 0]
    net_profit = float(nets.sum()) if n else 0.0
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    if gross_loss > 0:
        profit_factor = min(gross_win / gross_loss, PF_CAP)
    else:
        profit_factor = PF_CAP if gross_win > 0 else 0.0

    eq = np.asarray(equity, dtype=float)
    if eq.size:
        peak = np.maximum.accumulate(eq)
        drawdown = peak - eq
        max_dd = float(drawdown.max())
        with np.errstate(divide="ignore", invalid="ignore"):
            dd_pct = np.where(peak > 0, drawdown / peak, 0.0)
        max_dd_pct = float(dd_pct.max() * 100)
    else:
        max_dd, max_dd_pct = 0.0, 0.0

    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t["reason"]] = reasons.get(t["reason"], 0) + 1

    return {
        "trades": n,
        "net_profit": round(net_profit, 2),
        "return_pct": round(net_profit / initial_capital * 100, 4) if initial_capital else 0.0,
        "final_capital": round(float(final_capital if final_capital is not None
                                     else initial_capital + net_profit), 2),
        "win_rate": round(len(wins) / n * 100, 2) if n else 0.0,
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct, 4),
        "recovery_factor": round(net_profit / max_dd, 4) if max_dd > 0 else 0.0,
        "avg_trade": round(net_profit / n, 2) if n else 0.0,
        "avg_win": round(float(wins.mean()), 2) if wins.size else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if losses.size else 0.0,
        "total_costs": round(float(sum(t["costs"] for t in trades)), 2),
        "longs": sum(1 for t in trades if t["side"] == "LONG"),
        "shorts": sum(1 for t in trades if t["side"] == "SHORT"),
        "exit_reasons": reasons,
    }
