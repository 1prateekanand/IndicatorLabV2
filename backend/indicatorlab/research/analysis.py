"""Optional analysis on finished results: Pareto optimisation and indicator clustering.

Neither module is required by the core pipeline. They read stored metrics and correlation matrices
and never run backtests themselves.
"""
from __future__ import annotations

import numpy as np

OBJECTIVES = (("net_return_pct", "max"), ("raw_win_rate", "max"), ("max_drawdown_pct", "min"))


def pareto_front(points: list[dict], objectives=OBJECTIVES) -> dict:
    """Non-dominated set for several objectives. Each point needs the objective fields; missing or
    null values are treated as worst. Returns the front (rank 1) and every point's front index."""
    def value(p: dict, key: str, sense: str) -> float:
        v = p.get(key)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return -np.inf
        return float(v) if sense == "max" else -float(v)  # internally always maximise

    vectors = [tuple(value(p, k, s) for k, s in objectives) for p in points]
    n = len(points)
    dominated_by = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vi, vj = vectors[i], vectors[j]
            if all(b >= a for a, b in zip(vi, vj)) and any(b > a for a, b in zip(vi, vj)):
                dominated_by[i] += 1
    # successive fronts (rank 1 = non-dominated)
    ranks = [0] * n
    remaining = set(range(n))
    rank = 1
    while remaining:
        front = [i for i in remaining if not any(
            all(b >= a for a, b in zip(vectors[i], vectors[j])) and any(b > a for a, b in zip(vectors[i], vectors[j]))
            for j in remaining if j != i)]
        if not front:  # pragma: no cover - a finite set always has a non-dominated point
            front = list(remaining)
        for i in front:
            ranks[i] = rank
            remaining.discard(i)
        rank += 1
    front_points = [{**points[i], "pareto_rank": ranks[i]} for i in range(n) if ranks[i] == 1]
    return {"objectives": [{"key": k, "sense": s} for k, s in objectives], "front": front_points,
            "ranks": ranks, "points": n}


def cluster_indicators(names: list[str], corr: list[list], threshold: float = 0.8) -> dict:
    """Average-linkage clusters on distance 1 - |rho|. Indicators whose |rho| is at or above
    ``threshold`` (directly or through a chain of such pairs) share a cluster."""
    n = len(names)
    if n == 0:
        return {"clusters": [], "threshold": threshold}
    M = np.array([[1.0 if v is None and i == j else (np.nan if v is None else float(v))
                   for j, v in enumerate(row)] for i, row in enumerate(corr)], dtype=float)
    D = 1.0 - np.abs(np.nan_to_num(M, nan=0.0))
    np.fill_diagonal(D, 0.0)
    clusters = [[i] for i in range(n)]
    cut = 1.0 - threshold

    def link(a: list[int], b: list[int]) -> float:
        return float(np.mean([D[i, j] for i in a for j in b]))

    while len(clusters) > 1:
        best = None
        for x in range(len(clusters)):
            for y in range(x + 1, len(clusters)):
                d = link(clusters[x], clusters[y])
                if best is None or d < best[0]:
                    best = (d, x, y)
        if best is None or best[0] > cut:
            break
        _, x, y = best
        clusters[x] = clusters[x] + clusters[y]
        del clusters[y]
    out = []
    for cid, members in enumerate(sorted(clusters, key=lambda c: min(c)), start=1):
        out.append({"cluster": cid, "members": [names[i] for i in sorted(members)],
                    "size": len(members),
                    "max_abs_rho_inside": round(float(max([abs(M[i, j]) for i in members for j in members if i != j],
                                                          default=1.0)), 4)})
    return {"clusters": out, "threshold": threshold, "indicators": n}
