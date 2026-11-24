"""Pareto front and indicator clustering on known inputs."""
from indicatorlab.research.analysis import cluster_indicators, pareto_front


def test_pareto_keeps_only_non_dominated_points():
    pts = [
        {"name": "a", "net_return_pct": 10, "raw_win_rate": 50, "max_drawdown_pct": 5},
        {"name": "b", "net_return_pct": 5, "raw_win_rate": 40, "max_drawdown_pct": 8},   # dominated by a
        {"name": "c", "net_return_pct": 12, "raw_win_rate": 30, "max_drawdown_pct": 20},  # trades return for risk
        {"name": "d", "net_return_pct": 1, "raw_win_rate": 90, "max_drawdown_pct": 2},   # best win rate and risk
    ]
    res = pareto_front(pts)
    front = {p["name"] for p in res["front"]}
    assert front == {"a", "c", "d"}
    assert res["ranks"][1] == 2


def test_pareto_treats_missing_values_as_worst():
    pts = [{"name": "ok", "net_return_pct": 1, "raw_win_rate": 50, "max_drawdown_pct": 1},
           {"name": "missing", "net_return_pct": None, "raw_win_rate": 50, "max_drawdown_pct": 1}]
    assert {p["name"] for p in pareto_front(pts)["front"]} == {"ok"}


def test_pareto_empty_input():
    assert pareto_front([])["front"] == []


def test_clusters_group_highly_correlated_indicators():
    names = ["a", "b", "c", "d"]
    corr = [[1.0, 0.95, 0.1, 0.0],
            [0.95, 1.0, 0.05, 0.1],
            [0.1, 0.05, 1.0, -0.9],
            [0.0, 0.1, -0.9, 1.0]]
    res = cluster_indicators(names, corr, threshold=0.8)
    groups = [sorted(c["members"]) for c in res["clusters"]]
    assert ["a", "b"] in groups and ["c", "d"] in groups
    assert len(groups) == 2


def test_clusters_keep_unrelated_indicators_apart():
    names = ["x", "y"]
    res = cluster_indicators(names, [[1.0, 0.2], [0.2, 1.0]], threshold=0.8)
    assert [c["members"] for c in res["clusters"]] == [["x"], ["y"]]
