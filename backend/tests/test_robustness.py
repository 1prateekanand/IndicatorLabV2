"""Statistical robustness: SPA, permutation p-values, Monte Carlo and the circular-shift null."""
import numpy as np
import pytest

from indicatorlab.research import robustness as rb


def test_spa_on_pure_noise_does_not_reject():
    rng = np.random.default_rng(1)
    T, K = 800, 12
    bench = rng.normal(0, 0.001, T)
    cands = bench[:, None] + rng.normal(0, 0.001, (T, K))  # no candidate has an edge over the benchmark
    res = rb.hansen_spa(cands, bench, B=400, seed=2)
    assert res["p_value"] > 0.05


def test_spa_detects_a_real_edge():
    rng = np.random.default_rng(3)
    T = 800
    bench = rng.normal(0, 0.001, T)
    edge = bench + 0.0015 + rng.normal(0, 0.0005, T)
    noise = rng.normal(0, 0.001, (T, 5))
    res = rb.hansen_spa(np.column_stack([noise, edge]), bench, B=400, seed=4)
    assert res["p_value"] < 0.05
    assert res["best_candidate"] == 5


def test_spa_input_checks():
    with pytest.raises(ValueError):
        rb.hansen_spa(np.zeros((100, 2)), np.zeros(99))
    with pytest.raises(ValueError):
        rb.hansen_spa(np.zeros((10, 2)), np.zeros(10))


def test_permutation_p_value_counts_draws_at_least_as_good():
    res = rb.permutation_p_value(10.0, np.array([1.0, 12.0, 15.0, 3.0]))
    assert res["p_value"] == pytest.approx(3 / 5)
    with pytest.raises(ValueError):
        rb.permutation_p_value(1.0, np.array([]))


def test_circular_shift_preserves_signal_counts():
    v = np.array([1, 0, 0, -1, 0, 1, 0])
    for s in range(len(v)):
        shifted = rb.circular_shift(v, s)
        assert sorted(shifted.tolist()) == sorted(v.tolist())


def test_monte_carlo_profitable_and_losing_books():
    winners = np.full(60, 100.0) + np.random.default_rng(5).normal(0, 5, 60)
    res = rb.monte_carlo(winners, capital=1_000_000, iterations=800, seed=6)
    assert res["probability_of_loss"] == 0.0
    assert res["net_profit"]["p5"] > 0
    losers = -np.abs(np.random.default_rng(7).normal(100, 5, 60))
    res2 = rb.monte_carlo(losers, capital=1_000_000, iterations=800, seed=8)
    assert res2["probability_of_loss"] == 1.0
    assert res2["max_drawdown"]["p50"] > 0
    with pytest.raises(ValueError):
        rb.monte_carlo(np.array([1.0, -1.0]), capital=1.0)


def test_bar_returns_from_equity():
    eq = np.array([100.0, 110.0, 99.0, 0.0, 5.0])
    r = rb.bar_returns(eq)
    assert r[0] == pytest.approx(0.1)
    assert r[1] == pytest.approx(-0.1)
    assert np.isfinite(r).all()
