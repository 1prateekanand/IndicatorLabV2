"""Synthetic scenarios: each has its intended character, is deterministic, and is labelled as synthetic."""
import numpy as np
import pytest

from indicatorlab.db import get_db
from indicatorlab.research import synthetic as syn


def test_scenarios_have_their_intended_character():
    n = 40_000
    trend = np.diff(np.log(syn.generate_path("trending", n, 1)))
    side = syn.generate_path("sideways", n, 1)
    vol = np.diff(np.log(syn.generate_path("volatile", n, 1)))
    jumps = np.diff(np.log(syn.generate_path("sudden_move", n, 1)))
    assert trend.mean() > 0.00002, "trending should drift up"
    assert abs(np.log(side[-1] / side[0])) < 0.5 and side.std() / side.mean() < 0.05, "sideways should stay bounded"
    assert vol.std() > 0.0008, "volatile should have high realised volatility"
    assert np.abs(jumps).max() > 0.025, "sudden_move should contain large gaps"


def test_generation_is_deterministic_for_a_seed():
    a = syn.generate_path("volatile", 2000, 7)
    b = syn.generate_path("volatile", 2000, 7)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, syn.generate_path("volatile", 2000, 8))


def test_frame_is_valid_nse_session_ohlc():
    f = syn.generate_frame("trending", 3, 5)
    assert len(f) == 3 * syn.BARS_PER_DAY
    assert (f["High"] >= f[["Open", "Close"]].max(axis=1) - 1e-9).all()
    assert (f["Low"] <= f[["Open", "Close"]].min(axis=1) + 1e-9).all()
    assert f["Timestamp"].dt.hour.min() == 9 and f["Timestamp"].dt.day_of_week.max() < 5


def test_unknown_scenario_and_bad_length_are_refused():
    with pytest.raises(ValueError):
        syn.generate_path("bull_market_of_my_dreams", 100, 1)
    with pytest.raises(ValueError):
        syn.generate_frame("trending", 1, 1)


def test_synthetic_dataset_is_labelled_and_research_runs_on_it():
    db = get_db()
    ds = syn.create_synthetic_dataset("sideways", days=12, seed=99, db=db)
    row = db.one("SELECT name, symbol, is_synthetic FROM datasets WHERE id=?", (ds["id"],))
    assert row["is_synthetic"] == 1 and row["name"].startswith("SYNTH_") and row["symbol"] == "SYNTH"
    with pytest.raises(ValueError):
        syn.create_synthetic_dataset("sideways", days=12, seed=99, db=db)
    from indicatorlab.research.individual import load_window
    win = load_window(ds["id"], "1min", "train", db=db)
    assert win.bars_in_split > 1000
