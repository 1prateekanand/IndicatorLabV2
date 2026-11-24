"""Configurable signals: continuous values, thresholds, crossovers, AND/OR, causality and storage."""
import numpy as np
import pytest

from indicatorlab import studio
from indicatorlab.db import get_db
from indicatorlab.research import individual as ind
from indicatorlab.research.signals import SpecError, get_spec, set_spec, validate_spec, votes_from_raw

CONT = '''def compute(df, length=5):
    closes = df["close"]
    out = []
    for i in range(len(closes)):
        if i < length:
            out.append(float("nan"))
        else:
            out.append(float(closes[i] - closes[i - length]))
    return out'''


def _cond(op, value=None, other=None):
    return {"op": op, "value": value, "other": other}


def _spec(long=None, short=None, logic_l="AND", logic_s="AND"):
    return {"mode": "conditions", "long": {"logic": logic_l, "conditions": long or []},
            "short": {"logic": logic_s, "conditions": short or []}}


# -- conditions on a known series -----------------------------------------------------------------
def test_threshold_and_or_logic():
    x = np.array([np.nan, 0.5, 1.5, 2.5, 0.2, 3.0])
    gt1 = _cond("gt", 1.0)
    lt3 = _cond("lt", 3.0)
    lookup = lambda k: None
    assert votes_from_raw(_spec([gt1]), x, lookup).tolist() == [0, 0, 1, 1, 0, 1]
    assert votes_from_raw(_spec([gt1, lt3], logic_l="AND"), x, lookup).tolist() == [0, 0, 1, 1, 0, 0]
    assert votes_from_raw(_spec([gt1, lt3], logic_l="OR"), x, lookup).tolist() == [0, 1, 1, 1, 1, 1]


def test_cross_above_and_below_fire_once_per_crossing():
    x = np.array([0.0, 0.5, 1.5, 1.2, 0.8, 1.1, 0.4])
    up = votes_from_raw(_spec([_cond("cross_above", 1.0)]), x, lambda k: None)
    down = votes_from_raw(_spec(short=[_cond("cross_below", 1.0)]), x, lambda k: None)
    assert up.tolist() == [0, 0, 1, 0, 0, 1, 0]
    assert down.tolist() == [0, 0, 0, 0, -1, 0, -1]


def test_crossover_against_another_indicator():
    x = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    other = np.array([2.5, 2.5, 2.5, 2.5, 2.5])
    v = votes_from_raw(_spec([_cond("cross_above", None, "other:key")]), x, lambda k: other)
    assert v.tolist() == [0, 0, 1, 0, 0]


def test_long_and_short_both_true_is_neutral():
    x = np.array([2.0, 2.0])
    v = votes_from_raw(_spec([_cond("gt", 1.0)], [_cond("gt", 1.5)]), x, lambda k: None)
    assert v.tolist() == [0, 0]


def test_missing_values_never_satisfy_a_condition():
    x = np.array([np.nan, 5.0])
    v = votes_from_raw(_spec([_cond("gt", 1.0)]), x, lambda k: None)
    assert v.tolist() == [0, 1]


# -- causality: a signal at bar t must not change when later bars are removed --------------------
def test_signals_are_causal_under_truncation():
    rng = np.random.default_rng(7)
    x = np.cumsum(rng.normal(size=400))
    spec = _spec([_cond("cross_above", 0.5), _cond("gt", -2.0)], [_cond("cross_below", -0.5)], logic_l="AND")
    full = votes_from_raw(spec, x, lambda k: None)
    for cut in (50, 137, 300):
        assert np.array_equal(votes_from_raw(spec, x[:cut], lambda k: None), full[:cut])


# -- validation ------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad", [
    {"mode": "sideways"},
    {"mode": "conditions", "long": {"conditions": [{"op": "wobble", "value": 1}]}},
    {"mode": "conditions", "long": {"conditions": [{"op": "gt"}]}},
    {"mode": "conditions", "long": {"logic": "XOR", "conditions": [{"op": "gt", "value": 1}]}},
    {"mode": "conditions", "long": {"conditions": [{"op": "equals_change", "other": "code:x"}]}},
    {"mode": "conditions"},
])
def test_invalid_specs_are_refused(bad):
    with pytest.raises(SpecError):
        validate_spec(bad)


def test_legacy_spec_is_the_default():
    assert validate_spec(None) == {"mode": "legacy"}
    assert validate_spec({"mode": "legacy"}) == {"mode": "legacy"}


def test_store_refuses_unknown_and_self_references(dataset_id):
    db = get_db()
    known = {"code:ref_rsi", "code:ref_macd"}
    with pytest.raises(LookupError):
        set_spec(db, "studio:999999", _spec([_cond("gt", 1)]), known)
    with pytest.raises(SpecError):
        set_spec(db, "code:ref_rsi", _spec([_cond("cross_above", None, "code:nope")]), known)
    with pytest.raises(SpecError):
        set_spec(db, "code:ref_rsi", _spec([_cond("cross_above", None, "code:ref_rsi")]), known)
    saved = set_spec(db, "code:ref_rsi", _spec([_cond("gt", 1)]), known)
    assert get_spec(db, "code:ref_rsi") == saved
    set_spec(db, "code:ref_rsi", None, known)
    assert get_spec(db, "code:ref_rsi") is None


# -- end to end: a studio indicator with continuous output, configured through rules ------------------
def test_continuous_studio_indicator_with_rules_trades_and_changes_cache_key(dataset_id):
    db = get_db()
    item = studio.create_indicator({"name": "sig_mom_cont", "code": CONT}, db)
    studio.update_indicator(item["id"], {"name": "sig_mom_cont", "code": CONT}, db)
    key = f"studio:{item['id']}"
    try:
        win = ind.load_window(dataset_id, "1min", "train", db=db)
        plain, _ = ind.discover_sources(db)
        src_plain = next(s for s in plain if s.key == key)
        set_spec(db, key, _spec([_cond("cross_above", 0.0)], [_cond("cross_below", 0.0)]),
                 {s.key for s in plain})
        available, _ = ind.discover_sources(db)
        src = next(s for s in available if s.key == key)
        assert src.spec is not None and src.code_hash != src_plain.code_hash, "a rule change must change the cache key"
        raw = ind.raw_series(src, win.full.iloc[:win.eval_end])
        assert np.isnan(raw[:5]).all() and np.isfinite(raw[10:]).all(), "continuous values reach the engine"
        metrics = ind.evaluate_source(src, win, ind.Rules(stop_loss_pct=0.5, target_pct=1.0))
        assert metrics["trades"] > 0
    finally:
        set_spec(db, key, None, {key})
        studio.delete_item("indicator", item["id"], db)


def test_signal_votes_match_direct_rule_on_candles(dataset_id):
    db = get_db()
    win = ind.load_window(dataset_id, "1min", "validation", db=db)
    prefix = win.full.iloc[:win.eval_end]
    from indicatorlab.registry import REGISTRY
    REGISTRY.discover()
    rsi = next(s for s in ind.discover_sources(db)[0] if s.name == "ref_rsi")
    spec = _spec([_cond("cross_above", 40.0)], [_cond("cross_below", 60.0)])
    from dataclasses import replace
    src = replace(rsi, spec=spec)
    votes = ind.signal_votes(src, prefix, {})
    raw_rsi = ind.raw_series(rsi, prefix)
    expected = votes_from_raw(spec, raw_rsi, lambda k: None)
    assert np.array_equal(votes, expected)
    assert set(np.unique(votes)).issubset({-1, 0, 1})
