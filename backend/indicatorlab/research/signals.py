"""Configurable signal rules on continuous indicator output.

An indicator's signal can be:

* ``legacy``: the -1 / 0 / +1 votes the indicator returns (unchanged behaviour), or
* ``conditions``: built from its raw continuous values with two blocks, ``long`` and ``short``. Each
  block combines its conditions with AND or OR. A bar votes +1 when only the long block holds, -1 when
  only the short block holds, and 0 otherwise (including when both blocks hold: a conflict is neutral).

Conditions (``x`` is the indicator's value, ``y`` is a constant ``value`` or another indicator's value
given as ``other``):

    gt, ge, lt, le          x compared with y on the current bar
    cross_above             x was at or below y on the previous bar and is above y now
    cross_below             x was at or above y on the previous bar and is below y now
    equals_change           x equals ``value`` now and did not on the previous bar (discrete states)

Causality: every condition reads only the current and the previous bar, and the signal of bar t is
used to trade from the open of bar t+1 by the engine. Missing values never satisfy a condition.
"""
from __future__ import annotations

import json
from typing import Callable

import numpy as np

OPS = ("gt", "ge", "lt", "le", "cross_above", "cross_below", "equals_change")
LOGICS = ("AND", "OR")
BLOCKS = ("long", "short")


class SpecError(ValueError):
    pass


def legacy_spec() -> dict:
    return {"mode": "legacy"}


def validate_spec(spec: dict | None) -> dict:
    """Return a normalised spec or raise SpecError. ``None`` means legacy."""
    if spec is None:
        return legacy_spec()
    if not isinstance(spec, dict):
        raise SpecError("signal spec must be an object")
    mode = spec.get("mode", "legacy")
    if mode == "legacy":
        return legacy_spec()
    if mode != "conditions":
        raise SpecError("mode must be 'legacy' or 'conditions'")
    out = {"mode": "conditions"}
    for block in BLOCKS:
        b = spec.get(block)
        if b is None:
            out[block] = {"logic": "AND", "conditions": []}
            continue
        logic = str(b.get("logic", "AND")).upper()
        if logic not in LOGICS:
            raise SpecError(f"{block}.logic must be AND or OR")
        conds = b.get("conditions", [])
        if not isinstance(conds, list) or len(conds) > 10:
            raise SpecError(f"{block}.conditions must be a list of at most 10 conditions")
        norm = []
        for c in conds:
            op = c.get("op")
            if op not in OPS:
                raise SpecError(f"unknown condition '{op}'; use one of {', '.join(OPS)}")
            other = c.get("other") or None
            if other is None:
                if c.get("value") is None:
                    raise SpecError(f"condition '{op}' needs a value or another indicator")
                value = float(c["value"])
                if not np.isfinite(value):
                    raise SpecError("condition value must be finite")
            else:
                if op == "equals_change":
                    raise SpecError("equals_change compares with a value, not another indicator")
                value = None
            norm.append({"op": op, "value": value, "other": other})
        out[block] = {"logic": logic, "conditions": norm}
    if not out["long"]["conditions"] and not out["short"]["conditions"]:
        raise SpecError("add at least one long or short condition")
    return out


def spec_json(spec: dict | None) -> str:
    return json.dumps(validate_spec(spec), sort_keys=True)


def _shift(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    out[0] = np.nan
    out[1:] = x[:-1]
    return out


def _condition(c: dict, x: np.ndarray, lookup: Callable[[str], np.ndarray]) -> np.ndarray:
    op = c["op"]
    if c.get("other"):
        y = lookup(c["other"])
    else:
        y = np.full_like(x, float(c["value"]))
    with np.errstate(invalid="ignore"):
        if op == "gt":
            out = x > y
        elif op == "ge":
            out = x >= y
        elif op == "lt":
            out = x < y
        elif op == "le":
            out = x <= y
        elif op == "cross_above":
            out = (_shift(x) <= _shift(y)) & (x > y)
        elif op == "cross_below":
            out = (_shift(x) >= _shift(y)) & (x < y)
        elif op == "equals_change":
            out = (x == y) & (_shift(x) != y)
        else:  # pragma: no cover - validated earlier
            raise SpecError(op)
    return np.where(np.isnan(x) | np.isnan(y), False, out).astype(bool)


def _block(block: dict, x: np.ndarray, lookup) -> np.ndarray | None:
    if not block["conditions"]:
        return None
    masks = [_condition(c, x, lookup) for c in block["conditions"]]
    if block["logic"] == "AND":
        return np.logical_and.reduce(masks)
    return np.logical_or.reduce(masks)


def votes_from_raw(spec: dict, raw: np.ndarray, lookup: Callable[[str], np.ndarray]) -> np.ndarray:
    """Votes (-1, 0, +1) for a validated ``conditions`` spec. ``lookup(key)`` returns another indicator's
    raw series on the same candles."""
    spec = validate_spec(spec)
    if spec["mode"] != "conditions":
        raise SpecError("votes_from_raw needs a conditions spec")
    x = np.asarray(raw, dtype=float)
    longs = _block(spec["long"], x, lookup)
    shorts = _block(spec["short"], x, lookup)
    n = len(x)
    long_m = longs if longs is not None else np.zeros(n, dtype=bool)
    short_m = shorts if shorts is not None else np.zeros(n, dtype=bool)
    votes = np.zeros(n, dtype=np.int8)
    votes[long_m & ~short_m] = 1
    votes[short_m & ~long_m] = -1
    return votes


def references(spec: dict | None) -> set[str]:
    """Other indicator keys a spec reads from."""
    spec = validate_spec(spec)
    keys = set()
    if spec["mode"] == "conditions":
        for block in BLOCKS:
            for c in spec[block]["conditions"]:
                if c.get("other"):
                    keys.add(c["other"])
    return keys


# -- storage ---------------------------------------------------------------------------------------
def get_spec(db, source_key: str) -> dict | None:
    row = db.one("SELECT spec_json FROM signal_specs WHERE source_key=?", (source_key,))
    return json.loads(row["spec_json"]) if row else None


def set_spec(db, source_key: str, spec: dict | None, known_keys: set[str]) -> dict | None:
    """Save (or clear with ``None``) the rules of one indicator. Unknown keys are refused."""
    from ..db import utcnow
    if source_key not in known_keys:
        raise LookupError(f"indicator '{source_key}' is not registered")
    if spec is None or validate_spec(spec)["mode"] == "legacy":
        db.execute("DELETE FROM signal_specs WHERE source_key=?", (source_key,))
        return None
    normalised = validate_spec(spec)
    for ref in references(normalised):
        if ref not in known_keys:
            raise SpecError(f"rule refers to '{ref}', which is not a registered indicator")
        if ref == source_key:
            raise SpecError("an indicator cannot be compared with itself")
    db.execute("INSERT OR REPLACE INTO signal_specs(source_key, spec_json, updated_at) VALUES (?,?,?)",
               (source_key, json.dumps(normalised, sort_keys=True), utcnow()))
    return normalised
