"""Indicator Studio and Strategy Studio: stored user code, validated in an isolated sandbox.

Behaviour mirrors the Express routes in ``server.js`` (same states, same error messages, same
validation order) so the existing Studio pages keep working when they are pointed at FastAPI.

* Items live in the ``studio_items`` table (migration v2). Ids of items imported from the old
  Express state file are preserved (see ``legacy.py``).
* User code runs in ``sandbox_runner.py`` as a separate ``python -I`` process with CPU and memory
  limits, exactly as before. The sandbox is reused, not reimplemented.
* Strategy validation metrics come from the Python backtest engine, which now accepts the custom
  ``exit_signals`` rule used by Strategy Studio code.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .config import DEFAULT_BACKTEST, DEFAULT_SPLIT, MIN_TRADES
from .db import Database, get_db, utcnow
from .engine import Exit, run_backtest
from .metrics import compute_metrics

SANDBOX = Path(__file__).with_name("sandbox_runner.py")
INDICATOR_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
STRATEGY_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_\- ]{1,79}$")
DIRECTIONS = ("both", "long", "short")
KINDS = ("indicator", "strategy")


# -- sandbox -------------------------------------------------------------------------------------
def run_sandbox(payload: dict, *, cpu_limit_s: int = 5, mem_limit_mb: int = 256, timeout_s: float = 10.0) -> dict:
    """Run the isolated sandbox process. Returns its JSON result or a SandboxError result."""
    request = {**payload, "cpu_limit_s": cpu_limit_s, "mem_limit_mb": mem_limit_mb}
    try:
        proc = subprocess.run([sys.executable, "-I", str(SANDBOX)], input=json.dumps(request),
                              capture_output=True, text=True, timeout=timeout_s, cwd=str(SANDBOX.parent))
    except subprocess.TimeoutExpired:
        return {"ok": False, "stage": "sandbox", "error": {"type": "SandboxError", "line": 1,
                                                           "message": f"sandbox timed out after {timeout_s}s"}}
    except OSError as exc:
        return {"ok": False, "stage": "sandbox", "error": {"type": "SandboxError", "line": 1, "message": str(exc)}}
    out = proc.stdout.strip()
    if out:
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            pass
    message = proc.stderr.strip() or f"Python sandbox exited with code {proc.returncode}"
    return {"ok": False, "stage": "sandbox", "error": {"type": "SandboxError", "line": 1, "message": message}}


# -- parameter specs (ported from normalizeParamSpecs) --------------------------------------------
def normalize_param_specs(params: list | None) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for p in params or []:
        name = str(p.get("name", "")).strip()
        if not name or not INDICATOR_NAME_RE.match(name):
            raise ValueError(f"Invalid parameter name '{name}'. Use lowercase snake_case.")
        if name in seen:
            raise ValueError(f"Duplicate parameter '{name}'")
        seen.add(name)
        kind = "float" if p.get("kind") == "float" else "int"
        try:
            low, high, default = float(p.get("low")), float(p.get("high")), float(p.get("default"))
        except (TypeError, ValueError):
            raise ValueError(f"Parameter '{name}' requires finite numeric low, high, and default") from None
        if not all(np.isfinite([low, high, default])):
            raise ValueError(f"Parameter '{name}' requires finite numeric low, high, and default")
        if low > high:
            raise ValueError(f"Parameter '{name}': low ({low:g}) > high ({high:g})")
        if default < low or default > high:
            raise ValueError(f"Parameter '{name}': default ({default:g}) must lie within [{low:g}, {high:g}]")
        step = p.get("step")
        out.append({
            "name": name, "kind": kind,
            "default": round(default) if kind == "int" else default,
            "low": round(low) if kind == "int" else low,
            "high": round(high) if kind == "int" else high,
            "step": None if step in (None, "") else float(step),
        })
    return out


def _defaults(params: list[dict]) -> dict:
    return {p["name"]: p["default"] for p in params}


# -- candles used for validation ------------------------------------------------------------------
def probe_candles(n: int = 400) -> list[dict]:
    """Deterministic synthetic bars used only when no dataset is chosen. Labelled as synthetic."""
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    start = pd.Timestamp("2000-01-03 09:15")
    rows = []
    for i, c in enumerate(close):
        o = close[i - 1] if i else c
        rows.append({"time": (start + pd.Timedelta(minutes=i)).isoformat(), "open": float(o),
                     "high": float(max(o, c) + 0.1), "low": float(min(o, c) - 0.1), "close": float(c),
                     "volume": 0.0})
    return rows


def resolve_validation_candles(dataset_id: int | None, max_bars: int = 1500, db: Database | None = None) -> tuple[list[dict], str]:
    if dataset_id:
        frame = data.load_candles(int(dataset_id))
        a, b = _train_bounds(len(frame))
        part = frame.iloc[a:b].tail(max_bars)
        rows = [{"time": t.isoformat(), "open": float(r.open), "high": float(r.high), "low": float(r.low),
                 "close": float(r.close), "volume": float(r.volume)} for t, r in zip(part.index, part.itertuples())]
        return rows, f"dataset #{dataset_id} (train split)"
    return probe_candles(400), "synthetic probe bars (import a dataset for historical market testing)"


def _train_bounds(n: int) -> tuple[int, int]:
    return 0, int(n * DEFAULT_SPLIT.train_end)


# -- storage ------------------------------------------------------------------------------------
def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"unknown studio kind '{kind}'")


def _item(row: dict) -> dict:
    payload = json.loads(row["payload_json"])
    payload["id"] = row["id"]
    payload["registered"] = bool(row["registered"])
    return payload


def list_items(kind: str, db: Database | None = None) -> list[dict]:
    _check_kind(kind)
    db = db or get_db()
    return [_item(r) for r in db.query("SELECT * FROM studio_items WHERE kind=? ORDER BY id", (kind,))]


def get_item(kind: str, item_id: int, db: Database | None = None) -> dict | None:
    _check_kind(kind)
    db = db or get_db()
    row = db.one("SELECT * FROM studio_items WHERE kind=? AND id=?", (kind, int(item_id)))
    return _item(row) if row else None


def _save(kind: str, item: dict, db: Database) -> dict:
    """Insert (id None) or overwrite an item. Keeps the Express field set exactly."""
    payload = {k: v for k, v in item.items() if k not in ("id", "registered")}
    registered = 1 if item.get("registered") else 0
    now = utcnow()
    with db.transaction() as cur:
        if item.get("id") is None:
            cur.execute("INSERT INTO studio_items(kind, name, payload_json, registered, created_at, updated_at)"
                        " VALUES (?,?,?,?,?,?)",
                        (kind, item["name"], json.dumps(payload, default=str), registered, now, now))
            item["id"] = cur.lastrowid
        else:
            cur.execute("UPDATE studio_items SET name=?, payload_json=?, registered=?, updated_at=?"
                        " WHERE kind=? AND id=?",
                        (item["name"], json.dumps(payload, default=str), registered, now, kind, item["id"]))
    return item


def _log(message: str, level: str = "info", db: Database | None = None) -> None:
    (db or get_db()).execute("INSERT INTO agent_logs(agent, ts, level, message) VALUES (?,?,?,?)",
                             ("research_agent", utcnow(), level, message[:2000]))


def registered_items(kind: str, db: Database | None = None) -> list[dict]:
    return [i for i in list_items(kind, db) if i.get("registered") and i.get("available")]


# -- indicator metadata ----------------------------------------------------------------------------
def validate_indicator_meta(body: dict, existing_id: int | None = None, db: Database | None = None) -> dict:
    name = str(body.get("name") or "").strip()
    if not INDICATOR_NAME_RE.match(name):
        raise ValueError("Indicator name must be snake_case (2-64 chars, lowercase letters/digits/underscores, "
                         "starting with a letter).")
    for other in list_items("indicator", db):
        if other["name"] == name and other["id"] != existing_id:
            raise ValueError(f"An indicator named '{name}' already exists.")
    code = str(body.get("code") or "")
    if not code.strip():
        raise ValueError("Python code cannot be empty.")
    return {
        "name": name,
        "version": str(body.get("version") or "1.0.0").strip() or "1.0.0",
        "category": str(body.get("category") or "Custom").strip() or "Custom",
        "description": str(body.get("description") or "").strip(),
        "code": code,
        "params": normalize_param_specs(body.get("params") or []),
    }


def create_indicator(body: dict, db: Database | None = None) -> dict:
    db = db or get_db()
    meta = validate_indicator_meta(body, None, db)
    now = utcnow()
    item = {**meta, "id": None, "status": "draft", "registered": False, "available": False,
            "status_reason": "draft — click Update Indicator to validate and register",
            "last_validated_at": None, "last_validation": None, "created_at": now, "updated_at": now}
    return _save("indicator", item, db)


def update_indicator_meta(item_id: int, body: dict, db: Database | None = None) -> dict:
    db = db or get_db()
    prev = get_item("indicator", item_id, db)
    if prev is None:
        raise LookupError("Indicator not found")
    meta = validate_indicator_meta(body, item_id, db)
    updated = {**prev, **meta, "status": "registered" if prev["registered"] else "draft", "updated_at": utcnow()}
    return _save("indicator", updated, db)


def delete_item(kind: str, item_id: int, db: Database | None = None) -> dict:
    _check_kind(kind)
    db = db or get_db()
    item = get_item(kind, item_id, db)
    if item is None:
        raise LookupError(f"{kind.capitalize()} not found")
    db.execute("DELETE FROM studio_items WHERE kind=? AND id=?", (kind, int(item_id)))
    if kind == "indicator":
        _log(f"Indicator '{item['name']}' deleted and removed from catalogue", db=db)
    else:
        _log(f"Strategy '{item['name']}' deleted and removed from catalogue", db=db)
    return {"deleted": True, "id": int(item_id), "name": item["name"]}


def validate_indicator(body: dict, db: Database | None = None) -> dict:
    code = str(body.get("code") or "")
    if not code.strip():
        raise ValueError("Python code cannot be empty.")
    params = normalize_param_specs(body.get("params") or [])
    syntax_only = bool(body.get("syntax_only", False))
    if syntax_only:
        candles, source = [], "syntax check only"
    else:
        candles, source = resolve_validation_candles(body.get("dataset_id") or None, db=db)
    result = run_sandbox({"mode": "indicator", "code": code, "candles": candles,
                          "params": _defaults(params), "syntax_only": syntax_only})
    return {**result, "data_source": source}


def update_indicator(item_id: int, body: dict, db: Database | None = None) -> tuple[dict, dict | None]:
    """Validate, then save and register. Returns (item, validation) and never starts optimisation.

    On failure the item is stored in the ``error`` state and ``ValidationFailed`` is raised.
    """
    db = db or get_db()
    prev = get_item("indicator", item_id, db)
    if prev is None:
        raise LookupError("Indicator not found")
    meta = validate_indicator_meta(body, item_id, db)
    defaults = _defaults(meta["params"])
    candles, source = resolve_validation_candles(body.get("dataset_id") or None, db=db)
    validation = run_sandbox({"mode": "indicator", "code": meta["code"], "candles": candles,
                              "params": defaults, "syntax_only": False})
    if not validation.get("ok"):
        err = validation.get("error") or {}
        reason = f"{err.get('type', 'ValidationError')} (line {err.get('line', 1)}): {err.get('message')}"
        failed = {**prev, **meta, "status": "error", "registered": False, "available": False,
                  "status_reason": reason, "last_validated_at": utcnow(),
                  "last_validation": {**validation, "data_source": source}, "updated_at": utcnow()}
        _save("indicator", failed, db)
        raise ValidationFailed(reason, {**validation, "data_source": source}, failed)
    registered = {**prev, **meta, "status": "registered", "registered": True, "available": True,
                  "status_reason": "ok (causal verified)" if validation.get("causal")
                  else "registered (warning: prefix check flagged look-ahead)",
                  "last_validated_at": utcnow(), "last_validation": {**validation, "data_source": source},
                  "updated_at": utcnow()}
    _save("indicator", registered, db)
    _log(f"Indicator '{meta['name']}' v{meta['version']} registered in dynamic catalogue "
         f"({len(registered_items('indicator', db))} total available). No optimization started automatically.", db=db)
    return registered, registered["last_validation"]


# -- strategy metadata ------------------------------------------------------------------------------
def validate_strategy_meta(body: dict, existing_id: int | None = None, db: Database | None = None) -> dict:
    name = str(body.get("name") or "").strip()
    if not STRATEGY_NAME_RE.match(name):
        raise ValueError("Strategy name must be 2-80 characters and start with a letter.")
    for other in list_items("strategy", db):
        if other["name"].lower() == name.lower() and other["id"] != existing_id:
            raise ValueError(f"A strategy named '{name}' already exists.")
    direction = body.get("direction") if body.get("direction") in DIRECTIONS else "both"
    code = str(body.get("code") or "")
    if not code.strip():
        raise ValueError("Python strategy code cannot be empty.")
    return {
        "name": name,
        "version": str(body.get("version") or "1.0.0").strip() or "1.0.0",
        "description": str(body.get("description") or "").strip(),
        "direction": direction,
        "stop_loss_pct": max(0.0, min(25.0, _num(body.get("stop_loss_pct"), 0.5))),
        "target_pct": max(0.0, min(50.0, _num(body.get("target_pct"), 1.0))),
        "entry_notes": str(body.get("entry_notes") or "").strip(),
        "exit_notes": str(body.get("exit_notes") or "").strip(),
        "linked_indicators": [str(x) for x in (body.get("linked_indicators") or [])],
        "code": code,
        "params": normalize_param_specs(body.get("params") or []),
    }


def _num(value, default: float) -> float:
    if value is None:
        return default
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def create_strategy(body: dict, db: Database | None = None) -> dict:
    db = db or get_db()
    meta = validate_strategy_meta(body, None, db)
    now = utcnow()
    item = {**meta, "id": None, "status": "draft", "registered": False, "available": False,
            "status_reason": "draft — click Update Strategy to validate and register",
            "last_validated_at": None, "last_validation": None, "created_at": now, "updated_at": now}
    return _save("strategy", item, db)


def update_strategy_meta(item_id: int, body: dict, db: Database | None = None) -> dict:
    db = db or get_db()
    prev = get_item("strategy", item_id, db)
    if prev is None:
        raise LookupError("Strategy not found")
    meta = validate_strategy_meta(body, item_id, db)
    updated = {**prev, **meta, "status": "registered" if prev["registered"] else "draft", "updated_at": utcnow()}
    return _save("strategy", updated, db)


def execute_strategy_validation(body: dict, db: Database | None = None) -> dict:
    """Run strategy code in the sandbox, then backtest its signals with the Python engine."""
    db = db or get_db()
    code = str(body.get("code") or "")
    params = normalize_param_specs(body.get("params") or [])
    linked = [str(x) for x in (body.get("linked_indicators") or [])]
    direction = body.get("direction") if body.get("direction") in DIRECTIONS else "both"
    stop_loss = _num(body.get("stop_loss_pct"), 0.5)
    target = _num(body.get("target_pct"), 1.0)
    syntax_only = bool(body.get("syntax_only", False))

    registry = {i["name"]: i for i in registered_items("indicator", db)}
    indicator_codes = []
    for name in linked:
        ind = registry.get(name)
        if ind:
            indicator_codes.append({"name": ind["name"], "code": ind["code"], "params": _defaults(ind.get("params", []))})

    if syntax_only:
        candles, source = [], "syntax check only"
    else:
        candles, source = resolve_validation_candles(body.get("dataset_id") or None, db=db)
    result = run_sandbox({"mode": "strategy", "code": code, "candles": candles, "params": _defaults(params),
                          "indicator_codes": indicator_codes, "syntax_only": syntax_only})
    if not result.get("ok") or syntax_only:
        return {**result, "data_source": source, "metrics": None}

    frame = _frame(candles)
    votes = np.asarray(result["signals"], dtype=int)
    exits = np.asarray(result.get("exit_signals") or [0] * len(votes), dtype=int)
    exit_rules = Exit(stop_loss_pct=float(result.get("stop_loss_pct") if result.get("stop_loss_pct") is not None else stop_loss),
                      target_pct=float(result.get("target_pct") if result.get("target_pct") is not None else target))
    bt = run_backtest(frame, votes, min_votes=1, direction=direction, exit_rules=exit_rules,
                      cfg=DEFAULT_BACKTEST, exit_signals=exits)
    metrics = compute_metrics(bt.trades, bt.equity, DEFAULT_BACKTEST.initial_capital, bt.final_capital)
    metrics["skipped_entries"] = int(bt.skipped_entries)
    metrics["eligible"] = metrics["trades"] >= MIN_TRADES
    out = {k: v for k, v in result.items() if k not in ("signals", "exit_signals")}
    return {**out, "data_source": source, "metrics": metrics}


def _frame(candles: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(candles)
    frame.index = pd.to_datetime(frame["time"])
    return frame[["open", "high", "low", "close", "volume"]].astype(float)


class ValidationFailed(Exception):
    def __init__(self, reason: str, validation: dict, item: dict):
        super().__init__(reason)
        self.validation = validation
        self.item = item


def update_strategy(item_id: int, body: dict, db: Database | None = None) -> tuple[dict, dict]:
    db = db or get_db()
    prev = get_item("strategy", item_id, db)
    if prev is None:
        raise LookupError("Strategy not found")
    meta = validate_strategy_meta(body, item_id, db)
    validation = execute_strategy_validation({**meta, "dataset_id": body.get("dataset_id") or None,
                                              "syntax_only": False}, db)
    if not validation.get("ok"):
        err = validation.get("error") or {}
        reason = f"{err.get('type', 'ValidationError')} (line {err.get('line', 1)}): {err.get('message')}"
        failed = {**prev, **meta, "status": "error", "registered": False, "available": False,
                  "status_reason": reason, "last_validated_at": utcnow(), "last_validation": validation,
                  "updated_at": utcnow()}
        _save("strategy", failed, db)
        raise ValidationFailed(reason, validation, failed)
    registered = {**prev, **meta, "status": "registered", "registered": True, "available": True,
                  "status_reason": "ok (validated in isolated sandbox)", "last_validated_at": utcnow(),
                  "last_validation": validation, "updated_at": utcnow()}
    _save("strategy", registered, db)
    _log(f"Strategy '{meta['name']}' v{meta['version']} registered in dynamic catalogue "
         f"({len(registered_items('strategy', db))} total available). No optimization started automatically.", db=db)
    return registered, validation
