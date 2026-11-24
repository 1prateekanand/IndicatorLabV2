"""Studio migration tests: stored items, sandboxed validation, custom exits, legacy import, upload."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from indicatorlab import studio
from indicatorlab.config import DEFAULT_BACKTEST
from indicatorlab.db import Database
from indicatorlab.engine import Exit, run_backtest
from indicatorlab.legacy import import_express_store

GOOD_INDICATOR = '''
def compute(df, length=5):
    closes = df["close"]
    out = []
    for i in range(len(closes)):
        if i < length:
            out.append(0)
        else:
            out.append(1 if closes[i] > closes[i - length] else -1)
    return out
'''

BAD_INDICATOR = "def compute(df, length=5)\n    return []\n"  # syntax error

GOOD_STRATEGY = '''
def generate_signals(df, indicators, length=3):
    closes = df["close"]
    entry = []
    exits = []
    for i in range(len(closes)):
        if i < length:
            entry.append(0)
            exits.append(0)
        else:
            up = closes[i] > closes[i - length]
            entry.append(1 if up else 0)
            exits.append(-1 if not up else 0)
    return {"signals": entry, "exit_signals": exits}
'''


def _params():
    return [{"name": "length", "kind": "int", "low": 2, "high": 20, "default": 5}]


# -- stored items -------------------------------------------------------------------------------
def test_indicator_crud_rules(tmp_path):
    db = Database(tmp_path / "crud.db")
    item = studio.create_indicator({"name": "my_ind", "code": GOOD_INDICATOR, "params": _params()}, db)
    assert item["id"] and item["registered"] is False and item["status"] == "draft"
    with pytest.raises(ValueError, match="already exists"):
        studio.create_indicator({"name": "my_ind", "code": GOOD_INDICATOR}, db)
    with pytest.raises(ValueError, match="snake_case"):
        studio.create_indicator({"name": "Bad Name", "code": GOOD_INDICATOR}, db)
    with pytest.raises(ValueError, match="cannot be empty"):
        studio.create_indicator({"name": "empty_ind", "code": "  "}, db)
    with pytest.raises(ValueError, match="default"):
        studio.create_indicator({"name": "bad_params", "code": GOOD_INDICATOR,
                                 "params": [{"name": "pp", "low": 5, "high": 9, "default": 1}]}, db)
    updated = studio.update_indicator_meta(item["id"], {"name": "my_ind", "code": GOOD_INDICATOR,
                                                        "category": "Trend"}, db)
    assert updated["category"] == "Trend" and updated["id"] == item["id"]
    assert studio.delete_item("indicator", item["id"], db)["deleted"] is True
    with pytest.raises(LookupError):
        studio.delete_item("indicator", item["id"], db)


def test_strategy_names_are_case_insensitive_unique(tmp_path):
    db = Database(tmp_path / "strat.db")
    studio.create_strategy({"name": "Trend Follow", "code": GOOD_STRATEGY}, db)
    with pytest.raises(ValueError, match="already exists"):
        studio.create_strategy({"name": "trend follow", "code": GOOD_STRATEGY}, db)


# -- sandbox validation -------------------------------------------------------------------------
def test_indicator_validation_success_and_failure_are_recorded(tmp_path):
    db = Database(tmp_path / "val.db")
    good = studio.create_indicator({"name": "ok_ind", "code": GOOD_INDICATOR, "params": _params()}, db)
    registered, validation = studio.update_indicator(good["id"], {"name": "ok_ind", "code": GOOD_INDICATOR, "params": _params()}, db)
    assert registered["registered"] is True and registered["available"] is True
    assert validation["ok"] is True and validation["data_source"].startswith("synthetic probe")
    assert studio.registered_items("indicator", db)[0]["name"] == "ok_ind"

    bad = studio.create_indicator({"name": "broken_ind", "code": BAD_INDICATOR}, db)
    with pytest.raises(studio.ValidationFailed) as info:
        studio.update_indicator(bad["id"], {"name": "broken_ind", "code": BAD_INDICATOR}, db)
    assert info.value.item["status"] == "error" and info.value.item["registered"] is False
    stored = studio.get_item("indicator", bad["id"], db)
    assert stored["status"] == "error" and "(line " in stored["status_reason"]


def test_registration_writes_agent_log_and_never_starts_optimisation(tmp_path):
    db = Database(tmp_path / "log.db")
    item = studio.create_indicator({"name": "logged_ind", "code": GOOD_INDICATOR, "params": _params()}, db)
    registered, _ = studio.update_indicator(item["id"], {"name": "logged_ind", "code": GOOD_INDICATOR, "params": _params()}, db)
    logs = db.query("SELECT message FROM agent_logs WHERE agent='research_agent'")
    assert any("logged_ind" in row["message"] and "No optimization started" in row["message"] for row in logs)


def test_strategy_validation_produces_engine_metrics(dataset_id):
    result = studio.execute_strategy_validation({"code": GOOD_STRATEGY, "params": _params(),
                                                 "dataset_id": dataset_id, "direction": "both"})
    assert result["ok"] is True
    assert result["data_source"].startswith(f"dataset #{dataset_id}")
    assert result["metrics"] is not None
    assert {"trades", "net_profit", "win_rate", "eligible", "max_drawdown_pct"} <= set(result["metrics"])
    assert "signals" not in result and "exit_signals" not in result


def test_strategy_syntax_only_returns_no_metrics():
    result = studio.execute_strategy_validation({"code": GOOD_STRATEGY, "syntax_only": True})
    assert result["metrics"] is None and result["data_source"] == "syntax check only"


# -- custom exit rule ---------------------------------------------------------------------------
def _frame(n=40):
    idx = pd.date_range("2024-01-02 09:15", periods=n, freq="min")
    close = 100 + np.arange(n) * 0.01
    return pd.DataFrame({"open": close, "high": close + 0.02, "low": close - 0.02, "close": close,
                         "volume": 1.0}, index=idx)


def test_custom_exit_signal_closes_position_with_strategy_reason():
    df = _frame()
    votes = np.zeros(len(df), dtype=int)
    votes[2] = 1  # enter long at the close of bar 2
    exits = np.zeros(len(df), dtype=int)
    exits[10] = 1  # custom exit at bar 10: matches the open long side
    res = run_backtest(df, votes, min_votes=1, direction="long", exit_rules=Exit(0.0, 0.0),
                       cfg=DEFAULT_BACKTEST, exit_signals=exits)
    reasons = [t["reason"] for t in res.trades]
    assert "strategy_exit" in reasons
    assert res.trades[0]["exit_time"] == df.index[11].isoformat()  # filled at the next bar's open


def test_no_exit_signals_keeps_original_behaviour():
    df = _frame()
    votes = np.zeros(len(df), dtype=int)
    votes[2] = 1
    res = run_backtest(df, votes, min_votes=1, direction="long", exit_rules=Exit(0.0, 0.0), cfg=DEFAULT_BACKTEST)
    assert all(t["reason"] != "strategy_exit" for t in res.trades)


# -- legacy import ------------------------------------------------------------------------------
def _write_express_store(root: Path, n=600):
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp("2025-01-02 09:15")
    candles = []
    for i in range(n):
        c = 100 + i * 0.05
        candles.append({"time": (start + pd.Timedelta(minutes=i)).isoformat(), "open": c, "high": c + 0.1,
                        "low": c - 0.1, "close": c, "volume": 0.0})
    (root / "datasets" / "7.json").write_text(json.dumps(candles))
    state = {
        "nextIds": {"dataset": 8, "indicator": 4, "strategy": 3},
        "datasets": [{"id": 7, "name": "legacy_nifty", "symbol": "NIFTY50", "timeframe": "1min",
                      "source_file": "upload", "rows": n, "start_ts": candles[0]["time"],
                      "end_ts": candles[-1]["time"], "imported_at": "2026-01-01T00:00:00+00:00"}],
        "studioIndicators": [{"id": 3, "name": "legacy_ind", "version": "1.0.0", "category": "Custom",
                              "description": "", "code": GOOD_INDICATOR, "params": _params(), "status": "registered",
                              "registered": True, "available": True, "status_reason": "ok", "created_at": "2025",
                              "updated_at": "2025"}],
        "studioStrategies": [],
        "experiments": [{"id": 1}],
    }
    (root / "studio_state.json").write_text(json.dumps(state))
    return candles


def test_legacy_import_preserves_ids_and_is_idempotent(tmp_path):
    express = tmp_path / "express"
    express.mkdir()
    _write_express_store(express)
    db = Database(tmp_path / "legacy.db")
    first = import_express_store(express, db, tmp_path / "backups")
    assert first["status"] == "imported"
    assert len(first["datasets_imported"]) == 1 and first["studio_imported"] == 1
    assert Path(first["backup"]).exists()
    assert "experiments" in first["not_migrated"]

    sqlite_id = first["datasets_imported"][0]["sqlite_id"]
    rows = db.one("SELECT COUNT(*) AS n FROM candles WHERE dataset_id=?", (sqlite_id,))["n"]
    assert rows == 600
    kept = studio.get_item("indicator", 3, db)
    assert kept is not None and kept["name"] == "legacy_ind" and kept["registered"] is True

    second = import_express_store(express, db, tmp_path / "backups")
    assert second["datasets_imported"] == [] and second["studio_imported"] == 0
    assert db.one("SELECT COUNT(*) AS n FROM datasets WHERE name='legacy_nifty'")["n"] == 1


def test_legacy_import_without_state_file_is_a_noop(tmp_path):
    db = Database(tmp_path / "none.db")
    report = import_express_store(tmp_path / "empty", db, tmp_path / "backups")
    assert report["status"] == "no_legacy_state"


# -- HTTP surface -------------------------------------------------------------------------------
@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from indicatorlab.api import app

    with TestClient(app) as c:
        yield c


def test_studio_http_shapes_match_express_contract(client):
    created = client.post("/api/studio/indicators", json={"name": "http_ind", "code": GOOD_INDICATOR,
                                                          "params": _params()})
    assert created.status_code == 200
    item_id = created.json()["id"]
    listing = client.get("/api/studio/indicators").json()
    assert set(listing) >= {"total", "registered_count", "items"}

    failed = client.post(f"/api/studio/indicators/{item_id}/update", json={"name": "http_ind", "code": BAD_INDICATOR})
    assert failed.status_code == 400
    body = failed.json()
    assert "detail" in body and "validation" in body and body["indicator"]["status"] == "error"

    ok = client.post(f"/api/studio/indicators/{item_id}/update", json={"name": "http_ind", "code": GOOD_INDICATOR, "params": _params()})
    assert ok.status_code == 200 and ok.json()["auto_started_optimization"] is False

    missing = client.put("/api/studio/indicators/999999", json={"name": "x_ind", "code": GOOD_INDICATOR})
    assert missing.status_code == 404
    assert client.delete(f"/api/studio/indicators/{item_id}").json()["deleted"] is True


def test_upload_endpoint_imports_csv_text(client):
    start = pd.Timestamp("2025-03-03 09:15")
    lines = ["Timestamp,Open,High,Low,Close,Volume"]
    for i in range(600):
        c = 200 + i * 0.02
        lines.append(f"{(start + pd.Timedelta(minutes=i)).isoformat()},{c},{c + 0.1},{c - 0.1},{c},0")
    r = client.post("/api/datasets/upload", json={"name": "upload_test", "symbol": "NIFTY50",
                                                  "timeframe": "1min", "csv_text": "\n".join(lines)})
    assert r.status_code == 200, r.text
    assert r.json()["rows"] == 600 and r.json()["symbol"] == "NIFTY50"
    bad = client.post("/api/datasets/upload", json={"name": "bad_upload", "symbol": "X", "csv_text": "a,b\n1,2"})
    assert bad.status_code == 400


def test_sandbox_accepts_params_dict_and_keyword_styles():
    dict_style = '''def compute_indicator(df, params):
    n = params.get("length", 3)
    closes = df["close"]
    return [0 if i < n else (1 if closes[i] > closes[i - n] else -1) for i in range(len(closes))]'''
    kw_style = GOOD_INDICATOR
    for code in (dict_style, kw_style):
        res = studio.run_sandbox({"mode": "indicator", "code": code, "candles": studio.probe_candles(100),
                                  "params": {"length": 4}, "syntax_only": False})
        assert res["ok"] is True, res
        assert res["bars"] == 100


# -- candles endpoint for Market Data -----------------------------------------------------------
def test_candles_range_is_aggregated_not_cut(client, dataset_id):
    from indicatorlab.data import load_candles
    frame = load_candles(dataset_id)
    start, end = frame.index[0].isoformat(), frame.index[-1].isoformat()
    full = client.get(f"/api/datasets/{dataset_id}/candles",
                      params={"timeframe": "1min", "start": start, "end": end, "limit": 100}).json()
    assert full["total_bars"] == len(frame)
    assert full["aggregated"] is True and full["source_bars_per_candle"] > 1
    assert len(full["bars"]) <= 100
    # the first aggregated candle starts at the range start and the highs/lows cover the source bars
    first = full["bars"][0]
    assert first["time"] == start
    assert first["high"] >= first["low"]
    total_high = max(b["high"] for b in full["bars"])
    assert total_high == pytest.approx(float(frame["high"].max()))


def test_candles_default_returns_latest_bars(client, dataset_id):
    body = client.get(f"/api/datasets/{dataset_id}/candles", params={"timeframe": "5min", "limit": 50}).json()
    assert len(body["bars"]) == 50 and body["aggregated"] is False and body["truncated"] is False
