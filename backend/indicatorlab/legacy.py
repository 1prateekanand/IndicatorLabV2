"""One-time, idempotent import of the Express JSON store into SQLite.

Source (read only, never modified):
    <DATA_DIR>/studio_state.json        datasets metadata and Studio items
    <DATA_DIR>/datasets/<id>.json       candle arrays for each Express dataset

Imported: datasets with their candles, and Studio indicator/strategy items (ids preserved).
Not imported here: Express experiments, validations, optimization runs, agent logs and checkpoints.
Those are reported as ``not_migrated`` so nothing is silently dropped.

Every imported source is recorded in ``legacy_imports``. Running the import again skips it.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from . import data
from .db import Database, utcnow

NOT_MIGRATED = ["experiments", "validations", "oos_tests", "statistical_tests", "optimization_runs",
                "agent_logs", "agent_checkpoints", "agents"]


def _already(db: Database, key: str) -> bool:
    return db.one("SELECT 1 FROM legacy_imports WHERE source_key=?", (key,)) is not None


def _mark(db: Database, key: str, detail: dict) -> None:
    db.execute("INSERT INTO legacy_imports(source_key, imported_at, detail_json) VALUES (?,?,?)",
               (key, utcnow(), json.dumps(detail, default=str)))


def import_express_store(data_dir: Path, db: Database, backup_dir: Path | None = None) -> dict:
    """Import datasets and Studio items from an Express data directory. Safe to run repeatedly."""
    state_file = Path(data_dir) / "studio_state.json"
    report = {"source": str(state_file), "datasets_imported": [], "datasets_skipped": [],
              "studio_imported": 0, "studio_skipped": 0, "not_migrated": NOT_MIGRATED, "backup": None}
    if not state_file.exists():
        report["status"] = "no_legacy_state"
        return report

    backups = Path(backup_dir) if backup_dir else Path(data_dir) / "backups"
    backups.mkdir(parents=True, exist_ok=True)
    stamp = utcnow().replace(":", "").replace("-", "")
    backup = backups / f"studio_state-{stamp}.json"
    shutil.copy2(state_file, backup)
    report["backup"] = str(backup)

    state = json.loads(state_file.read_text(encoding="utf-8"))
    for row in state.get("datasets", []):
        key = f"express:dataset:{row['id']}"
        if _already(db, key):
            report["datasets_skipped"].append({"express_id": row["id"], "reason": "already imported"})
            continue
        if db.one("SELECT id FROM datasets WHERE name=?", (row["name"],)):
            report["datasets_skipped"].append({"express_id": row["id"], "name": row["name"],
                                               "reason": "a dataset with this name already exists in SQLite"})
            continue
        candle_file = Path(data_dir) / "datasets" / f"{row['id']}.json"
        if not candle_file.exists():
            report["datasets_skipped"].append({"express_id": row["id"], "reason": "candle file missing"})
            continue
        candles = json.loads(candle_file.read_text(encoding="utf-8"))
        new_id = _insert_dataset(db, row, candles)
        _mark(db, key, {"sqlite_dataset_id": new_id, "rows": len(candles)})
        report["datasets_imported"].append({"express_id": row["id"], "sqlite_id": new_id, "rows": len(candles)})

    for kind, list_key in (("indicator", "studioIndicators"), ("strategy", "studioStrategies")):
        for item in state.get(list_key, []):
            key = f"express:studio:{kind}:{item['id']}"
            if _already(db, key):
                report["studio_skipped"] += 1
                continue
            clash = db.one("SELECT id FROM studio_items WHERE kind=? AND name=?", (kind, item["name"]))
            if clash:
                report["studio_skipped"] += 1
                continue
            payload = {k: v for k, v in item.items() if k not in ("id", "registered")}
            now = utcnow()
            with db.transaction() as cur:
                cur.execute("INSERT INTO studio_items(id, kind, name, payload_json, registered, created_at, updated_at)"
                            " VALUES (?,?,?,?,?,?,?)",
                            (item["id"], kind, item["name"], json.dumps(payload, default=str),
                             1 if item.get("registered") else 0, item.get("created_at") or now, item.get("updated_at") or now))
            _mark(db, key, {"id": item["id"]})
            report["studio_imported"] += 1

    report["status"] = "imported"
    return report


def _insert_dataset(db: Database, row: dict, candles: list[dict]) -> int:
    frame = pd.DataFrame(candles)
    frame.index = pd.to_datetime(frame["time"])
    frame = frame[~frame.index.duplicated(keep="first")].sort_index()
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame["volume"] = frame["volume"].fillna(0.0)
    with db.transaction() as cur:
        cur.execute(
            "INSERT INTO datasets(name, symbol, timeframe, source_file, rows, start_ts, end_ts, imported_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (row["name"], row.get("symbol", "UNKNOWN"), row.get("timeframe", "1min"),
             f"express:{row.get('source_file', 'unknown')}", len(frame),
             frame.index[0].isoformat(), frame.index[-1].isoformat(), utcnow()))
        dataset_id = cur.lastrowid
        cur.executemany(
            "INSERT INTO candles(dataset_id, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
            [(dataset_id, ts.isoformat(), float(o), float(h), float(l), float(c), float(v))
             for ts, o, h, l, c, v in zip(frame.index, frame["open"], frame["high"], frame["low"],
                                          frame["close"], frame["volume"])])
    data.load_candles.cache_clear()
    return dataset_id
