"""SQLite persistence. A single connection is shared and guarded by a re-entrant lock."""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    source_file TEXT,
    rows INTEGER NOT NULL,
    start_ts TEXT NOT NULL,
    end_ts TEXT NOT NULL,
    imported_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candles (
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    ts TEXT NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
    volume REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (dataset_id, ts)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    agent TEXT NOT NULL,
    study_name TEXT,
    trial_number INTEGER,
    strategy_hash TEXT NOT NULL,
    dataset_id INTEGER NOT NULL,
    split TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    metrics_json TEXT,
    eligible INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    duration_s REAL,
    error TEXT
);
-- Never overwrite a finished result: one completed row per strategy/dataset/split/config.
CREATE UNIQUE INDEX IF NOT EXISTS ux_experiment_completed
    ON experiments(strategy_hash, dataset_id, split, config_hash) WHERE status = 'completed';
CREATE INDEX IF NOT EXISTS ix_experiments_study ON experiments(study_name);
CREATE TABLE IF NOT EXISTS validations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    strategy_hash TEXT NOT NULL,
    dataset_id INTEGER NOT NULL,
    segments_json TEXT NOT NULL,
    profitable_segments INTEGER NOT NULL,
    total_segments INTEGER NOT NULL,
    validation_net_profit REAL NOT NULL,
    validated INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS agents (
    name TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    current_task TEXT,
    algorithm TEXT,
    trials_processed INTEGER NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    progress REAL NOT NULL DEFAULT 0,
    started_at TEXT,
    last_heartbeat TEXT,
    detail_json TEXT
);
CREATE TABLE IF NOT EXISTS agent_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent TEXT NOT NULL,
    ts TEXT NOT NULL,
    level TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_agent_logs_agent ON agent_logs(agent, id);
CREATE TABLE IF NOT EXISTS agent_checkpoints (
    agent TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS optimization_runs (
    study_name TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS oos_tests (
    strategy_hash TEXT NOT NULL,
    dataset_id INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    PRIMARY KEY (strategy_hash, dataset_id)
);
"""

SCHEMA_VERSION = 6

# Additive migrations only: new tables or nullable columns. Each entry upgrades FROM the key version.
MIGRATIONS: dict[int, str] = {
    1: """
CREATE TABLE IF NOT EXISTS statistical_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    dataset_id INTEGER NOT NULL,
    split TEXT,
    experiment_id INTEGER,
    strategy_hash TEXT,
    config_json TEXT NOT NULL,
    seed INTEGER,
    status TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_stat_kind ON statistical_tests(kind, dataset_id, split);
""",
    2: """
CREATE TABLE IF NOT EXISTS studio_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('indicator', 'strategy')),
    name TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    registered INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (kind, name)
);
CREATE TABLE IF NOT EXISTS legacy_imports (
    source_key TEXT PRIMARY KEY,
    imported_at TEXT NOT NULL,
    detail_json TEXT NOT NULL
);
""",
    3: """
CREATE TABLE IF NOT EXISTS research_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    dataset_id INTEGER NOT NULL,
    split TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT,
    error TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS indicator_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    run_id INTEGER REFERENCES research_runs(id),
    source_key TEXT NOT NULL,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    code_hash TEXT NOT NULL,
    dataset_id INTEGER NOT NULL,
    split TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    metrics_json TEXT
);
-- One completed result per indicator version, dataset, split and rule set (cache key).
CREATE UNIQUE INDEX IF NOT EXISTS ux_indicator_eval_completed
    ON indicator_evaluations(source_key, code_hash, dataset_id, split, config_hash) WHERE status = 'completed';
""",
    4: """
-- v5: combination research. Cache key = sorted member code hashes + segment + rule set (see research/combination.py).
CREATE TABLE IF NOT EXISTS combination_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    run_id INTEGER REFERENCES research_runs(id),
    combo_key TEXT NOT NULL,
    size INTEGER NOT NULL,
    members_json TEXT NOT NULL,
    dataset_id INTEGER NOT NULL,
    segment TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    metrics_json TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_combo_eval_completed
    ON combination_evaluations(combo_key, dataset_id, segment, config_hash) WHERE status = 'completed';
CREATE TABLE IF NOT EXISTS walkforward_folds (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    run_id INTEGER NOT NULL REFERENCES research_runs(id),
    fold INTEGER NOT NULL,
    selection_range_json TEXT NOT NULL,
    test_range_json TEXT NOT NULL,
    purge_bars INTEGER NOT NULL,
    selected_members_json TEXT,
    selection_metrics_json TEXT,
    test_metrics_json TEXT,
    status TEXT NOT NULL,
    message TEXT,
    UNIQUE(run_id, fold)
);
CREATE TABLE IF NOT EXISTS combination_rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES research_runs(id),
    rank INTEGER NOT NULL,
    combo_key TEXT NOT NULL,
    size INTEGER NOT NULL,
    members_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    UNIQUE(run_id, rank)
);
CREATE TABLE IF NOT EXISTS run_artifacts (
    run_id INTEGER NOT NULL REFERENCES research_runs(id),
    kind TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    PRIMARY KEY(run_id, kind)
);
CREATE TABLE IF NOT EXISTS oos_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    run_id INTEGER NOT NULL UNIQUE REFERENCES research_runs(id),
    combo_key TEXT NOT NULL,
    members_json TEXT NOT NULL,
    window_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    curve_json TEXT NOT NULL
);
""",
    5: """
-- v6: configurable signal rules per indicator, and a synthetic-data flag on datasets.
CREATE TABLE IF NOT EXISTS signal_specs (
    source_key TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
ALTER TABLE datasets ADD COLUMN is_synthetic INTEGER NOT NULL DEFAULT 0;
""",
}

def _backup_file(path, from_version: int) -> str | None:
    """Copy the live database with SQLite's online backup API before any schema change."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    from datetime import datetime as _dt
    stamp = _dt.now().strftime("%Y%m%d-%H%M%S")
    target = path.parent / "backups" / f"{path.stem}-v{from_version}-{stamp}.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(str(path))
    dst = sqlite3.connect(str(target))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return str(target)


def migrate(conn: sqlite3.Connection, path) -> dict:
    """Bring the schema to SCHEMA_VERSION. Backs up an existing database first."""
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        has_v1 = conn.execute("SELECT 1 FROM sqlite_master WHERE name='experiments'").fetchone()
        current = 1 if has_v1 else 0
    else:
        current = int(row[0])
    if current > SCHEMA_VERSION:
        raise RuntimeError(f"database schema v{current} is newer than this application (v{SCHEMA_VERSION})")
    report = {"from": current, "to": SCHEMA_VERSION, "backup": None}
    if current == SCHEMA_VERSION:
        return report
    if current >= 1:
        conn.commit()
        report["backup"] = _backup_file(path, current)
    conn.executescript(SCHEMA)  # idempotent v1 base
    for version in range(max(current, 1), SCHEMA_VERSION):
        conn.executescript(MIGRATIONS[version])
    conn.execute("DELETE FROM schema_version")
    conn.execute("INSERT INTO schema_version(version) VALUES (?)", (SCHEMA_VERSION,))
    conn.commit()
    return report


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path=DB_PATH):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.migration_report = migrate(self._conn, path)

    @contextmanager
    def transaction(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        with self.transaction() as cur:
            cur.execute(sql, tuple(params))
            return cur.rowcount

    @staticmethod
    def dumps(obj: Any) -> str:
        return json.dumps(obj, sort_keys=True, default=_json_default)

    @staticmethod
    def loads(text: str | None) -> Any:
        return json.loads(text) if text else None


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, float) and o != o:  # NaN
        return None
    raise TypeError(f"not JSON serialisable: {type(o)}")


_db: Database | None = None
_db_lock = threading.Lock()


def get_db(path=None) -> Database:
    """Return the process-wide database. Pass a path only in tests before first use."""
    global _db
    with _db_lock:
        if _db is None:
            _db = Database(path or DB_PATH)
        return _db
