"""Background runs for individual indicator evaluation (uses the existing ManagedWorker lifecycle)."""
from __future__ import annotations

import json
import threading
import traceback
from dataclasses import dataclass

from ..agents.runtime import ManagedWorker, StopRequested, WorkerContext
from ..db import Database, get_db, utcnow
from .individual import (Rules, cached_result, config_hash, discover_sources, evaluate_source, latest_any_hash_results,
                         load_window, store_result)

_LOCK = threading.Lock()
_ACTIVE: dict[int, ManagedWorker] = {}
WORKER_PREFIX = "research_individual"


@dataclass(frozen=True)
class RunRequest:
    dataset_id: int
    split: str = "train"
    timeframe: str = "1min"
    start: str | None = None
    end: str | None = None
    rules: Rules = Rules()


def _request_dict(req: RunRequest) -> dict:
    return {"dataset_id": req.dataset_id, "split": req.split, "timeframe": req.timeframe,
            "start": req.start, "end": req.end, "rules": req.rules.to_dict()}


def start_run(req: RunRequest, db: Database | None = None) -> dict:
    """Validate, create the run row and start its worker. Only one run is active at a time."""
    db = db or get_db()
    req.rules.validate()
    win = load_window(req.dataset_id, req.timeframe, req.split, req.start, req.end, db)
    with _LOCK:
        if any(w.is_alive for w in _ACTIVE.values()):
            raise RuntimeError("an individual evaluation is already running; stop it or wait for it to finish")
        ds = db.one("SELECT rows, end_ts FROM datasets WHERE id=?", (req.dataset_id,))
        cfg = config_hash({"timeframe": req.timeframe, "start": req.start, "end": req.end, "split": req.split,
                           "eval_start": win.start_ts, "eval_end": win.end_ts, "bars": win.bars_in_split},
                          req.rules, ds["rows"], ds["end_ts"])
        with db.transaction() as cur:
            cur.execute(
                "INSERT INTO research_runs(created_at, kind, dataset_id, split, config_hash, config_json, status, message)"
                " VALUES (?,?,?,?,?,?,?,?)",
                (utcnow(), "individual", req.dataset_id, req.split, cfg,
                 json.dumps(_request_dict(req), default=str), "running", "starting"))
            run_id = int(cur.lastrowid)
        worker = ManagedWorker(f"{WORKER_PREFIX}_{run_id}", lambda ctx: _execute(ctx, run_id, req, cfg, db),
                               throttle=False)
        _ACTIVE[run_id] = worker
        worker.start()
    return run_status(run_id, db)


def _execute(ctx: WorkerContext, run_id: int, req: RunRequest, cfg: str, db: Database) -> None:
    status, error = "completed", None
    try:
        win = load_window(req.dataset_id, req.timeframe, req.split, req.start, req.end, db)
        sources, unavailable = discover_sources(db)
        total = len(sources)
        ctx.set_task(f"Individual evaluation: {total} indicators on {win.dataset_name} ({req.timeframe}, {req.split})",
                     algorithm="single-indicator backtest", progress=0.0)
        done = cached = failed = 0
        for src in sources:
            ctx.checkpoint()  # pause blocks here; stop raises StopRequested
            if cached_result(db, src, req.dataset_id, req.split, cfg) is not None:
                cached += 1
            else:
                try:
                    metrics = evaluate_source(src, win, req.rules)
                    store_result(db, run_id=run_id, src=src, win=win, cfg_hash=cfg, status="completed", metrics=metrics)
                    ctx.count_trial()
                except StopRequested:
                    raise
                except Exception as exc:  # noqa: BLE001 - one bad indicator must not stop the run
                    failed += 1
                    ctx.count_error()
                    store_result(db, run_id=run_id, src=src, win=win, cfg_hash=cfg, status="failed",
                                 metrics=None, error=f"{type(exc).__name__}: {exc}")
                    ctx.log(f"{src.name} failed: {type(exc).__name__}: {exc}", "error")
            done += 1
            progress = done / total if total else 1.0
            ctx.set_task(f"Individual evaluation {done}/{total} ({cached} cached, {failed} failed)", progress=progress)
            db.execute("UPDATE research_runs SET progress=?, message=? WHERE id=?",
                       (progress, f"{done}/{total}", run_id))
        for u in unavailable:
            store_result(db, run_id=run_id, src=u, win=win, cfg_hash=cfg, status="unavailable",
                         metrics=None, error=u.status_reason)
        ctx.log(f"individual evaluation finished: {total} evaluated, {cached} from cache, {failed} failed, "
                f"{len(unavailable)} unavailable")
    except StopRequested:
        status = "stopped"
    except Exception as exc:  # noqa: BLE001 - recorded on the run row
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        ctx.log(traceback.format_exc()[-1500:], "error")
    finally:
        db.execute("UPDATE research_runs SET status=?, error=?, finished_at=?, message=? WHERE id=?",
                   (status, error, utcnow(), status, run_id))
        with _LOCK:
            _ACTIVE.pop(run_id, None)


def control_run(run_id: int, action: str, db: Database | None = None) -> dict:
    db = db or get_db()
    with _LOCK:
        worker = _ACTIVE.get(run_id)
    if worker is None:
        raise LookupError(f"run {run_id} is not active")
    if action == "pause":
        worker.pause()
    elif action == "resume":
        worker.resume()
    elif action == "stop":
        worker.stop(wait=0)
    else:
        raise ValueError("action must be pause, resume or stop")
    return run_status(run_id, db)


def run_status(run_id: int, db: Database | None = None) -> dict:
    db = db or get_db()
    row = db.one("SELECT * FROM research_runs WHERE id=?", (run_id,))
    if row is None:
        raise LookupError(f"run {run_id} not found")
    with _LOCK:
        worker = _ACTIVE.get(run_id)
    return {"id": row["id"], "created_at": row["created_at"], "kind": row["kind"], "dataset_id": row["dataset_id"],
            "split": row["split"], "config_hash": row["config_hash"], "status": row["status"],
            "progress": row["progress"], "message": row["message"], "error": row["error"],
            "finished_at": row["finished_at"], "request": json.loads(row["config_json"]),
            "live": bool(worker and worker.is_alive), "paused": bool(worker and worker.ctx.pause_event.is_set())}


def recover_after_restart(db: Database | None = None) -> int:
    """Runs left active by a previous process cannot be resumed; mark them interrupted."""
    db = db or get_db()
    n = db.execute("UPDATE research_runs SET status='interrupted', finished_at=?, message='interrupted by restart'"
                   " WHERE status='running'", (utcnow(),))
    return int(n or 0)


def individual_table(dataset_id: int, split: str = "train", run_id: int | None = None,
                     db: Database | None = None) -> dict:
    """Every indicator currently discovered, merged with the results of one run (default: latest).

    Rows are built from the database and live discovery only. Nothing is estimated.
    """
    from .individual import latest_results  # local import keeps module import light
    db = db or get_db()
    if run_id is None:
        row = db.one("SELECT id FROM research_runs WHERE kind='individual' AND dataset_id=? AND split=?"
                     " ORDER BY id DESC LIMIT 1", (dataset_id, split))
        run_id = row["id"] if row else None
    run = run_status(run_id, db) if run_id else None
    results = latest_results(db, dataset_id, split, run["config_hash"]) if run else {}
    older = latest_any_hash_results(db, dataset_id, split, run["config_hash"]) if run else {}
    sources, unavailable = discover_sources(db)
    rows = []
    for src in sources:
        hit = results.get(src.key)
        if hit and hit["code_hash"] == src.code_hash:
            rows.append({"source_key": src.key, "name": src.name, "source": src.source, "category": src.category,
                         "status": "evaluated", "evaluated_at": hit["created_at"], **hit["metrics"]})
        elif older.get(src.key):  # a result exists, but the code or parameters changed since it was computed
            rows.append({"source_key": src.key, "name": src.name, "source": src.source, "category": src.category,
                         "status": "changed_since_run",
                         "reason": "code or parameters changed after this run; re-run to evaluate"})
        else:      # newly registered, or not reached by this run
            rows.append({"source_key": src.key, "name": src.name, "source": src.source, "category": src.category,
                         "status": "not_evaluated",
                         "reason": "no result for this run yet" if run else "no run yet; start an evaluation"})
    for src in unavailable:
        rows.append({"source_key": src.key, "name": src.name, "source": src.source, "category": src.category,
                     "status": "unavailable", "reason": src.status_reason})
    failed = db.query("SELECT source_key, name, error FROM indicator_evaluations WHERE run_id=? AND status='failed'",
                      (run_id,)) if run_id else []
    for f in failed:
        for r in rows:
            if r["source_key"] == f["source_key"] and r["status"] in ("not_evaluated", "changed_since_run"):
                r["status"], r["reason"] = "failed", f["error"]
    return {"run": run, "dataset_id": dataset_id, "split": split, "rows": rows,
            "counts": {k: sum(1 for r in rows if r["status"] == k)
                       for k in ("evaluated", "failed", "not_evaluated", "changed_since_run", "unavailable")}}
