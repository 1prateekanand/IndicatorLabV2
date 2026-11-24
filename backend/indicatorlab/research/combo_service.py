"""Background combination runs, the one-shot OOS evaluation and the dashboard payload.

Runs use the same single-active gate and ManagedWorker lifecycle as individual evaluation
(pause / resume / stop / restart recovery). Results are written as they are produced.

OOS isolation: the out-of-sample tail (last 20% of the selected range) is never used by any selection
step. The best combination of a finished run is frozen. Its OOS result can be requested exactly once
per run; the database enforces this with a UNIQUE constraint. Indicators edited after the freeze are
refused, because the frozen result would describe different code.
"""
from __future__ import annotations

import json
import sqlite3

import pandas as pd

from ..agents.runtime import ManagedWorker, StopRequested, WorkerContext
from ..config import DEFAULT_SPLIT
from ..db import Database, get_db, utcnow
from . import service as individual_service
from .combination import (BudgetExceeded, ComboRequest, Evaluator, _curve, fold_plan, prepare_window, select)
from .combination import and_votes
from .individual import (Rules, dataset_frame, discover_sources, load_window, signal_votes, simulate, summarize)

WORKER_PREFIX = "research_combination"


def _request_dict(req: ComboRequest) -> dict:
    return {"dataset_id": req.dataset_id, "timeframe": req.timeframe, "start": req.start, "end": req.end,
            "rules": req.rules.to_dict(), "segments": req.segments, "pool_size": req.pool_size,
            "corr_threshold": req.corr_threshold, "purge_bars": req.purge_bars,
            "max_evaluations": req.max_evaluations, "top_n": req.top_n}


def request_from_dict(d: dict) -> ComboRequest:
    rules = Rules(**d["rules"])
    return ComboRequest(dataset_id=d["dataset_id"], timeframe=d["timeframe"], start=d.get("start"),
                        end=d.get("end"), rules=rules, segments=d["segments"], pool_size=d["pool_size"],
                        corr_threshold=d["corr_threshold"], purge_bars=d["purge_bars"],
                        max_evaluations=d["max_evaluations"], top_n=d["top_n"])


def start_combination(req: ComboRequest, db: Database | None = None) -> dict:
    db = db or get_db()
    _win, cfg = prepare_window(req, db)  # validates ranges, rules and the date span
    with individual_service._LOCK:
        if any(w.is_alive for w in individual_service._ACTIVE.values()):
            raise RuntimeError("a research run is already active; stop it or wait for it to finish")
        with db.transaction() as cur:
            cur.execute("INSERT INTO research_runs(created_at, kind, dataset_id, split, config_hash, config_json,"
                        " status, message) VALUES (?,?,?,?,?,?,?,?)",
                        (utcnow(), "combination", req.dataset_id, "train", cfg,
                         json.dumps(_request_dict(req), default=str), "running", "starting"))
            run_id = int(cur.lastrowid)
        worker = ManagedWorker(f"{WORKER_PREFIX}_{run_id}", lambda ctx: _execute(ctx, run_id, req, cfg, db),
                               throttle=False)
        individual_service._ACTIVE[run_id] = worker
        worker.start()
    return individual_service.run_status(run_id, db)


def _artifact(db: Database, run_id: int, kind: str, payload) -> None:
    db.execute("INSERT OR REPLACE INTO run_artifacts(run_id, kind, payload_json) VALUES (?,?,?)",
               (run_id, kind, json.dumps(payload, default=str)))


def _execute(ctx: WorkerContext, run_id: int, req: ComboRequest, cfg: str, db: Database) -> None:
    status, error = "completed", None
    try:
        win = load_window(req.dataset_id, req.timeframe, "train", req.start, req.end, db)
        sources, unavailable = discover_sources(db)
        by_key = {s.key: s for s in sources}
        ev = Evaluator(db, win, req.rules, cfg, req.max_evaluations, run_id, by_key=by_key)
        plan = fold_plan(win.eval_start, win.eval_end, req.segments, req.purge_bars)
        total_steps = len(plan) + 1
        ctx.set_task(f"Walk-forward: {len(sources)} indicators, {len(plan)} folds on {win.dataset_name}",
                     algorithm="SFFS + Beta-Binomial/Wilson + Spearman filter", progress=0.0)
        for step, fp in enumerate(plan, start=1):
            ctx.checkpoint()
            f = fp["fold"]
            fold_status, fold_msg, sel_m, test_m, members = "completed", None, None, None, None
            try:
                sel = select(ev, sources, *fp["sel"], tag=f"fold{f}_select", corr_threshold=req.corr_threshold,
                             pool_size=req.pool_size, top_n=req.top_n, checkpoint=ctx.checkpoint)
                best = sel["best"]
                if best is None:
                    fold_status, fold_msg = "no_eligible_combination", "no combination reached the minimum trades"
                else:
                    members = best["members"]
                    sel_m = best["metrics"]
                    test_m = ev.evaluate([by_key[m["key"]] for m in members], f"fold{f}_test", *fp["test"])
                    ctx.count_trial()
            except BudgetExceeded as exc:
                fold_status, fold_msg = "budget_reached", str(exc)
            db.execute(
                "INSERT OR REPLACE INTO walkforward_folds(created_at, run_id, fold, selection_range_json,"
                " test_range_json, purge_bars, selected_members_json, selection_metrics_json, test_metrics_json,"
                " status, message) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (utcnow(), run_id, f, json.dumps(_span(win, fp["sel"])), json.dumps(_span(win, fp["test"])),
                 req.purge_bars, json.dumps(members), json.dumps(sel_m), json.dumps(test_m), fold_status, fold_msg))
            ctx.set_task(f"Walk-forward fold {f}/{len(plan)} ({fold_status})", progress=step / total_steps)
            db.execute("UPDATE research_runs SET progress=?, message=? WHERE id=?",
                       (step / total_steps, f"fold {f}/{len(plan)}", run_id))
            if fold_status == "budget_reached":
                ctx.log(fold_msg, "error")
                break

        ctx.set_task("Final selection on the full train span", progress=len(plan) / total_steps)
        final = select(ev, sources, win.eval_start, win.eval_end, tag="full_select",
                       corr_threshold=req.corr_threshold, pool_size=req.pool_size, top_n=req.top_n,
                       checkpoint=ctx.checkpoint)
        for row in final["ranked"]:
            db.execute("INSERT OR REPLACE INTO combination_rankings(run_id, rank, combo_key, size, members_json,"
                       " metrics_json) VALUES (?,?,?,?,?,?)",
                       (run_id, row["rank"], row["combo_key"], row["size"], json.dumps(row["members"]),
                        json.dumps(row["metrics"], default=str)))
        _artifact(db, run_id, "selection", {"coverage": final["coverage"], "dropped": final["dropped"],
                                            "singles": final["singles"], "unavailable": [
                                                {"key": u.key, "name": u.name, "reason": u.status_reason}
                                                for u in unavailable]})
        _artifact(db, run_id, "correlation", {"keys": final["corr_keys"], "names": [p["name"] for p in final["pool"]],
                                              "matrix": final["corr"], "threshold": req.corr_threshold})
        best = final["best"]
        if best is not None:
            frozen = {"combo_key": best["combo_key"], "members": best["members"], "size": best["size"],
                      "train_metrics": best["metrics"], "rules": req.rules.to_dict(), "frozen_at": utcnow()}
            _artifact(db, run_id, "frozen", frozen)
            members = [by_key[m["key"]] for m in best["members"]]
            curve = _train_curve(ev, members, win)
            _artifact(db, run_id, "train_curve", curve)
        else:
            _artifact(db, run_id, "frozen", None)
        ctx.log(f"combination run finished: {final['coverage']['combinations_ranked']} ranked combinations, "
                f"{ev.fresh} fresh backtests, {ev.cache_hits} cache hits")
    except StopRequested:
        status = "stopped"
    except Exception as exc:  # noqa: BLE001 - recorded on the run row
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        ctx.log(f"combination run failed: {error}", "error")
    finally:
        db.execute("UPDATE research_runs SET status=?, error=?, finished_at=?, message=? WHERE id=?",
                   (status, error, utcnow(), status, run_id))
        with individual_service._LOCK:
            individual_service._ACTIVE.pop(run_id, None)


def _span(win, rng) -> dict:
    a, b = rng
    return {"start": win.full.index[a].isoformat(), "end": win.full.index[b - 1].isoformat(), "bars": b - a}


def _train_curve(ev: Evaluator, members, win) -> dict:
    votes = ev.votes(members[0]) if len(members) == 1 else and_votes([ev.votes(m) for m in members])
    res, _cfg, df = simulate(votes, win.full, win.eval_start, win.eval_end, ev.rules)
    return _curve(res, df)


# -- one-shot OOS ---------------------------------------------------------------------------------
def run_oos(run_id: int, db: Database | None = None) -> dict:
    db = db or get_db()
    run = db.one("SELECT * FROM research_runs WHERE id=?", (run_id,))
    if run is None or run["kind"] != "combination":
        raise LookupError(f"combination run {run_id} not found")
    if run["status"] != "completed":
        raise ValueError("OOS can only be evaluated for a completed combination run")
    done = db.one("SELECT created_at FROM oos_results WHERE run_id=?", (run_id,))
    if done:
        raise ValueError(f"OOS was already evaluated for run {run_id} at {done['created_at']} (one-shot)")
    art = db.one("SELECT payload_json FROM run_artifacts WHERE run_id=? AND kind='frozen'", (run_id,))
    frozen = json.loads(art["payload_json"]) if art else None
    if not frozen:
        raise ValueError("this run has no frozen combination (no eligible combination was found)")
    req = request_from_dict(json.loads(run["config_json"]))
    frame = dataset_frame(req.dataset_id, req.timeframe, db)
    n = len(frame)
    lo, hi = 0, n
    if req.start:
        lo = int(frame.index.searchsorted(pd.Timestamp(req.start)))
    if req.end:
        hi = int(frame.index.searchsorted(pd.Timestamp(req.end), side="right"))
    span = hi - lo
    if span < 50:
        raise ValueError("the selected range holds fewer than 50 candles")
    a = lo + int(span * DEFAULT_SPLIT.validation_end)
    b = hi
    if b - a < 50:
        raise ValueError("the OOS tail holds fewer than 50 candles")
    sources, _ = discover_sources(db)
    by_key = {s.key: s for s in sources}
    members = []
    for m in frozen["members"]:
        src = by_key.get(m["key"])
        if src is None or not src.available:
            raise ValueError(f"{m['name']} is no longer registered; the frozen combination cannot be tested")
        if src.code_hash != m["code_hash"]:
            raise ValueError(f"{m['name']} changed after the freeze; start a new run before OOS testing")
        members.append(src)
    prefix = frame.iloc[:hi]
    votes_list = [signal_votes(m, prefix, by_key) for m in members]
    votes = votes_list[0] if len(votes_list) == 1 else and_votes(votes_list)
    rules = req.rules
    rules.validate()
    res, cfg, df = simulate(votes, frame, a, b, rules)
    metrics = summarize(res, cfg, len(df), df.index[0].isoformat(), df.index[-1].isoformat())
    curve = _curve(res, df)
    window = {"start": df.index[0].isoformat(), "end": df.index[-1].isoformat(), "bars": len(df),
              "after": "validation_end (last 20% of the selected range)"}
    try:
        with db.transaction() as cur:
            cur.execute("INSERT INTO oos_results(created_at, run_id, combo_key, members_json, window_json,"
                        " metrics_json, curve_json) VALUES (?,?,?,?,?,?,?)",
                        (utcnow(), run_id, frozen["combo_key"], json.dumps(frozen["members"]), json.dumps(window),
                         json.dumps(metrics, default=str), json.dumps(curve)))
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"OOS was already evaluated for run {run_id} (one-shot)") from exc
    return {"run_id": run_id, "combo_key": frozen["combo_key"], "members": frozen["members"], "window": window,
            "metrics": metrics, "curve": curve}


def oos_for_run(run_id: int, db: Database | None = None) -> dict | None:
    db = db or get_db()
    row = db.one("SELECT * FROM oos_results WHERE run_id=?", (run_id,))
    if row is None:
        return None
    return {"run_id": run_id, "created_at": row["created_at"], "combo_key": row["combo_key"],
            "members": json.loads(row["members_json"]), "window": json.loads(row["window_json"]),
            "metrics": json.loads(row["metrics_json"]), "curve": json.loads(row["curve_json"])}


# -- read side ------------------------------------------------------------------------------------
def _artifacts(db: Database, run_id: int) -> dict:
    return {r["kind"]: json.loads(r["payload_json"])
            for r in db.query("SELECT kind, payload_json FROM run_artifacts WHERE run_id=?", (run_id,))}


def combination_payload(run_id: int, db: Database | None = None) -> dict:
    db = db or get_db()
    run = individual_service.run_status(run_id, db)
    if run["kind"] != "combination":
        raise LookupError(f"run {run_id} is not a combination run")
    rankings = [{"rank": r["rank"], "combo_key": r["combo_key"], "size": r["size"],
                 "members": json.loads(r["members_json"]), "metrics": json.loads(r["metrics_json"])}
                for r in db.query("SELECT * FROM combination_rankings WHERE run_id=? ORDER BY rank", (run_id,))]
    folds = [{"fold": r["fold"], "status": r["status"], "message": r["message"],
              "selection_range": json.loads(r["selection_range_json"]), "test_range": json.loads(r["test_range_json"]),
              "purge_bars": r["purge_bars"], "selected_members": json.loads(r["selected_members_json"] or "null"),
              "selection_metrics": json.loads(r["selection_metrics_json"] or "null"),
              "test_metrics": json.loads(r["test_metrics_json"] or "null")}
             for r in db.query("SELECT * FROM walkforward_folds WHERE run_id=? ORDER BY fold", (run_id,))]
    art = _artifacts(db, run_id)
    return {"run": run, "rankings": rankings, "folds": folds, "selection": art.get("selection"),
            "correlation": art.get("correlation"), "frozen": art.get("frozen"),
            "train_curve": art.get("train_curve"), "oos": oos_for_run(run_id, db)}


def latest_combination_run(dataset_id: int, db: Database | None = None) -> int | None:
    db = db or get_db()
    row = db.one("SELECT id FROM research_runs WHERE kind='combination' AND dataset_id=? ORDER BY id DESC LIMIT 1",
                 (dataset_id,))
    return row["id"] if row else None


def dashboard_payload(dataset_id: int, db: Database | None = None) -> dict:
    """Everything the research dashboard shows, read from stored runs. Empty sections are null."""
    db = db or get_db()
    ind = individual_service.individual_table(dataset_id, "train", None, db)
    ind_rows = sorted((r for r in ind["rows"] if r["status"] == "evaluated"),
                      key=lambda r: (r["wilson_lb"], r["shrunk_score"], r["net_profit"]), reverse=True)
    run_id = latest_combination_run(dataset_id, db)
    comb = combination_payload(run_id, db) if run_id else None
    return {"dataset_id": dataset_id,
            "individual": {"run": ind["run"], "counts": ind["counts"], "ranking": ind_rows,
                           "rows": ind["rows"]},
            "combination": comb}
