"""Background robustness run for a finished combination run.

Tests, on the train span of that run (never on OOS data):
* Hansen SPA: do the top candidates beat buy-and-hold?
* PBO (CSCV) across the top candidates.
* Deflated Sharpe of the frozen combination against the candidate trials.
* Permutation test: circular time-shifts of the frozen signals (same signal count, different timing).
* Monte Carlo over the frozen combination's trades.

Refused when a candidate's code or rule set changed after the combination run, because the results
would describe code that no longer exists.
"""
from __future__ import annotations

import json
import traceback

import numpy as np

from ..agents.runtime import ManagedWorker, StopRequested, WorkerContext
from ..db import Database, get_db, utcnow
from ..overfitting import deflated_sharpe, pbo_cscv, sharpe_moments
from . import service as individual_service
from .combination import and_votes
from .individual import discover_sources, load_window, signal_votes, simulate
from .robustness import bar_returns, hansen_spa, monte_carlo, permutation_p_value
from .combo_service import request_from_dict

WORKER_PREFIX = "research_robustness"
LIMITS = {"candidates": (2, 30), "permutations": (20, 1000), "mc_iterations": (500, 20000), "spa_draws": (100, 2000)}


def start_robustness(combination_run_id: int, *, candidates: int = 10, permutations: int = 200,
                     mc_iterations: int = 5000, spa_draws: int = 500, db: Database | None = None) -> dict:
    db = db or get_db()
    for name, (lo, hi) in LIMITS.items():
        val = {"candidates": candidates, "permutations": permutations, "mc_iterations": mc_iterations,
               "spa_draws": spa_draws}[name]
        if not lo <= int(val) <= hi:
            raise ValueError(f"{name} must be between {lo} and {hi}")
    parent = db.one("SELECT * FROM research_runs WHERE id=?", (combination_run_id,))
    if parent is None or parent["kind"] != "combination":
        raise LookupError(f"combination run {combination_run_id} not found")
    if parent["status"] != "completed":
        raise ValueError("robustness tests need a completed combination run")
    cfg = {"combination_run_id": combination_run_id, "candidates": candidates, "permutations": permutations,
           "mc_iterations": mc_iterations, "spa_draws": spa_draws}
    cfg_hash = json.dumps(cfg, sort_keys=True)
    with individual_service._LOCK:
        if any(w.is_alive for w in individual_service._ACTIVE.values()):
            raise RuntimeError("a research run is already active; stop it or wait for it to finish")
        with db.transaction() as cur:
            cur.execute("INSERT INTO research_runs(created_at, kind, dataset_id, split, config_hash, config_json,"
                        " status, message) VALUES (?,?,?,?,?,?,?,?)",
                        (utcnow(), "robustness", parent["dataset_id"], "train", cfg_hash, json.dumps(cfg),
                         "running", "starting"))
            run_id = int(cur.lastrowid)
        worker = ManagedWorker(f"{WORKER_PREFIX}_{run_id}",
                               lambda ctx: _execute(ctx, run_id, combination_run_id, cfg, db), throttle=False)
        individual_service._ACTIVE[run_id] = worker
        worker.start()
    return individual_service.run_status(run_id, db)


def _progress(db: Database, run_id: int, frac: float, message: str) -> None:
    db.execute("UPDATE research_runs SET progress=?, message=? WHERE id=?", (round(frac, 4), message, run_id))


def _execute(ctx: WorkerContext, run_id: int, parent_id: int, cfg: dict, db: Database) -> None:
    status, error = "completed", None
    try:
        parent = db.one("SELECT config_json, dataset_id FROM research_runs WHERE id=?", (parent_id,))
        req = request_from_dict(json.loads(parent["config_json"]))
        win = load_window(req.dataset_id, req.timeframe, "train", req.start, req.end, db)
        sources, _ = discover_sources(db)
        by_key = {s.key: s for s in sources}
        ranked = db.query("SELECT rank, members_json, metrics_json FROM combination_rankings WHERE run_id=?"
                          " ORDER BY rank LIMIT ?", (parent_id, cfg["candidates"]))
        if len(ranked) < 2:
            raise ValueError("the combination run ranked fewer than two combinations")
        frozen_row = db.one("SELECT payload_json FROM run_artifacts WHERE run_id=? AND kind='frozen'", (parent_id,))
        frozen = json.loads(frozen_row["payload_json"]) if frozen_row else None
        if not frozen:
            raise ValueError("the combination run froze no combination")
        prefix = win.full.iloc[:win.eval_end]
        a, b = win.eval_start, win.eval_end

        def votes_of(members_meta: list[dict]) -> np.ndarray:
            """Full-length votes on the prefix, so the simulator indexes them exactly as in the search."""
            members = []
            for m in members_meta:
                src = by_key.get(m["key"])
                if src is None or not src.available:
                    raise ValueError(f"{m['name']} is no longer registered; re-run the combination search")
                if src.code_hash != m["code_hash"]:
                    raise ValueError(f"{m['name']} changed after the combination run; re-run the combination search")
                members.append(src)
            per = [signal_votes(m, prefix, by_key) for m in members]
            return per[0] if len(per) == 1 else and_votes(per)

        cols, labels = [], []
        for i, row in enumerate(ranked, start=1):
            ctx.checkpoint()
            members = json.loads(row["members_json"])
            res, _cfg, _df = simulate(votes_of(members), win.full, a, b, req.rules)
            cols.append(bar_returns(res.equity))
            labels.append(" + ".join(m["name"] for m in members))
            _progress(db, run_id, 0.5 * i / len(ranked), f"candidate returns {i}/{len(ranked)}")
        T = min(len(c) for c in cols)
        R = np.column_stack([c[:T] for c in cols])
        close = win.full["close"].to_numpy(dtype=float)[a:b]
        bench = (np.diff(close) / close[:-1])[:T]

        spa = hansen_spa(R, bench, B=cfg["spa_draws"], seed=11)
        _progress(db, run_id, 0.55, "Hansen SPA done")
        pbo = pbo_cscv(R, blocks=8, labels=labels) if R.shape[1] >= 2 else \
            {"status": "insufficient_sample", "reason": "needs two candidates"}

        fv = votes_of(frozen["members"])
        res_frozen, cfg_bt, _ = simulate(fv, win.full, a, b, req.rules)
        r_frozen = bar_returns(res_frozen.equity)
        sr_frozen = sharpe_moments(r_frozen)
        trial_sr = [x for x in (sharpe_moments(c).get("sharpe_per_bar") for c in cols) if x is not None]
        if sr_frozen.get("sharpe_per_bar") is not None and len(trial_sr) >= 2:
            dsr = deflated_sharpe(sr_frozen["sharpe_per_bar"], len(r_frozen), sr_frozen["return_skew"] or 0.0,
                                  sr_frozen["return_kurtosis"] or 3.0, trial_sr)
        else:
            dsr = {"status": "insufficient_sample", "reason": "no Sharpe ratio, or fewer than two trials"}
        dsr = {**dsr, "trial_set": f"the {len(cols)} top candidates re-evaluated here, not the full search"}
        _progress(db, run_id, 0.6, "DSR done")

        observed = res_frozen.final_capital - cfg_bt.initial_capital
        rng = np.random.default_rng(23)
        window_votes = fv[a:b]
        null = np.empty(cfg["permutations"])
        for k in range(cfg["permutations"]):
            ctx.checkpoint()
            shift = int(rng.integers(max(1, len(window_votes) // 10), len(window_votes) - len(window_votes) // 10))
            shifted = fv.copy()
            shifted[a:b] = np.roll(window_votes, shift)
            rs, cb, _ = simulate(shifted, win.full, a, b, req.rules)
            null[k] = rs.final_capital - cb.initial_capital
            if (k + 1) % 10 == 0 or k + 1 == cfg["permutations"]:
                _progress(db, run_id, 0.6 + 0.35 * (k + 1) / cfg["permutations"],
                          f"permutation {k + 1}/{cfg['permutations']}")
        perm = permutation_p_value(observed, null)

        pnls = np.array([t["net_pnl"] for t in res_frozen.trades], dtype=float)
        mc = monte_carlo(pnls, capital=cfg_bt.initial_capital, iterations=cfg["mc_iterations"], seed=31) \
            if len(pnls) >= 5 else {"status": "insufficient_sample", "reason": "needs at least 5 trades"}

        dataset = db.one("SELECT name, is_synthetic FROM datasets WHERE id=?", (req.dataset_id,))
        synthetic = bool(dataset["is_synthetic"])
        payload = {
            "dataset": {"id": req.dataset_id, "name": dataset["name"], "is_synthetic": synthetic,
                        "label": "SYNTHETIC DATA: not market performance" if synthetic else "market data"},
            "window": {"start": win.full.index[a].isoformat(), "end": win.full.index[b - 1].isoformat(),
                       "bars": int(b - a), "span": "train split (no OOS data used)"},
            "candidates": [{"rank": r["rank"], "members": labels[i]} for i, r in enumerate(ranked)],
            "spa": {**spa, "benchmark": "buy and hold on the same bars"},
            "pbo": pbo, "dsr": dsr,
            "frozen_sharpe_per_bar": sr_frozen.get("sharpe_per_bar"),
            "permutation": perm, "monte_carlo": mc,
            "frozen_train": {"trades": int(len(pnls)), "net_profit": round(float(observed), 2),
                             "members": [m["name"] for m in frozen["members"]]},
            "generated_at": utcnow(),
        }
        db.execute("INSERT OR REPLACE INTO run_artifacts(run_id, kind, payload_json) VALUES (?,?,?)",
                   (run_id, "robustness", json.dumps(payload, default=str)))
        ctx.log("robustness tests finished")
    except StopRequested:
        status = "stopped"
    except Exception as exc:  # noqa: BLE001 - recorded on the run row
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        ctx.log(traceback.format_exc()[-1500:], "error")
    finally:
        db.execute("UPDATE research_runs SET status=?, error=?, finished_at=?, message=?, progress=? WHERE id=?",
                   (status, error, utcnow(), status, 1.0 if status == "completed" else 0.0, run_id))
        with individual_service._LOCK:
            individual_service._ACTIVE.pop(run_id, None)


def latest_robustness(combination_run_id: int, db: Database | None = None) -> dict | None:
    db = db or get_db()
    row = db.one("SELECT id, status, error, finished_at FROM research_runs WHERE kind='robustness'"
                 " AND json_extract(config_json, '$.combination_run_id')=? ORDER BY id DESC LIMIT 1",
                 (combination_run_id,))
    if row is None:
        return None
    art = db.one("SELECT payload_json FROM run_artifacts WHERE run_id=? AND kind='robustness'", (row["id"],))
    return {"run_id": row["id"], "status": row["status"], "error": row["error"], "finished_at": row["finished_at"],
            "result": json.loads(art["payload_json"]) if art else None}
