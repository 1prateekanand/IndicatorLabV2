"""Lifecycle control for the two agents and Optuna optimization runs.

State shown to the UI is read from the database and from live worker objects. A worker that
is not alive is reported as stopped, never as running.
"""
from __future__ import annotations

import threading
from ..config import MIN_TRADES
from ..db import get_db, utcnow
from ..evaluator import ResearchConfig
from ..optimizer import OptimizationConfig, run_study, study_summary
from .evaluation import AGENT_NAME as EVAL_NAME, evaluation_loop
from .research import AGENT_NAME as RESEARCH_NAME, ResearchAgent, ResearchSettings
from .runtime import ManagedWorker

SETTINGS_KEY = "research_agent:settings"


class AgentController:
    def __init__(self):
        self._lock = threading.RLock()
        self._research: ManagedWorker | None = None
        self._evaluation: ManagedWorker | None = None
        self._optimizations: dict[str, ManagedWorker] = {}
        self.db = get_db()

    # -- startup recovery -------------------------------------------------------------------
    def recover_after_restart(self) -> list[str]:
        """Agents left 'running' or 'paused' by a previous process are marked stopped, not resumed."""
        recovered = []
        rows = self.db.query("SELECT name, status FROM agents WHERE status IN ('running','paused','stopping')")
        for row in rows:
            self.db.execute("UPDATE agents SET status='stopped', current_task='' WHERE name=?", (row["name"],))
            self.db.execute("INSERT INTO agent_logs(agent, ts, level, message) VALUES (?,?,?,?)",
                            (row["name"], utcnow(), "warn",
                             "process restarted while this worker was active; marked stopped (checkpoint kept)"))
            recovered.append(row["name"])
        self.db.execute("UPDATE optimization_runs SET status='interrupted' WHERE status='running'")
        return recovered

    def saved_settings(self) -> ResearchSettings | None:
        row = self.db.one("SELECT state_json FROM agent_checkpoints WHERE agent=?", (SETTINGS_KEY,))
        if not row:
            return None
        return ResearchSettings(**self.db.loads(row["state_json"]))

    def _save_settings(self, settings: ResearchSettings) -> None:
        self.db.execute(
            "INSERT INTO agent_checkpoints(agent, state_json, updated_at) VALUES (?,?,?)"
            " ON CONFLICT(agent) DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at",
            (SETTINGS_KEY, self.db.dumps(settings.to_dict()), utcnow()))

    # -- research agent + evaluation agent --------------------------------------------------
    def start_agents(self, settings: ResearchSettings, research: ResearchConfig | None = None) -> dict:
        settings.validate()
        research = research or ResearchConfig(min_trades=MIN_TRADES)
        with self._lock:
            if self._research and self._research.is_alive:
                raise RuntimeError("Strategy Research Agent is already running")
            self._save_settings(settings)
            self._evaluation = ManagedWorker(
                EVAL_NAME, lambda ctx: evaluation_loop(ctx, settings.dataset_id, research.min_trades))
            self._evaluation.start()
            self._research = ManagedWorker(
                RESEARCH_NAME, lambda ctx: ResearchAgent(settings, ctx, research).run(), throttle=True)
            self._research.start()
        return self.agents_status()

    def control(self, agent: str, action: str) -> dict:
        if action not in ("pause", "resume", "stop"):
            raise ValueError("action must be pause, resume or stop")
        with self._lock:
            worker = {RESEARCH_NAME: self._research, EVAL_NAME: self._evaluation}.get(agent)
            if worker is None or not worker.is_alive:
                raise RuntimeError(f"{agent} is not running")
            if action == "pause":
                worker.pause()
            elif action == "resume":
                worker.resume()
            else:
                worker.stop()
        return self.agents_status()

    def stop_all(self, wait: float = 5.0) -> None:
        for w in (self._research, self._evaluation, *self._optimizations.values()):
            if w is not None:
                w.stop(wait=wait)

    def agents_status(self) -> dict:
        out = []
        for name, worker in ((RESEARCH_NAME, self._research), (EVAL_NAME, self._evaluation)):
            if worker is not None:
                out.append(worker.snapshot())
            else:
                row = self.db.one("SELECT * FROM agents WHERE name=?", (name,))
                out.append(_row_snapshot(name, row))
        settings = self.saved_settings()
        return {"agents": out, "settings": settings.to_dict() if settings else None,
                "checkpoint": {n: self.db.one("SELECT updated_at FROM agent_checkpoints WHERE agent=?", (n,))
                               for n in (RESEARCH_NAME, EVAL_NAME)}}

    def evaluation_snapshot(self) -> dict | None:
        row = self.db.one("SELECT state_json FROM agent_checkpoints WHERE agent=?", (EVAL_NAME,))
        return self.db.loads(row["state_json"]) if row else None

    # -- optimization runs --------------------------------------------------------------------
    def start_optimization(self, cfg: OptimizationConfig, research: ResearchConfig | None = None) -> dict:
        cfg.validate()
        name = f"optuna:{cfg.study_name}"
        with self._lock:
            if any(w.is_alive for w in self._optimizations.values()):
                raise RuntimeError("an optimization run is already active; stop it first")
            worker = ManagedWorker(name, lambda ctx: run_study(cfg, ctx, research), throttle=True)
            self._optimizations[name] = worker
            worker.start()
        return worker.snapshot()

    def control_optimization(self, name: str, action: str) -> dict:
        with self._lock:
            worker = self._optimizations.get(name)
            if worker is None or not worker.is_alive:
                raise RuntimeError(f"{name} is not running")
            {"pause": worker.pause, "resume": worker.resume, "stop": worker.stop}[action]()
        return worker.snapshot()

    def optimization_runs(self) -> list[dict]:
        runs = self.db.query("SELECT * FROM optimization_runs ORDER BY created_at DESC LIMIT 50")
        for r in runs:
            r["config"] = self.db.loads(r.pop("config_json"))
            worker = self._optimizations.get(f"optuna:{r['study_name']}")
            r["live"] = worker.snapshot() if worker else None
        return runs

    def optimization_study(self, study_name: str) -> dict:
        import optuna

        from ..config import OPTUNA_STORAGE

        study = optuna.load_study(study_name=study_name, storage=OPTUNA_STORAGE)
        return study_summary(study)


def _row_snapshot(name: str, row: dict | None) -> dict:
    if row is None:
        return {"name": name, "status": "stopped", "current_task": "", "algorithm": "",
                "trials_processed": 0, "error_count": 0, "progress": 0.0, "started_at": None,
                "last_heartbeat": None, "uptime_s": None, "alive": False,
                "duplicates_prevented": 0, "detail": {}}
    status = row["status"] if row["status"] in ("completed", "error", "stopped") else "stopped"
    return {"name": name, "status": status, "current_task": row["current_task"] or "",
            "algorithm": row["algorithm"] or "", "trials_processed": row["trials_processed"],
            "error_count": row["error_count"], "progress": row["progress"],
            "started_at": row["started_at"], "last_heartbeat": row["last_heartbeat"],
            "uptime_s": None, "alive": False, "duplicates_prevented": 0,
            "detail": (get_db().loads(row["detail_json"]) or {})}

