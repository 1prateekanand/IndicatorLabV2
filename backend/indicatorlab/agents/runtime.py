"""Shared runtime for background workers: lifecycle, pause/stop, heartbeats, logs, checkpoints.

Every status shown in the UI is written here from real execution. Nothing is simulated.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import psutil

from ..config import CPU_LIMIT_PCT, RAM_LIMIT_PCT
from ..db import Database, get_db, utcnow

log = logging.getLogger(__name__)


class StopRequested(Exception):
    """Raised inside a worker when a stop was requested. Caught by the runner."""


@dataclass
class WorkerState:
    name: str
    status: str = "idle"  # idle | running | paused | stopping | stopped | completed | error
    current_task: str = ""
    algorithm: str = ""
    trials_processed: int = 0
    error_count: int = 0
    progress: float = 0.0
    started_at: str | None = None
    last_heartbeat: str | None = None
    detail: dict = field(default_factory=dict)


class WorkerContext:
    """Passed to a worker's target function. All operations are safe to call from its thread."""

    def __init__(self, name: str, db: Database, *, throttle: bool = True):
        self.name = name
        self.db = db
        self.throttle = throttle
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.state = WorkerState(name=name)
        self._lock = threading.Lock()
        self._throttle_warned = False
        self.duplicates_prevented = 0

    # -- state -----------------------------------------------------------------------------
    def set_status(self, status: str) -> None:
        self.state.status = status
        self.heartbeat()

    def set_task(self, task: str, algorithm: str | None = None, progress: float | None = None) -> None:
        self.state.current_task = task
        if algorithm is not None:
            self.state.algorithm = algorithm
        if progress is not None:
            self.state.progress = round(float(progress), 4)
        self.heartbeat()

    def heartbeat(self) -> None:
        with self._lock:
            self.state.last_heartbeat = utcnow()
            self._persist()

    def count_trial(self, n: int = 1) -> None:
        with self._lock:
            self.state.trials_processed += n
        self.heartbeat()

    def count_error(self) -> None:
        with self._lock:
            self.state.error_count += 1
        self.heartbeat()

    def _persist(self) -> None:
        s = self.state
        self.db.execute(
            "INSERT INTO agents(name, status, current_task, algorithm, trials_processed, error_count,"
            " progress, started_at, last_heartbeat, detail_json) VALUES (?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(name) DO UPDATE SET status=excluded.status, current_task=excluded.current_task,"
            " algorithm=excluded.algorithm, trials_processed=excluded.trials_processed,"
            " error_count=excluded.error_count, progress=excluded.progress,"
            " started_at=COALESCE(agents.started_at, excluded.started_at),"
            " last_heartbeat=excluded.last_heartbeat, detail_json=excluded.detail_json",
            (s.name, s.status, s.current_task, s.algorithm, s.trials_processed, s.error_count,
             s.progress, s.started_at, s.last_heartbeat, self.db.dumps(s.detail)))

    # -- logs ------------------------------------------------------------------------------
    def log(self, message: str, level: str = "info") -> None:
        getattr(log, "error" if level == "error" else "info")("[%s] %s", self.name, message)
        self.db.execute("INSERT INTO agent_logs(agent, ts, level, message) VALUES (?,?,?,?)",
                        (self.name, utcnow(), level, message[:2000]))

    # -- checkpoints -----------------------------------------------------------------------
    def save_checkpoint(self, state: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO agent_checkpoints(agent, state_json, updated_at) VALUES (?,?,?)"
            " ON CONFLICT(agent) DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at",
            (self.name, self.db.dumps(state), utcnow()))

    def load_checkpoint(self) -> dict[str, Any] | None:
        row = self.db.one("SELECT state_json FROM agent_checkpoints WHERE agent=?", (self.name,))
        return json.loads(row["state_json"]) if row else None

    # -- control points --------------------------------------------------------------------
    def checkpoint(self) -> None:
        """Call between units of work. Blocks while paused, raises on stop, backs off under load."""
        while self.pause_event.is_set() and not self.stop_event.is_set():
            if self.state.status != "paused":
                self.set_status("paused")
                self.log("paused")
            time.sleep(0.5)
            self.heartbeat()
        if self.stop_event.is_set():
            raise StopRequested()
        if self.state.status == "paused":
            self.set_status("running")
            self.log("resumed after pause")
        if self.throttle:
            self._respect_resources()

    def _respect_resources(self) -> None:
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        if cpu > CPU_LIMIT_PCT or ram > RAM_LIMIT_PCT:
            if not self._throttle_warned:
                self.log(f"resource limit reached (cpu {cpu:.0f}%, ram {ram:.0f}%); backing off", "warn")
                self._throttle_warned = True
            time.sleep(2.0)
        else:
            self._throttle_warned = False

    def sleep(self, seconds: float) -> None:
        """Interruptible sleep that respects pause and stop."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.checkpoint()
            time.sleep(min(0.25, max(0.0, end - time.monotonic())))


class ManagedWorker:
    """Runs a target function in a daemon thread and tracks its lifecycle."""

    def __init__(self, name: str, target: Callable[[WorkerContext], None], *, throttle: bool = True):
        self.name = name
        self._target = target
        self._db = get_db()
        self.ctx = WorkerContext(name, self._db, throttle=throttle)
        self._thread: threading.Thread | None = None

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_alive:
            raise RuntimeError(f"{self.name} is already running")
        self.ctx.stop_event.clear()
        self.ctx.pause_event.clear()
        self.ctx.state.started_at = utcnow()
        self.ctx.state.status = "running"
        self.ctx.heartbeat()
        self._thread = threading.Thread(target=self._run, name=f"worker-{self.name}", daemon=True)
        self._thread.start()

    def pause(self) -> None:
        self.ctx.pause_event.set()

    def resume(self) -> None:
        self.ctx.pause_event.clear()

    def stop(self, wait: float = 0.0) -> None:
        self.ctx.stop_event.set()
        self.ctx.pause_event.clear()
        if wait and self._thread is not None:
            self._thread.join(timeout=wait)

    def _run(self) -> None:
        try:
            self._target(self.ctx)
            final = "completed"
        except StopRequested:
            final = "stopped"
        except Exception as exc:  # noqa: BLE001 - recorded, then the worker ends in error state
            self.ctx.count_error()
            self.ctx.log(f"fatal: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}", "error")
            final = "error"
            self.ctx.state.detail["last_error"] = f"{type(exc).__name__}: {exc}"
        self.ctx.state.status = final
        self.ctx.state.current_task = ""
        self.ctx.heartbeat()
        self.ctx.log(f"worker {final}")

    def snapshot(self) -> dict[str, Any]:
        s = self.ctx.state
        uptime = None
        if s.started_at and self.is_alive:
            from datetime import datetime, timezone

            started = datetime.fromisoformat(s.started_at)
            uptime = int((datetime.now(timezone.utc) - started).total_seconds())
        return {
            "name": self.name, "status": s.status, "current_task": s.current_task,
            "algorithm": s.algorithm, "trials_processed": s.trials_processed,
            "error_count": s.error_count, "progress": s.progress, "started_at": s.started_at,
            "last_heartbeat": s.last_heartbeat, "uptime_s": uptime, "alive": self.is_alive,
            "duplicates_prevented": self.ctx.duplicates_prevented, "detail": s.detail,
        }


def system_metrics() -> dict[str, float]:
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_percent": psutil.virtual_memory().percent,
        "ram_used_mb": round(psutil.virtual_memory().used / 1e6, 1),
        "cores": psutil.cpu_count() or 0,
    }
