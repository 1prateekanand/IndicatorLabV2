"""Dynamic indicator registry.

An indicator is a function ``compute(df, **params) -> pd.Series`` that returns a causal
signal per bar with values in {-1, 0, 1} (+1 bullish, -1 bearish, 0 neutral). Causal means
the value at bar *t* may only use bars ``<= t``. Indicators are discovered at runtime from:

* ``indicatorlab.plugins`` (built-in reference examples), and
* every ``*.py`` file in the plugin directory (``IL_INDICATOR_PLUGIN_DIR``).

Adding an indicator never requires editing the agents or the frontend: decorate a function
with ``@indicator(...)`` in a new file, then call ``POST /api/indicators/reload``.
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import re
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import PLUGIN_DIR

log = logging.getLogger(__name__)
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


@dataclass(frozen=True)
class ParamSpec:
    name: str
    default: float | int
    low: float | int
    high: float | int
    kind: str = "int"  # "int" or "float"
    step: float | None = None

    def __post_init__(self):
        if self.kind not in ("int", "float"):
            raise ValueError(f"param {self.name}: kind must be 'int' or 'float'")
        if not (self.low <= self.default <= self.high):
            raise ValueError(f"param {self.name}: default {self.default} outside [{self.low}, {self.high}]")

    def coerce(self, value: float) -> float | int:
        v = int(round(value)) if self.kind == "int" else round(float(value), 6)
        return min(max(v, self.low), self.high)

    def sample(self, rng: np.random.Generator) -> float | int:
        if self.kind == "int":
            return int(rng.integers(self.low, self.high + 1))
        return round(float(rng.uniform(self.low, self.high)), 6)

    def neighbours(self, value: float | int) -> list:
        step = self.step or (1 if self.kind == "int" else (self.high - self.low) / 20)
        out = []
        for candidate in (value - step, value + step):
            coerced = self.coerce(candidate)
            if coerced != value and coerced not in out:
                out.append(coerced)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "default": self.default, "low": self.low, "high": self.high,
                "kind": self.kind, "step": self.step}


@dataclass
class IndicatorDef:
    name: str
    category: str
    params: tuple[ParamSpec, ...]
    compute: Callable[..., pd.Series]
    description: str = ""
    module: str = ""
    available: bool = False
    status_reason: str = "not yet tested"
    outputs: tuple[str, ...] = ("signal",)
    _param_index: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._param_index = {p.name: p for p in self.params}

    @property
    def defaults(self) -> dict[str, float | int]:
        return {p.name: p.default for p in self.params}

    def param(self, name: str) -> ParamSpec:
        return self._param_index[name]

    def clean_params(self, params: dict | None) -> dict[str, float | int]:
        """Fill defaults, reject unknown names and clamp values to their declared ranges."""
        merged = dict(self.defaults)
        for key, value in (params or {}).items():
            if key not in self._param_index:
                raise ValueError(f"indicator '{self.name}' has no parameter '{key}'")
            merged[key] = self._param_index[key].coerce(value)
        return merged

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "params": [p.to_dict() for p in self.params],
            "defaults": self.defaults,
            "outputs": list(self.outputs),
            "available": self.available,
            "status_reason": self.status_reason,
            "module": self.module,
        }


class IndicatorRegistry:
    def __init__(self):
        self._items: dict[str, IndicatorDef] = {}
        self._lock = threading.RLock()
        self.load_errors: list[dict[str, str]] = []

    # -- registration -----------------------------------------------------------------
    def register(self, definition: IndicatorDef) -> None:
        if not _NAME_RE.match(definition.name):
            raise ValueError(f"invalid indicator name '{definition.name}'")
        with self._lock:
            existing = self._items.get(definition.name)
            if existing is not None and existing.module != definition.module:
                raise ValueError(
                    f"duplicate indicator '{definition.name}' in {definition.module} "
                    f"and {existing.module}")
            self._items[definition.name] = definition

    def get(self, name: str) -> IndicatorDef:
        with self._lock:
            if name not in self._items:
                raise KeyError(f"unknown indicator '{name}'")
            return self._items[name]

    def names(self, only_available: bool = False) -> list[str]:
        with self._lock:
            return sorted(n for n, d in self._items.items() if d.available or not only_available)

    def all(self) -> list[IndicatorDef]:
        with self._lock:
            return [self._items[n] for n in sorted(self._items)]

    def available(self) -> list[IndicatorDef]:
        return [d for d in self.all() if d.available]

    def categories(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.all():
            counts[d.category] = counts.get(d.category, 0) + 1
        return counts

    # -- discovery --------------------------------------------------------------------
    def discover(self, plugin_dir: Path = PLUGIN_DIR) -> dict[str, Any]:
        """Rebuild the registry from builtin modules and plugin files, then self-test each."""
        plugin_dir = Path(plugin_dir)
        with self._lock:
            self._items = {}
            self.load_errors = []
            _PENDING_SINK.clear()
            import indicatorlab.plugins as builtin_pkg

            for info in pkgutil.iter_modules(builtin_pkg.__path__):
                self._import_module(f"indicatorlab.plugins.{info.name}")
            if plugin_dir.is_dir():
                for path in sorted(plugin_dir.glob("*.py")):
                    if not path.name.startswith("_"):
                        self._import_file(path)
            return self.summary()

    def _drain(self, source: str) -> None:
        """Register definitions queued by decorators during one import, then self-test them."""
        while _PENDING_SINK:
            definition = _PENDING_SINK.pop(0)
            try:
                self.register(definition)
            except ValueError as exc:
                self._record_error(source, exc)
                continue
            self._self_test(definition)

    def _import_module(self, dotted: str) -> None:
        try:
            module = importlib.import_module(dotted)
            importlib.reload(module)  # re-run decorators so a reload picks up edits
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI, never fatal
            self._record_error(dotted, exc)
        self._drain(dotted)

    def _import_file(self, path: Path) -> None:
        import importlib.util

        mod_name = f"il_plugin_{path.stem}"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                raise ImportError("cannot create module spec")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            self._record_error(str(path), exc)
        self._drain(str(path))

    def _record_error(self, where: str, exc: BaseException) -> None:
        log.warning("indicator load failed for %s: %s", where, exc)
        self.load_errors.append({"source": where, "error": f"{type(exc).__name__}: {exc}",
                                 "trace": traceback.format_exc(limit=3)})

    def _self_test(self, definition: IndicatorDef) -> None:
        """Run the indicator on synthetic bars. Failure marks it unavailable with a reason."""
        probe = _probe_frame(400)
        try:
            out = definition.compute(probe, **definition.defaults)
            if not isinstance(out, pd.Series) or len(out) != len(probe):
                raise ValueError("compute() must return a pandas Series aligned to the input")
            values = pd.unique(out.dropna())
            if not set(np.asarray(values).tolist()) <= {-1, 0, 1}:
                raise ValueError("signal values must be within {-1, 0, 1}")
            definition.available, definition.status_reason = True, "ok"
        except Exception as exc:  # noqa: BLE001
            definition.available = False
            definition.status_reason = f"self-test failed: {type(exc).__name__}: {exc}"
            log.warning("indicator %s unavailable: %s", definition.name, definition.status_reason)
        with self._lock:
            self._items[definition.name] = definition

    def summary(self) -> dict[str, Any]:
        items = self.all()
        return {
            "total": len(items),
            "available": sum(1 for d in items if d.available),
            "unavailable": sum(1 for d in items if not d.available),
            "categories": self.categories(),
            "load_errors": list(self.load_errors),
        }


def _probe_frame(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0.1, 0.8, n)
    low = close - rng.uniform(0.1, 0.8, n)
    idx = pd.date_range("2024-01-01 09:15", periods=n, freq="min")
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                         "volume": rng.integers(1, 100, n)}, index=idx)


# Definitions made while a discovery pass is running are queued here, then registered.
_PENDING_SINK: list[IndicatorDef] = []
REGISTRY = IndicatorRegistry()


def indicator(name: str, *, category: str, params: tuple[ParamSpec, ...] = (),
              description: str = "", outputs: tuple[str, ...] = ("signal",)):
    """Decorator that registers a signal function. The function body must be causal."""

    def decorator(fn: Callable[..., pd.Series]):
        definition = IndicatorDef(name=name, category=category, params=tuple(params),
                                  compute=fn, description=description,
                                  module=fn.__module__, outputs=outputs)
        _PENDING_SINK.append(definition)
        return fn

    return decorator
