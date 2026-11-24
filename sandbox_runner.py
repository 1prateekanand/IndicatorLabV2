"""Isolated Python 3.11 sandbox for validating and executing user-written indicators and strategies.

Runs in `python3 -I` (isolated mode) as a separate child process with CPU & memory limits.
Provides a lightweight pure-Python pandas/numpy-compatible API (`pd`, `np`) so standard
vectorized indicator and strategy expressions work without external dependencies, while
also supporting pure Python loops over `df` / `candles`.
"""
from __future__ import annotations

import ast
import builtins
import datetime as _dt
import json
import math
import statistics
import sys
import time
import traceback

try:
    import resource
except ImportError:
    resource = None


BANNED_MODULES = {
    "subprocess", "socket", "urllib", "http", "ftplib", "poplib", "imaplib",
    "smtplib", "telnetlib", "ctypes", "multiprocessing", "threading", "shutil",
    "pathlib", "glob", "tempfile", "signal", "pty", "fcntl", "os", "sys",
}

BANNED_CALLS = {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint"}


def apply_resource_limits(cpu_seconds: int = 5, mem_mb: int = 256) -> None:
    if resource is None:
        return
    try:
        if cpu_seconds and cpu_seconds > 0:
            resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_seconds), int(cpu_seconds)))
    except Exception:
        pass
    try:
        if mem_mb and mem_mb > 0:
            limit = int(mem_mb) * 1024 * 1024
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            target = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
            resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    except Exception:
        pass


def check_ast_safety(code: str) -> tuple[bool, dict | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, {
            "type": "SyntaxError",
            "line": exc.lineno or 1,
            "column": exc.offset or 1,
            "message": exc.msg or str(exc),
            "text": (exc.text or "").rstrip("\n"),
        }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod in BANNED_MODULES:
                    return False, {
                        "type": "SecurityError",
                        "line": getattr(node, "lineno", 1),
                        "column": getattr(node, "col_offset", 0) + 1,
                        "message": f"Import of '{alias.name}' is blocked inside the isolated sandbox.",
                    }
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            if mod in BANNED_MODULES:
                return False, {
                    "type": "SecurityError",
                    "line": getattr(node, "lineno", 1),
                    "column": getattr(node, "col_offset", 0) + 1,
                    "message": f"Import from '{node.module}' is blocked inside the isolated sandbox.",
                }
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALLS:
                return False, {
                    "type": "SecurityError",
                    "line": getattr(node, "lineno", 1),
                    "column": getattr(node, "col_offset", 0) + 1,
                    "message": f"Call to '{node.func.id}()' is blocked inside the isolated sandbox.",
                }
    return True, None


# ---------------------------------------------------------------------------
# Lightweight pure-Python pandas/numpy compatibility layer
# ---------------------------------------------------------------------------
def _is_nan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _to_float(v) -> float:
    if v is None:
        return math.nan
    if isinstance(v, bool):
        return 1.0 if v else 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


class _Rolling:
    def __init__(self, data: list, window: int):
        self._data = data
        self._w = max(1, int(window))

    def mean(self) -> "Series":
        out = []
        w = self._w
        d = self._data
        for i in range(len(d)):
            if i + 1 < w:
                out.append(math.nan)
            else:
                chunk = [_to_float(x) for x in d[i - w + 1: i + 1]]
                if any(math.isnan(x) for x in chunk):
                    out.append(math.nan)
                else:
                    out.append(sum(chunk) / w)
        return Series(out, index=None)

    def sum(self) -> "Series":
        out = []
        w = self._w
        d = self._data
        for i in range(len(d)):
            if i + 1 < w:
                out.append(math.nan)
            else:
                chunk = [_to_float(x) for x in d[i - w + 1: i + 1]]
                out.append(math.nan if any(math.isnan(x) for x in chunk) else sum(chunk))
        return Series(out)

    def min(self) -> "Series":
        out = []
        w = self._w
        d = self._data
        for i in range(len(d)):
            if i + 1 < w:
                out.append(math.nan)
            else:
                chunk = [_to_float(x) for x in d[i - w + 1: i + 1]]
                out.append(math.nan if any(math.isnan(x) for x in chunk) else min(chunk))
        return Series(out)

    def max(self) -> "Series":
        out = []
        w = self._w
        d = self._data
        for i in range(len(d)):
            if i + 1 < w:
                out.append(math.nan)
            else:
                chunk = [_to_float(x) for x in d[i - w + 1: i + 1]]
                out.append(math.nan if any(math.isnan(x) for x in chunk) else max(chunk))
        return Series(out)

    def std(self, ddof: int = 1) -> "Series":
        out = []
        w = self._w
        d = self._data
        for i in range(len(d)):
            if i + 1 < w or w - ddof <= 0:
                out.append(math.nan)
            else:
                chunk = [_to_float(x) for x in d[i - w + 1: i + 1]]
                if any(math.isnan(x) for x in chunk):
                    out.append(math.nan)
                else:
                    mu = sum(chunk) / w
                    var = sum((x - mu) ** 2 for x in chunk) / (w - ddof)
                    out.append(math.sqrt(max(0.0, var)))
        return Series(out)


class _EWM:
    def __init__(self, data: list, span: float | None = None, alpha: float | None = None):
        self._data = data
        if alpha is not None:
            self._alpha = float(alpha)
        elif span is not None:
            self._alpha = 2.0 / (float(span) + 1.0)
        else:
            self._alpha = 0.5

    def mean(self) -> "Series":
        out = []
        a = self._alpha
        prev = None
        for raw in self._data:
            v = _to_float(raw)
            if math.isnan(v):
                out.append(prev if prev is not None else math.nan)
                continue
            prev = v if prev is None else (a * v + (1.0 - a) * prev)
            out.append(prev)
        return Series(out)


class Series:
    def __init__(self, data, index=None, dtype=None):
        if isinstance(data, Series):
            vals = list(data._data)
            idx = index if index is not None else data.index
        elif isinstance(data, (list, tuple)):
            vals = list(data)
            idx = index if index is not None else list(range(len(vals)))
        else:
            vals = list(data)
            idx = index if index is not None else list(range(len(vals)))
        if dtype in ("int", "int8", int):
            vals = [0 if _is_nan(v) else int(v) for v in vals]
        elif dtype in ("float", float):
            vals = [_to_float(v) for v in vals]
        elif dtype in ("bool", bool):
            vals = [False if _is_nan(v) else bool(v) for v in vals]
        self._data = vals
        self.index = idx

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return Series(self._data[item], index=self.index[item] if self.index else None)
        if isinstance(item, Series):
            return Series([v for v, keep in zip(self._data, item._data) if keep])
        return self._data[item]

    @property
    def iloc(self):
        return self._data

    def tolist(self) -> list:
        return list(self._data)

    def to_list(self) -> list:
        return list(self._data)

    def to_numpy(self, dtype=None) -> list:
        return self.astype(dtype)._data if dtype is not None else list(self._data)

    def rolling(self, window: int) -> _Rolling:
        return _Rolling(self._data, window)

    def ewm(self, span: float | None = None, alpha: float | None = None, adjust: bool = False) -> _EWM:
        return _EWM(self._data, span=span, alpha=alpha)

    def shift(self, periods: int = 1) -> "Series":
        p = int(periods)
        n = len(self._data)
        if p == 0:
            return Series(list(self._data), index=self.index)
        if p > 0:
            pad = [math.nan] * min(p, n)
            return Series(pad + self._data[:max(0, n - p)], index=self.index)
        k = min(-p, n)
        return Series(self._data[k:] + [math.nan] * k, index=self.index)

    def diff(self, periods: int = 1) -> "Series":
        return self - self.shift(periods)

    def pct_change(self, periods: int = 1) -> "Series":
        prev = self.shift(periods)
        return (self - prev) / prev

    def clip(self, lower=None, upper=None) -> "Series":
        out = []
        for v in self._data:
            fv = _to_float(v)
            if math.isnan(fv):
                out.append(math.nan)
                continue
            if lower is not None:
                fv = max(fv, float(lower))
            if upper is not None:
                fv = min(fv, float(upper))
            out.append(fv)
        return Series(out, index=self.index)

    def abs(self) -> "Series":
        return Series([math.nan if _is_nan(v) else abs(v) for v in self._data], index=self.index)

    def fillna(self, value) -> "Series":
        return Series([value if _is_nan(v) else v for v in self._data], index=self.index)

    def replace(self, to_replace, value) -> "Series":
        return Series([value if v == to_replace else v for v in self._data], index=self.index)

    def where(self, cond, other=math.nan) -> "Series":
        c_list = cond._data if isinstance(cond, Series) else list(cond)
        o_list = other._data if isinstance(other, Series) else [other] * len(self._data)
        return Series([a if bool(c) else b for a, c, b in zip(self._data, c_list, o_list)], index=self.index)

    def astype(self, dtype) -> "Series":
        if dtype in (int, "int", "int8", "int64"):
            return Series([0 if _is_nan(v) else int(v) for v in self._data], index=self.index)
        if dtype in (float, "float", "float64"):
            return Series([_to_float(v) for v in self._data], index=self.index)
        if dtype in (bool, "bool"):
            return Series([False if _is_nan(v) else bool(v) for v in self._data], index=self.index)
        return Series(list(self._data), index=self.index)

    def mean(self) -> float:
        clean = [_to_float(v) for v in self._data if not _is_nan(v)]
        return sum(clean) / len(clean) if clean else math.nan

    def std(self) -> float:
        clean = [_to_float(v) for v in self._data if not _is_nan(v)]
        return statistics.pstdev(clean) if len(clean) > 1 else 0.0

    def _binop(self, other, fn):
        if isinstance(other, Series):
            o = other._data
        elif isinstance(other, (list, tuple)):
            o = other
        else:
            o = [other] * len(self._data)
        out = []
        for a, b in zip(self._data, o):
            if _is_nan(a) or _is_nan(b):
                out.append(math.nan)
            else:
                try:
                    out.append(fn(a, b))
                except ZeroDivisionError:
                    out.append(math.nan)
        return Series(out, index=self.index)

    def _cmp(self, other, fn):
        if isinstance(other, Series):
            o = other._data
        elif isinstance(other, (list, tuple)):
            o = other
        else:
            o = [other] * len(self._data)
        out = []
        for a, b in zip(self._data, o):
            if _is_nan(a) or _is_nan(b):
                out.append(False)
            else:
                out.append(bool(fn(a, b)))
        return Series(out, index=self.index)

    def __add__(self, o): return self._binop(o, lambda a, b: a + b)
    def __radd__(self, o): return self._binop(o, lambda a, b: b + a)
    def __sub__(self, o): return self._binop(o, lambda a, b: a - b)
    def __rsub__(self, o): return self._binop(o, lambda a, b: b - a)
    def __mul__(self, o): return self._binop(o, lambda a, b: a * b)
    def __rmul__(self, o): return self._binop(o, lambda a, b: b * a)
    def __truediv__(self, o): return self._binop(o, lambda a, b: a / b if b != 0 else math.nan)
    def __rtruediv__(self, o): return self._binop(o, lambda a, b: b / a if a != 0 else math.nan)
    def __neg__(self): return Series([-v if not _is_nan(v) else math.nan for v in self._data], index=self.index)
    def __gt__(self, o): return self._cmp(o, lambda a, b: a > b)
    def __ge__(self, o): return self._cmp(o, lambda a, b: a >= b)
    def __lt__(self, o): return self._cmp(o, lambda a, b: a < b)
    def __le__(self, o): return self._cmp(o, lambda a, b: a <= b)
    def __eq__(self, o): return self._cmp(o, lambda a, b: a == b)
    def __ne__(self, o): return self._cmp(o, lambda a, b: a != b)
    def __and__(self, o):
        ol = o._data if isinstance(o, Series) else [o] * len(self._data)
        return Series([bool(a) and bool(b) for a, b in zip(self._data, ol)], index=self.index)
    def __or__(self, o):
        ol = o._data if isinstance(o, Series) else [o] * len(self._data)
        return Series([bool(a) or bool(b) for a, b in zip(self._data, ol)], index=self.index)
    def __invert__(self):
        return Series([not bool(a) for a in self._data], index=self.index)


class _IlocIndexer:
    def __init__(self, df: "DataFrame"):
        self._df = df

    def __getitem__(self, key):
        if isinstance(key, slice):
            cols = {k: s._data[key] for k, s in self._df._cols.items()}
            idx = self._df.index[key]
            return DataFrame(cols, index=idx)
        i = int(key)
        return {k: s._data[i] for k, s in self._df._cols.items()}


class DataFrame:
    def __init__(self, cols: dict, index=None):
        self._cols = {k: (v if isinstance(v, Series) else Series(v)) for k, v in cols.items()}
        first = next(iter(self._cols.values()), Series([]))
        self.index = index if index is not None else list(range(len(first)))

    def __len__(self) -> int:
        first = next(iter(self._cols.values()), None)
        return len(first) if first is not None else 0

    def __getitem__(self, key: str) -> Series:
        if key not in self._cols:
            raise KeyError(f"Column '{key}' not found in DataFrame")
        return self._cols[key]

    def __contains__(self, key: str) -> bool:
        return key in self._cols

    @property
    def columns(self) -> list[str]:
        return list(self._cols.keys())

    @property
    def iloc(self) -> _IlocIndexer:
        return _IlocIndexer(self)

    def to_records(self) -> list[dict]:
        n = len(self)
        keys = list(self._cols.keys())
        return [{k: self._cols[k]._data[i] for k in keys} for i in range(n)]


class _PDShim:
    Series = Series
    DataFrame = DataFrame

    @staticmethod
    def unique(s):
        seen = []
        vals = s._data if isinstance(s, Series) else list(s)
        for v in vals:
            if not _is_nan(v) and v not in seen:
                seen.append(v)
        return seen


class _NPShim:
    nan = math.nan
    int8 = int
    int64 = int
    float64 = float

    @staticmethod
    def zeros(n, dtype=float):
        v = 0 if dtype in (int, "int", "int8") else 0.0
        return Series([v] * int(n))

    @staticmethod
    def ones(n, dtype=float):
        v = 1 if dtype in (int, "int", "int8") else 1.0
        return Series([v] * int(n))

    @staticmethod
    def where(cond, a, b):
        c_l = cond._data if isinstance(cond, Series) else list(cond)
        a_l = a._data if isinstance(a, Series) else (list(a) if isinstance(a, (list, tuple)) else [a] * len(c_l))
        b_l = b._data if isinstance(b, Series) else (list(b) if isinstance(b, (list, tuple)) else [b] * len(c_l))
        return Series([x if bool(c) else y for c, x, y in zip(c_l, a_l, b_l)])

    @staticmethod
    def asarray(x, dtype=None):
        return Series(x, dtype=dtype)

    @staticmethod
    def abs(x):
        return x.abs() if isinstance(x, Series) else abs(x)

    @staticmethod
    def maximum(a, b):
        al = a._data if isinstance(a, Series) else [a] * (len(b) if isinstance(b, Series) else 1)
        bl = b._data if isinstance(b, Series) else [b] * len(al)
        return Series([max(x, y) if not (_is_nan(x) or _is_nan(y)) else math.nan for x, y in zip(al, bl)])

    @staticmethod
    def minimum(a, b):
        al = a._data if isinstance(a, Series) else [a] * (len(b) if isinstance(b, Series) else 1)
        bl = b._data if isinstance(b, Series) else [b] * len(al)
        return Series([min(x, y) if not (_is_nan(x) or _is_nan(y)) else math.nan for x, y in zip(al, bl)])

    @staticmethod
    def sqrt(x):
        return Series([math.sqrt(v) if not _is_nan(v) and v >= 0 else math.nan for v in x._data]) if isinstance(x, Series) else math.sqrt(x)


pd_shim = _PDShim()
np_shim = _NPShim()


class ParamSpec:
    def __init__(self, name, default, low, high, kind="int", step=None):
        self.name = name
        self.default = default
        self.low = low
        self.high = high
        self.kind = kind
        self.step = step


def _make_indicator_decorator(collected):
    def indicator(name: str, *, category: str = "Custom", params=(), description: str = "", version: str = "1.0.0", **kwargs):
        def decorator(fn):
            collected.append({
                "kind": "indicator",
                "name": name,
                "category": category,
                "description": description,
                "version": version,
                "params": [
                    {"name": p.name, "default": p.default, "low": p.low, "high": p.high, "kind": p.kind, "step": p.step}
                    if isinstance(p, ParamSpec) else p for p in params
                ],
                "fn": fn,
            })
            return fn
        return decorator
    return indicator


def _make_strategy_decorator(collected):
    def strategy(name: str, *, description: str = "", version: str = "1.0.0", params=(), **kwargs):
        def decorator(fn):
            collected.append({
                "kind": "strategy",
                "name": name,
                "description": description,
                "version": version,
                "params": [
                    {"name": p.name, "default": p.default, "low": p.low, "high": p.high, "kind": p.kind, "step": p.step}
                    if isinstance(p, ParamSpec) else p for p in params
                ],
                "fn": fn,
            })
            return fn
        return decorator
    return strategy


class _RegistryShimModule:
    def __init__(self, ind_dec, strat_dec):
        self.ParamSpec = ParamSpec
        self.indicator = ind_dec
        self.strategy = strat_dec


def _safe_builtins():
    allowed = {
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "int", "isinstance", "len", "list", "map", "max", "min", "pow",
        "print", "range", "reversed", "round", "set", "slice", "sorted",
        "str", "sum", "tuple", "zip", "Exception", "ValueError", "KeyError",
        "TypeError", "RuntimeError", "ZeroDivisionError",
    }
    safe = {k: getattr(builtins, k) for k in allowed if hasattr(builtins, k)}
    safe["__build_class__"] = builtins.__build_class__

    def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if name == "pandas" or root == "pandas":
            return pd_shim
        if name == "numpy" or root == "numpy":
            return np_shim
        if root == "indicatorlab":
            return _RegistryShimModule(
                _safe_import._ind_dec,
                _safe_import._strat_dec,
            )
        if root in BANNED_MODULES:
            raise ImportError(f"Module '{name}' is blocked in the sandbox")
        if root in {"math", "statistics", "collections", "itertools", "functools", "datetime", "json", "re"}:
            return builtins.__import__(name, globals, locals, fromlist, level)
        raise ImportError(f"Module '{name}' is not available in the isolated sandbox")

    return safe, _safe_import


def _build_df(candles: list[dict]) -> DataFrame:
    return DataFrame({
        "open": [c["open"] for c in candles],
        "high": [c["high"] for c in candles],
        "low": [c["low"] for c in candles],
        "close": [c["close"] for c in candles],
        "volume": [c.get("volume", 0.0) for c in candles],
    }, index=[c["time"] for c in candles])


def _normalize_signals(raw, n: int) -> list[int]:
    if isinstance(raw, Series):
        vals = raw._data
    elif isinstance(raw, (list, tuple)):
        vals = list(raw)
    else:
        raise ValueError("Return value must be a Series or list aligned to candle count")
    if len(vals) != n:
        raise ValueError(f"Expected {n} bar values, got {len(vals)}")
    out = []
    for idx, v in enumerate(vals):
        if _is_nan(v):
            out.append(0)
            continue
        if isinstance(v, bool):
            iv = int(v)
        elif isinstance(v, (int, float)):
            if not math.isfinite(v):
                out.append(0)
                continue
            iv = int(round(v)) if abs(v - round(v)) < 1e-9 else None
        else:
            iv = None
        if iv not in (-1, 0, 1):
            raise ValueError(f"Signal at bar {idx} is {v!r}; values must be in {{-1, 0, 1}}")
        out.append(iv)
    return out


def _compile_and_load(code: str):
    collected = []
    ind_dec = _make_indicator_decorator(collected)
    strat_dec = _make_strategy_decorator(collected)
    safe_b, safe_imp = _safe_builtins()
    safe_imp._ind_dec = ind_dec
    safe_imp._strat_dec = strat_dec
    safe_b["__import__"] = safe_imp

    ns = {
        "__builtins__": safe_b,
        "__name__": "user_sandbox_module",
        "pd": pd_shim,
        "np": np_shim,
        "math": math,
        "statistics": statistics,
        "ParamSpec": ParamSpec,
        "indicator": ind_dec,
        "strategy": strat_dec,
    }
    compiled = compile(code, "<studio_code>", "exec")
    exec(compiled, ns, ns)
    return ns, collected


def _find_callable(ns: dict, collected: list, preferred_names: tuple[str, ...]):
    if collected:
        return collected[-1]["fn"], collected[-1]
    for name in preferred_names:
        fn = ns.get(name)
        if callable(fn):
            return fn, None
    user_fns = [
        v for k, v in ns.items()
        if callable(v) and not k.startswith("_") and getattr(v, "__module__", "") == "user_sandbox_module"
    ]
    if user_fns:
        return user_fns[0], None
    return None, None


def _run_indicator(payload: dict) -> dict:
    code = payload["code"]
    candles = payload.get("candles") or []
    params = payload.get("params") or {}
    syntax_only = bool(payload.get("syntax_only", False))

    ok, err = check_ast_safety(code)
    if not ok:
        return {"ok": False, "stage": "syntax", "error": err}

    try:
        ns, collected = _compile_and_load(code)
    except Exception as exc:
        tb = traceback.extract_tb(exc.__traceback__)
        line = next((f.lineno for f in reversed(tb) if f.filename == "<studio_code>"), 1)
        return {"ok": False, "stage": "load", "error": {"type": type(exc).__name__, "line": line, "message": str(exc)}}

    fn, dec_meta = _find_callable(ns, collected, ("compute", "indicator_signal", "signal"))
    if fn is None:
        return {
            "ok": False,
            "stage": "contract",
            "error": {"type": "ContractError", "line": 1, "message": "Define a callable function such as `def compute(df, **params):`."},
        }

    if syntax_only:
        return {"ok": True, "stage": "syntax", "detected_meta": {k: v for k, v in (dec_meta or {}).items() if k != "fn"}}

    df = _build_df(candles)
    t0 = time.perf_counter()
    try:
        raw = fn(df, **params)
        signals = _normalize_signals(raw, len(df))
    except Exception as exc:
        tb = traceback.extract_tb(exc.__traceback__)
        line = next((f.lineno for f in reversed(tb) if f.filename == "<studio_code>"), 1)
        return {"ok": False, "stage": "runtime", "error": {"type": type(exc).__name__, "line": line, "message": str(exc)}}
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    # Causality check on a prefix of the candles
    causal = True
    causality_note = "Passed prefix invariance check"
    if len(candles) >= 80:
        cut = len(candles) // 2
        try:
            prefix_raw = fn(_build_df(candles[:cut]), **params)
            prefix_sig = _normalize_signals(prefix_raw, cut)
            if prefix_sig != signals[:cut]:
                causal = False
                causality_note = "Look-ahead detected: truncated series signals differ from full series"
        except Exception as exc:
            causal = False
            causality_note = f"Causality check error: {exc}"

    bullish = sum(1 for s in signals if s == 1)
    bearish = sum(1 for s in signals if s == -1)
    neutral = len(signals) - bullish - bearish
    tail_n = min(12, len(candles))
    sample = [
        {
            "time": candles[i]["time"],
            "close": candles[i]["close"],
            "signal": signals[i],
        }
        for i in range(len(candles) - tail_n, len(candles))
    ]
    return {
        "ok": True,
        "stage": "executed",
        "execution_ms": elapsed_ms,
        "bars": len(signals),
        "causal": causal,
        "causality_note": causality_note,
        "signal_counts": {"bullish": bullish, "bearish": bearish, "neutral": neutral},
        "signals": signals if payload.get("return_signals") else None,
        "sample": sample,
        "detected_meta": {k: v for k, v in (dec_meta or {}).items() if k != "fn"},
    }


def _run_strategy(payload: dict) -> dict:
    code = payload["code"]
    candles = payload.get("candles") or []
    params = payload.get("params") or {}
    indicator_codes = payload.get("indicator_codes") or []
    syntax_only = bool(payload.get("syntax_only", False))

    ok, err = check_ast_safety(code)
    if not ok:
        return {"ok": False, "stage": "syntax", "error": err}

    try:
        ns, collected = _compile_and_load(code)
    except Exception as exc:
        tb = traceback.extract_tb(exc.__traceback__)
        line = next((f.lineno for f in reversed(tb) if f.filename == "<studio_code>"), 1)
        return {"ok": False, "stage": "load", "error": {"type": type(exc).__name__, "line": line, "message": str(exc)}}

    fn, dec_meta = _find_callable(ns, collected, ("generate_signals", "evaluate", "compute_strategy", "strategy_signals"))
    if fn is None:
        return {
            "ok": False,
            "stage": "contract",
            "error": {"type": "ContractError", "line": 1, "message": "Define a callable function such as `def generate_signals(df, indicators, **params):`."},
        }

    if syntax_only:
        return {"ok": True, "stage": "syntax", "detected_meta": {k: v for k, v in (dec_meta or {}).items() if k != "fn"}}

    df = _build_df(candles)
    ind_signals = {}
    t0 = time.perf_counter()

    # Compute each registered indicator referenced by this strategy inside the isolated sandbox
    for ind in indicator_codes:
        i_ok, i_err = check_ast_safety(ind["code"])
        if not i_ok:
            return {"ok": False, "stage": "indicator", "error": {"type": "IndicatorError", "line": 1, "message": f"Indicator '{ind['name']}': {i_err['message']}"}}
        try:
            i_ns, i_col = _compile_and_load(ind["code"])
            i_fn, _ = _find_callable(i_ns, i_col, ("compute", "indicator_signal", "signal"))
            if i_fn is None:
                raise ValueError(f"Indicator '{ind['name']}' has no compute() function")
            raw_sig = i_fn(df, **(ind.get("params") or {}))
            ind_signals[ind["name"]] = Series(_normalize_signals(raw_sig, len(df)), index=df.index)
        except Exception as exc:
            return {"ok": False, "stage": "indicator", "error": {"type": type(exc).__name__, "line": 1, "message": f"Indicator '{ind['name']}' failed: {exc}"}}

    try:
        res = fn(df, ind_signals, **params)
        exit_signals = [0] * len(df)
        sl_override = None
        tp_override = None
        if isinstance(res, dict):
            entry_raw = res.get("signals") if "signals" in res else res.get("entry_signals")
            if entry_raw is None:
                raise ValueError("Strategy dict return must include 'signals' or 'entry_signals'")
            signals = _normalize_signals(entry_raw, len(df))
            if res.get("exit_signals") is not None:
                exit_signals = _normalize_signals(res["exit_signals"], len(df))
            if res.get("stop_loss_pct") is not None:
                sl_override = float(res["stop_loss_pct"])
            if res.get("target_pct") is not None:
                tp_override = float(res["target_pct"])
        else:
            signals = _normalize_signals(res, len(df))
    except Exception as exc:
        tb = traceback.extract_tb(exc.__traceback__)
        line = next((f.lineno for f in reversed(tb) if f.filename == "<studio_code>"), 1)
        return {"ok": False, "stage": "runtime", "error": {"type": type(exc).__name__, "line": line, "message": str(exc)}}

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    long_signals = sum(1 for s in signals if s == 1)
    short_signals = sum(1 for s in signals if s == -1)

    return {
        "ok": True,
        "stage": "executed",
        "execution_ms": elapsed_ms,
        "bars": len(signals),
        "signal_counts": {"long": long_signals, "short": short_signals, "flat": len(signals) - long_signals - short_signals},
        "signals": signals,
        "exit_signals": exit_signals,
        "stop_loss_pct": sl_override,
        "target_pct": tp_override,
        "detected_meta": {k: v for k, v in (dec_meta or {}).items() if k != "fn"},
    }


def main() -> None:
    raw = sys.stdin.read()
    payload = json.loads(raw)
    apply_resource_limits(
        cpu_seconds=int(payload.get("cpu_limit_s", 5)),
        mem_mb=int(payload.get("mem_limit_mb", 256)),
    )
    mode = payload.get("mode")
    if mode == "indicator":
        result = _run_indicator(payload)
    elif mode == "strategy":
        result = _run_strategy(payload)
    else:
        result = {"ok": False, "stage": "input", "error": {"type": "ValueError", "line": 1, "message": f"Unknown mode: {mode}"}}
    sys.stdout.write(json.dumps(result))


if __name__ == "__main__":
    main()
