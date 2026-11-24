# Indicator Lab – Autonomous Indicator Discovery & Backtesting Platform

A standalone research platform for discovering, combining, optimizing and validating
indicator-based trading strategies. It has a FastAPI backend, a React dashboard with nine
pages, SQLite persistence, Optuna-driven search, and two autonomous agents.

It contains **no market data and no reference to any external project**. Historical candles
come from CSV files you import. The seven built-in indicators are generic textbook examples used to
exercise the registry. They are not a copy of anyone's indicator library.

---

## Current state (completion phase)

**Main navigation (exactly four sections):** Dashboard, Market Data, Indicator, Strategy.
The other pages (Indicator Studio, Strategy Studio, AI Agents, Optimization, Backtest, Results,
Experiments, Analytics) are unchanged and reachable from the "Other pages" list on the Dashboard.

**Research workflow**

1. *Register indicators* in Indicator Studio. Every registered indicator is discovered automatically;
   nothing is hardcoded. Newly registered indicators appear as "not evaluated" until evaluated.
2. *Signal rules* (Indicator page): use an indicator's own -1/0/+1 votes, or build long and short
   conditions on its continuous output: thresholds (>, ≥, <, ≤), crossovers (crosses above/below),
   state changes, AND/OR within a block. A bar votes long only when only the long block holds.
   Rules are part of the cache key, so an edited rule never reuses an older result.
3. *Individual evaluation* (Indicator page): trades, wins, losses, win rate, Beta-Binomial score,
   Wilson lower bound, net profit, profit factor, drawdown and costs, on the train split.
4. *Combination search* (Indicator page): singles, pairs, triples and SFFS up to five indicators,
   AND logic, Spearman filter (default |ρ| ≥ 0.80 dropped), walk-forward folds with a purge gap,
   one final selection on the full train span, and a frozen best combination.
5. *Robustness* (Dashboard, optional): Hansen SPA, PBO (CSCV), Deflated Sharpe, circular-shift
   permutation test and Monte Carlo over trades. Train span only.
6. *Analysis* (Dashboard, optional): Pareto front over return, win rate and drawdown; clusters of the
   Spearman matrix. Nothing is removed automatically.
7. *One-shot OOS* (Dashboard): the frozen combination is evaluated once on the last 20% of the range.
   It cannot be repeated for the same run.
8. *Synthetic test markets* (Indicator page): trending, sideways, volatile and sudden-move scenarios.
   Every synthetic dataset is flagged, and results on it are labelled as not market performance.

**Not implemented:** neural-network analysis and SHAP explanations. Neither library is in the
requirements, and the spec asks for no unnecessary dependencies. These can be added as an optional
module later.

**Setup for the browser UI:** build the frontend (`cd frontend && npm install && npm run build`), then
start the backend. The backend serves `frontend/dist` on the same port.

---

## Contents

1. [Quick start](#quick-start)
2. [Architecture](#architecture)
3. [Indicator registry](#indicator-registry)
4. [Backtesting engine](#backtesting-engine)
5. [Data splits and out-of-sample policy](#data-splits-and-out-of-sample-policy)
6. [The two agents](#the-two-agents)
7. [Optuna integration](#optuna-integration)
8. [Dashboard pages](#dashboard-pages)
9. [API reference](#api-reference)
10. [Configuration](#configuration)
11. [Testing](#testing)
12. [Limitations and what to verify](#limitations-and-what-to-verify)

---

## Quick start

Requirements: Python 3.11+ and Node.js 18+.

```bash
# 1. Backend
cd backend
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn indicatorlab.api:app --host 127.0.0.1 --port 8000

# 2. Frontend (new terminal)
cd frontend
npm install
npm run dev                                              # http://localhost:5173
```

Then open http://localhost:5173. Import a dataset (see below) and pick it in the top-right
dataset selector.

**Import your candles** (CSV, see `data/samples/README.md` for the accepted layouts):

```bash
curl -X POST localhost:8000/api/datasets/import -H 'Content-Type: application/json' \
  -d '{"path":"/abs/path/to/NIFTY_1m.csv","name":"NIFTY_1m","symbol":"NIFTY","timeframe":"1min"}'
```

**Test data only:** `python scripts/generate_sample_data.py` writes a *synthetic* random-walk series.
It has no market meaning and exists only to exercise the pipeline.

Run the tests with `cd backend && pytest tests/`.

---

## Architecture

```
backend/indicatorlab/
  config.py        paths, cost model, split and backtest defaults (environment-overridable)
  db.py            SQLite schema and access (single connection guarded by a lock)
  registry.py      dynamic indicator registry: discovery, parameter specs, self-test
  plugins/         built-in reference indicators (generic examples)
  data.py          CSV import and validation, candle loading, synthetic generator (testing only)
  strategy.py      strategy spec, order-independent hash, vote computation, GA operators
  engine.py        bar-by-bar backtest with explicit execution rules
  metrics.py       net profit, win rate, profit factor, drawdown, recovery factor, cost totals
  evaluator.py     chronological splits, chunked train evaluation, walk-forward, deduplicated records
  optimizer.py     Optuna studies: samplers, MedianPruner, multi-objective, stop/pause hooks
  ranking.py       leaderboards computed from stored experiments
  analytics.py     Optuna visualizations, equity/drawdown, walk-forward and train-vs-validation data
  agents/
    runtime.py       worker lifecycle, pause/stop, heartbeats, logs, checkpoints, CPU/RAM back-off
    research.py      Agent 1: Strategy Research Agent
    evaluation.py    Agent 2: Evaluation & Ranking Agent
    controller.py    start/pause/resume/stop, restart recovery, optimization runs
  api.py           FastAPI routes (all metrics come from the database or live workers)
frontend/src/      React + Vite; nine pages; Plotly charts
indicator_plugins/ drop-in folder for your own indicators (see its README and _template.py)
```

**Data flow.** CSV → validated candles in SQLite → indicator signals (cached per dataset,
parameters and indicator) → strategy votes → bar-by-bar simulation → metrics → experiment row
(one completed row per strategy, dataset, split and config, enforced by a unique index).
Agent 2 and the leaderboards read only those rows.

---

## Indicator registry

An indicator is a function `compute(df, **params) -> pd.Series` returning values in `{-1, 0, 1}`
(bullish, bearish, neutral), aligned to `df.index`. It must be **causal**: the value at bar *t*
may use only bars up to *t*.

```python
from indicatorlab.registry import indicator, ParamSpec

@indicator("my_indicator", category="Momentum",
           params=(ParamSpec("period", 14, 5, 30),), description="...")
def my_indicator(df, period=14):
    ...
    return signal  # int8 Series in {-1, 0, 1}
```

- **Discovery is runtime-based.** Built-in modules and every `*.py` in `indicator_plugins/`
  (files starting with `_` are skipped) are loaded and registered. Nothing in the agents,
  the API or the frontend names an indicator.
- **Self-test.** Each indicator runs on synthetic bars at discovery. Failures are listed as
  *unavailable* with the reason and are excluded from search. Load errors are reported, not fatal.
- **Causality is tested.** `tests/test_core.py` verifies that every built-in indicator gives the
  same values on a truncated series as on the full series, which catches look-ahead bugs.
- The dashboard's indicator count is `REGISTRY.summary()`. It is never a constant.

---

## Backtesting engine

**Execution rules (look-ahead prevention):**

1. Votes are read at the **close** of bar *t*.
2. An order decided at that close **fills at the open of bar *t+1***, plus adverse slippage.
3. Stop-loss and target are checked against the high and low of bars **after** the entry bar.
   If one bar touches both, the **stop is assumed to fill first** (conservative).
4. A gap through the stop fills at the bar's **open**, not at the stop price.
5. Positions still open are closed at the square-off minute (default 15:15) or at the end of the slice.
   No new entries after the no-entry minute (default 15:00).

**Sizing.** Each trade commits `position_pct` of current cash, rounded down to whole lots.
Entries that cannot be sized to one lot are counted as `skipped_entries` and shown in the UI.
The model is linear: there is no margin, exposure or lot-expiry model. Add one if you trade
derivatives that need it.

**Costs** (`CostModel`, all configurable per run):
brokerage per order (flat), taxes and charges as a fraction of order turnover, and slippage in
points per fill. **The default values are placeholders.** Replace them with your broker's and the
exchange's real schedule before drawing any conclusion.

**Accounting** is checked by tests: final capital equals initial capital plus the sum of trade net P&L,
and the last equity value equals final cash.

**Metrics.** Net profit, return %, win rate, profit factor (capped at 999 when there are no losses),
maximum drawdown (absolute and %), recovery factor (net profit ÷ max drawdown), average trade,
average win/loss, total costs, long/short counts, and exit-reason counts.

---

## Data splits and out-of-sample policy

Splits are chronological and set by bar position (defaults 60 / 20 / 20):

| Split | Used by | Notes |
|---|---|---|
| **train** | Agent 1, Optuna search, leaderboards | Evaluated in 3 sequential chunks (for pruning). Positions are flattened at each chunk end. |
| **validation** | Walk-forward validation (Agent 1) | Split into 4 independent segments, each starting with fresh capital. |
| **out-of-sample** | Only the explicit *Run out-of-sample* action | Runs **once per strategy and dataset**, then is stored permanently. |

The agents' code paths never read the out-of-sample tail. A test verifies that no agent-produced
experiment has split `oos`. Walk-forward **validated** means: at least half of the validation segments
are profitable, the validation net profit is positive, and the segments contain at least the minimum trade count.

**Minimum trades** (`IL_MIN_TRADES`, default 30) are required before a result is *eligible* for any leaderboard.

---

## The two agents

### Agent 1 – Strategy Research Agent

Each cycle runs these phases in order. Each phase feeds the next:

1. **Forward selection.** Starting from nothing, add the indicator that most improves train profit,
   until no addition helps or `max_indicators` is reached.
2. **Backward elimination.** Remove any indicator whose removal does not reduce train profit.
3. **Genetic algorithm.** Population of strategies. Tournament selection (k=3), two-point elite
   carry-over, crossover of indicator sets and parameters, and mutation (parameter, add, remove,
   votes, risk). Checkpointed every generation.
4. **Hill climbing.** Step to the best neighbour (one parameter ±step, votes ±1, adjacent stop/target,
   direction change) until no neighbour improves.
5. **Optuna tuning.** TPE over the parameters of the current best structure, with MedianPruner.
6. **Walk-forward validation.** The best candidates from this cycle and the current leaderboard are
   validated on the validation split, and each result is stored as a new row.

Cycles repeat (`cycles=0` runs until stopped). Random streams are seeded per cycle.
A checkpoint is written after every completed phase and every GA generation. After a restart the cycle resumes from the last completed phase, and the genetic algorithm resumes from its last completed generation. The interrupted step is re-run, but results already recorded are reused, not re-computed.

**Not an LLM.** The agent uses deterministic search. It makes no per-trial model calls.

### Agent 2 – Evaluation & Ranking Agent

Runs alongside Agent 1. It only reads stored experiments:

- processes new experiments since its last pass (watermark on the experiment id),
- audits the ledger: failed and pruned counts, missing metrics, eligibility mismatches, duplicate completed rows,
- recomputes all six leaderboards and publishes the snapshot to its checkpoint.

Leaderboards: highest net profit · highest win rate · highest profit factor · lowest maximum drawdown
(profitable strategies only) · best risk-adjusted (recovery factor) · best walk-forward validated.

**Never overwrites.** Completed experiments are unique by (strategy, dataset, split, config).
A repeated evaluation returns the stored result and does not re-run or re-record it. Failures and pruned
trials are recorded separately with their status.

### Controls and recovery

- Pause, resume and stop act on the live worker. Stop waits for the current evaluation to finish.
- Workers back off when host CPU or RAM is above the configured limits, and log it.
- On process start, any worker that was `running` or `paused` is marked **stopped** with a warning.
  Its checkpoint is kept. Start the agents again and they continue from the saved phase.

---

## Optuna integration

Uses the public APIs of Optuna 5.x with RDB storage (`data/optuna.db`):

- **Samplers:** TPE, Random, Grid, NSGA-II, CMA-ES.
- **Pruner:** MedianPruner with intermediate reports at each train chunk. Disabled for multi-objective
  studies, which Optuna does not support for intermediate reporting.
- **Objectives:** net profit (single objective), or net profit and minus drawdown % (multi-objective, Pareto front).
- **Search space:** every available indicator as a boolean `use__<name>`, parameters discretised to
  `choices_per_param` values (so GridSampler can enumerate them), votes, stop, target and direction.
- **Visualizations:** optimization history, parameter importance, contour, slice, parallel coordinates and
  the Pareto front are generated with `optuna.visualization` and rendered in the UI.
- **Stop and pause** are delivered through `study.stop()` and a gate inside the objective.
  Exceptions in trials are recorded as failed trials, and the study continues.

Caveats: CMA-ES works on continuous spaces. With categorical choices it falls back to independent
sampling, and the UI says so. Grid search is capped at 200,000 points.

The **Optuna Dashboard** package is installed as a dependency, but it is not mounted into this app.
To use it alongside, run `optuna-dashboard sqlite:///data/optuna.db`.

---

## Dashboard pages

| Page | Shows |
|---|---|
| Dashboard | Research results only: indicator ranking, top combinations, correlation heatmap, walk-forward folds, one-shot OOS, equity and drawdown, robustness, Pareto and clusters. |
| AI Agents | Start and control both agents, live logs, current task, algorithm, trials, errors, heartbeat, uptime, ledger audit. |
| Strategy | Research combinations (load one as an AND strategy), plus the vote-strategy builder. |
| Indicator (main) | Rules form, individual evaluation table (sort, filter, CSV, row details), combination search with coverage, signal rules, synthetic test markets. |
| Indicator Studio | Register, validate and delete indicators (unchanged). |
| Optimization | Study configuration (sampler, pruner, objective, trials, workers), runs with pause/resume/stop, best result, history and importance charts. |
| Backtest | Price chart with entry and exit markers, equity and drawdown, full metrics, trade history, cost inputs, one-time out-of-sample run. |
| Results | Six leaderboards (top 10 or top 50) with validation status; click a strategy for its configuration, splits, validations and out-of-sample result. |
| Experiments | Every stored experiment, filterable by status and study, with parameters, objective values, duration, and a side-by-side comparison. |
| Analytics | Optuna charts, equity and drawdown for a chosen strategy, walk-forward table, training vs validation scatter. |

---

## API reference

Interactive docs at http://127.0.0.1:8000/docs.

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health`, `/api/system`, `/api/dashboard` | status, CPU/RAM, dashboard figures |
| GET/POST | `/api/datasets`, `/api/datasets/import`, `/api/datasets/synthetic` | list, import CSV, generate synthetic test data |
| GET | `/api/datasets/{id}/candles?timeframe=` | OHLC bars for charting |
| GET | `/api/indicators`, `/api/indicators/{name}` | registry contents and status |
| POST | `/api/indicators/reload` | rediscover indicators |
| GET | `/api/strategy/templates`, `/api/strategy/options` | templates, valid choices and defaults |
| POST | `/api/strategy/describe` | validate and describe a strategy |
| POST | `/api/backtest` | simulate on `train` or `validation` |
| POST | `/api/backtest/oos` | one-time out-of-sample run (stored) |
| GET/POST | `/api/agents`, `/api/agents/start`, `/api/agents/{name}/{pause\|resume\|stop}` | agent lifecycle |
| GET | `/api/agents/{name}/logs?after=`, `/api/agents/evaluation/snapshot` | logs and ledger snapshot |
| GET/POST | `/api/optimization/options`, `/runs`, `/start`, `/{study}/{action}` | Optuna studies |
| GET | `/api/optimization/studies/{study}` | study summary |
| GET | `/api/results/rankings[/{key}]?dataset_id=` | leaderboards |
| GET | `/api/results/strategies/{hash}?dataset_id=` | strategy detail |
| GET | `/api/experiments?dataset_id=&study=&status=` | experiment ledger |
| GET | `/api/analytics/optuna/{study}/{history\|importance\|contour\|slice\|parallel\|pareto}` | Optuna charts |
| GET | `/api/analytics/equity`, `/walk_forward`, `/train_vs_validation` | analytics data |

---

## Configuration

Environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `IL_DATA_DIR` | `./data` | data folder |
| `IL_DB_PATH` | `data/indicator_lab.db` | application database |
| `IL_OPTUNA_STORAGE` | `sqlite:///data/optuna.db` | Optuna storage URL |
| `IL_INDICATOR_PLUGIN_DIR` | `./indicator_plugins` | drop-in indicator folder |
| `IL_MIN_TRADES` | `30` | minimum trades for leaderboard eligibility |
| `IL_CPU_LIMIT_PCT` / `IL_RAM_LIMIT_PCT` | `90` / `85` | back-off thresholds (host-wide) |

Costs, capital, sizing, square-off times and split fractions are defined in `indicatorlab/config.py`
and can be overridden per backtest through the API and the Backtest page.

---

## Testing

```bash
cd backend && pytest tests/ -v
```

40 tests, covering:

- **Core (22):** registry discovery, plugin loading, broken and invalid plugins, parameter validation,
  causality of every built-in indicator, next-open execution, intrabar stop-loss and gap fills, cost
  reconciliation, square-off and no-entry windows, metric arithmetic, CSV validation and
  deduplication, strategy hashing and validation.
- **Integration (18):** Optuna studies (TPE and NSGA-II, Pareto front), trial recording and deduplication,
  the full research cycle, phase logs, checkpoint resume, pause and stop, restart recovery, the
  out-of-sample boundary, the evaluation ledger audit and leaderboard ordering, walk-forward output,
  and the HTTP API (validation errors, one-time out-of-sample, candle resampling, Optuna charts).

Frontend: `npm run build` succeeds. A headless-browser check loaded all nine pages against the live API
with no runtime errors and populated tables and charts.

---

## Limitations and what to verify

- **Your indicators are not included.** Add them as plugins. Until you do, only the seven reference
  examples are available.
- **Costs and taxes are placeholders.** Set brokerage, charges and slippage to your real schedule.
  Results are sensitive to these.
- **No margin or derivatives model.** Position sizing is linear. Expiry, margin and lot-size changes
  are not modelled.
- **Signals are state-based votes.** An indicator that stays bullish keeps voting bullish. Entries are
  triggered by the vote threshold, not by a crossing event.
- **Chunked training evaluation.** Positions close at each train chunk boundary, so every strategy
  is measured the same way. This is a deliberate simplification.
- **Parameter discretisation.** Optuna and GridSampler search over discrete choices, not continuous ranges.
  The research agent's tuning phase uses continuous ranges.
- **Resource back-off uses host-wide CPU and RAM**, not this process alone.
- **SQLite and one process.** Suitable for single-user research. Not designed for concurrent multi-user
  access or horizontal scaling.
- **No authentication.** Bind to localhost, or put the API behind your own access control.
- **Frontend bundle is about 4.9 MB** (Plotly). Code-splitting is a worthwhile follow-up.
- **Not tested here:** Docker packaging (not provided), real market datasets (only synthetic data was
  used), long unattended runs, and load beyond a few thousand trials.
- **Results are only as good as your data and costs.** Use the walk-forward and out-of-sample results
  as evidence, and treat a strong in-sample result with suspicion.

## Statistical significance (added in the enhancement, phase 1 partial)

**Terms used throughout.** *Backtest* = simulated trades on a split. *Validation* = walk-forward segments on the validation split. *OOS* = the final split, read only by the explicit OOS endpoint. *Prediction accuracy* (ML, not yet built) is not a trading result. *Trading win rate* = winning trades / total trades from the backtest engine. *Statistical significance* = DSR/PBO/SPA values, which say how likely a result is to be luck given the search.

- **Schema v2 migration.** `statistical_tests` table, additive only. Any existing v1 database is copied with SQLite's online backup API to `data/backups/<name>-v1-<timestamp>.db` before the change. Existing tables and rows are untouched.
- **Per-experiment statistics.** New experiments store `returns_obs`, `sharpe_per_bar`, `return_skew` and `return_kurtosis` in `metrics_json`. Older experiments lack these keys and are reported as insufficient sample, not guessed.
- **Deflated Sharpe Ratio** (`POST /api/statistics/dsr`, train split only). Uses the eligible trials of the same dataset and config. Stores one row per experiment with the expected maximum Sharpe, trial variance, skew, kurtosis and assumptions. Requires at least 30 return observations and 2 trials. Sharpe is per bar, not annualised.
- **PBO via CSCV** (`POST /api/statistics/pbo`, train split only). Takes the top eligible train configurations (one per strategy), builds a returns matrix and runs combinatorial symmetric cross-validation over an even number of chronological blocks (4–16). Stores every run with its configs and block count.
- **Listing** `GET /api/statistics?dataset_id=&kind=dsr|pbo` and `GET /api/statistics/experiments/{id}/dsr` (returns `not_performed` when nothing has been computed).

Not yet built: condition-based signals (MACD, RSI, VWAP, ADX, Supertrend, AND/OR), Hansen SPA, permutation tests, Monte Carlo, synthetic scenario module, correlation analysis, Pareto optimisation, neural network and SHAP, the frontend pages for these, and DSR-based leaderboard filtering.
