import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import express from 'express';
import {
  DEFAULT_BACKTEST,
  MIN_TRADES,
  RANKINGS,
  addAgentLog,
  agentsStatus,
  buildEvaluationSnapshot,
  buildStrategySpec,
  computeDsrForDataset,
  controlAgent,
  controlOptimizationStudy,
  describeSpec,
  evaluateSpecFull,
  generateSyntheticCandles,
  getLeaderboard,
  getOptimizationRuns,
  getRegisteredIndicators,
  getRegisteredStrategies,
  indicatorDefaults,
  normalizeParamSpecs,
  parseCsvCandles,
  probeCandles,
  recoverAfterRestart,
  resampleCandles,
  runPythonSandbox,
  specHash,
  splitBounds,
  startAutonomousAgents,
  startOptimizationStudy,
  store,
  systemMetrics,
  upsertDataset,
  utcnow,
} from './server_engine.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT || 3000);
const HOST = process.env.HOST || '0.0.0.0';
const IS_PROD = process.env.NODE_ENV === 'production';

const app = express();
app.use(express.json({ limit: '25mb' }));

const recoveredAgents = recoverAfterRestart();

function errDetail(res, status, message) {
  return res.status(status).json({ detail: String(message) });
}

function findDataset(datasetId) {
  const ds = store.state.datasets.find((d) => d.id === Number(datasetId));
  if (!ds) throw Object.assign(new Error(`dataset ${datasetId} not found`), { status: 404 });
  return ds;
}

function makeEquityPayload(bt, metrics, maxPoints = 3000) {
  const eq = bt.equity || [];
  const ts = bt.timestamps || [];
  if (!eq.length) return { timestamps: [], equity: [], drawdown: [], metrics };
  let peak = -Infinity;
  const dd = eq.map((v) => {
    if (v > peak) peak = v;
    return peak - v;
  });
  const step = Math.max(1, Math.floor(eq.length / maxPoints));
  const timestamps = [];
  const equity = [];
  const drawdown = [];
  for (let i = 0; i < eq.length; i += step) {
    timestamps.push(ts[i]);
    equity.push(Math.round(eq[i] * 100) / 100);
    drawdown.push(Math.round(dd[i] * 100) / 100);
  }
  return { timestamps, equity, drawdown, metrics };
}

// ---------------------------------------------------------------------------
// System & Dashboard
// ---------------------------------------------------------------------------
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', recovered_agents: recoveredAgents, runtime: 'node+python3-sandbox' });
});

app.get('/api/system', (_req, res) => {
  res.json(systemMetrics());
});

app.get('/api/dashboard', (req, res) => {
  const dsId = req.query.dataset_id
    ? Number(req.query.dataset_id)
    : (store.state.datasets[store.state.datasets.length - 1]?.id ?? null);

  const exps = store.state.experiments;
  const completed = exps.filter((e) => e.status === 'completed').length;
  const failed = exps.filter((e) => e.status === 'failed').length;
  const pruned = exps.filter((e) => e.status === 'pruned').length;

  const distinctHashes = new Set(exps.map((e) => e.strategy_hash));
  let bestNetProfit = null;
  let bestWinRate = null;
  let bestSharpe = null;
  let eligibleStrategies = 0;
  let validatedStrategies = 0;

  if (dsId) {
    const trainEligible = exps.filter(
      (e) => e.dataset_id === dsId && e.split === 'train' && e.status === 'completed' && e.eligible && e.metrics
    );
    const byHash = new Map();
    for (const e of trainEligible) byHash.set(e.strategy_hash, e);
    eligibleStrategies = byHash.size;
    for (const e of byHash.values()) {
      if (bestNetProfit === null || e.metrics.net_profit > bestNetProfit) bestNetProfit = e.metrics.net_profit;
      if (bestWinRate === null || e.metrics.win_rate > bestWinRate) bestWinRate = e.metrics.win_rate;
      if (bestSharpe === null || (e.metrics.sharpe_ratio ?? -999) > bestSharpe) bestSharpe = e.metrics.sharpe_ratio;
    }
    const valHashes = new Set(
      store.state.validations.filter((v) => v.dataset_id === dsId && v.validated).map((v) => v.strategy_hash)
    );
    validatedStrategies = valHashes.size;
  }

  const regInds = getRegisteredIndicators();
  const regStrats = getRegisteredStrategies();
  const recent = [...store.state.agentLogs].slice(-15).reverse();

  res.json({
    system: systemMetrics(),
    indicators: regInds.length,
    indicators_total: store.state.studioIndicators.length,
    strategies_registered: regStrats.length,
    strategies_total: store.state.studioStrategies.length,
    strategies_discovered: distinctHashes.size,
    trials: { completed, pruned, failed, running: 0 },
    experiments: { completed, failed, pruned },
    best_net_profit: bestNetProfit,
    best_win_rate: bestWinRate,
    best_sharpe_ratio: bestSharpe,
    eligible_strategies: eligibleStrategies,
    validated_strategies: validatedStrategies,
    agents: agentsStatus().agents,
    recent_activity: recent,
  });
});

// ---------------------------------------------------------------------------
// Datasets (list, synthetic, CSV upload, local path import, candles)
// ---------------------------------------------------------------------------
app.get('/api/datasets', (_req, res) => {
  res.json(store.state.datasets);
});

app.post('/api/datasets/synthetic', (req, res) => {
  try {
    const name = String(req.body?.name || 'SYNTH_1m_demo').trim();
    const days = Math.min(500, Math.max(5, Number(req.body?.days || 60)));
    const seed = Number(req.body?.seed || 42);
    const candles = generateSyntheticCandles(days, seed);
    const row = upsertDataset({ name, symbol: 'SYNTHETIC', timeframe: '1min', source_file: 'synthetic_generator', candles, replace: true });
    res.json(row);
  } catch (err) {
    errDetail(res, 400, err.message);
  }
});

app.post('/api/datasets/upload', (req, res) => {
  try {
    const { csv_text, name, symbol, timeframe = '1min', replace = true } = req.body || {};
    if (!csv_text || !name || !symbol) {
      return errDetail(res, 400, 'csv_text, name, and symbol are required');
    }
    const candles = parseCsvCandles(String(csv_text));
    const row = upsertDataset({ name: String(name).trim(), symbol: String(symbol).trim(), timeframe, source_file: 'browser_upload', candles, replace });
    res.json(row);
  } catch (err) {
    errDetail(res, 400, err.message);
  }
});

app.post('/api/datasets/import', (req, res) => {
  try {
    const { path: filePath, name, symbol, timeframe = '1min', replace = false } = req.body || {};
    if (!filePath || !name || !symbol) return errDetail(res, 400, 'path, name, and symbol are required');
    if (!fs.existsSync(filePath)) return errDetail(res, 400, `File not found: ${filePath}`);
    const csvText = fs.readFileSync(filePath, 'utf8');
    const candles = parseCsvCandles(csvText);
    const row = upsertDataset({ name, symbol, timeframe, source_file: filePath, candles, replace });
    res.json(row);
  } catch (err) {
    errDetail(res, 400, err.message);
  }
});

app.get('/api/datasets/:id/candles', (req, res) => {
  try {
    findDataset(req.params.id);
    const timeframe = String(req.query.timeframe || '5min');
    const limit = Math.min(3000, Math.max(10, Number(req.query.limit || 500)));
    const raw = store.loadCandles(req.params.id);
    const resampled = resampleCandles(raw, timeframe);
    res.json({ timeframe, bars: resampled.slice(-limit) });
  } catch (err) {
    errDetail(res, err.status || 400, err.message);
  }
});

// ---------------------------------------------------------------------------
// Dynamic Indicator Registry (read by Agents, Optimization, Backtest, Studio)
// ---------------------------------------------------------------------------
app.get('/api/indicators', (_req, res) => {
  const all = store.state.studioIndicators.map((i) => ({
    id: i.id,
    name: i.name,
    category: i.category,
    version: i.version,
    description: i.description,
    params: i.params,
    defaults: indicatorDefaults(i),
    outputs: ['signal'],
    available: Boolean(i.registered && i.available),
    registered: Boolean(i.registered),
    status: i.status,
    status_reason: i.status_reason || (i.registered ? 'ok' : 'draft (not registered)'),
    module: `studio.indicators.${i.name}`,
  }));
  const categories = {};
  for (const d of all) categories[d.category] = (categories[d.category] || 0) + 1;
  res.json({
    summary: {
      total: all.length,
      available: all.filter((d) => d.available).length,
      unavailable: all.filter((d) => !d.available).length,
      categories,
      load_errors: [],
    },
    indicators: all,
  });
});

app.get('/api/indicators/:name', (req, res) => {
  const found = store.state.studioIndicators.find((i) => i.name === req.params.name);
  if (!found) return errDetail(res, 404, `unknown indicator '${req.params.name}'`);
  res.json({ ...found, defaults: indicatorDefaults(found) });
});

app.post('/api/indicators/reload', (_req, res) => {
  const all = store.state.studioIndicators;
  res.json({
    total: all.length,
    available: all.filter((d) => d.registered && d.available).length,
    unavailable: all.filter((d) => !(d.registered && d.available)).length,
  });
});

// ---------------------------------------------------------------------------
// Indicator Studio API (Code-First, Unlimited Custom Indicators, Isolated Sandbox)
// ---------------------------------------------------------------------------
function resolveValidationCandles(datasetId, maxBars = 1500) {
  if (datasetId) {
    const raw = store.loadCandles(datasetId);
    const bounds = splitBounds(raw.length);
    const trainSlice = raw.slice(bounds.train[0], bounds.train[1]);
    return { candles: trainSlice.slice(-maxBars), source: `dataset #${datasetId} (train split)` };
  }
  return { candles: probeCandles(400), source: 'synthetic probe bars (import a dataset for historical market testing)' };
}

function validateIndicatorMeta(body, existingId = null) {
  const name = String(body.name || '').trim();
  if (!/^[a-z][a-z0-9_]{1,63}$/.test(name)) {
    throw new Error("Indicator name must be snake_case (2-64 chars, lowercase letters/digits/underscores, starting with a letter).");
  }
  const dup = store.state.studioIndicators.find((i) => i.name === name && i.id !== existingId);
  if (dup) throw new Error(`An indicator named '${name}' already exists.`);
  const version = String(body.version || '1.0.0').trim() || '1.0.0';
  const category = String(body.category || 'Custom').trim() || 'Custom';
  const description = String(body.description || '').trim();
  const code = String(body.code || '');
  if (!code.trim()) throw new Error('Python code cannot be empty.');
  const params = normalizeParamSpecs(body.params || []);
  return { name, version, category, description, code, params };
}

app.get('/api/studio/indicators', (_req, res) => {
  res.json({
    total: store.state.studioIndicators.length,
    registered_count: getRegisteredIndicators().length,
    items: store.state.studioIndicators,
  });
});

app.post('/api/studio/indicators', (req, res) => {
  try {
    const meta = validateIndicatorMeta(req.body || {});
    const now = utcnow();
    const item = {
      id: store.state.nextIds.indicator++,
      ...meta,
      status: 'draft',
      registered: false,
      available: false,
      status_reason: 'draft — click Update Indicator to validate and register',
      last_validated_at: null,
      last_validation: null,
      created_at: now,
      updated_at: now,
    };
    store.state.studioIndicators.push(item);
    store.save();
    res.json(item);
  } catch (err) {
    errDetail(res, 400, err.message);
  }
});

app.put('/api/studio/indicators/:id', (req, res) => {
  try {
    const id = Number(req.params.id);
    const idx = store.state.studioIndicators.findIndex((i) => i.id === id);
    if (idx < 0) return errDetail(res, 404, 'Indicator not found');
    const prev = store.state.studioIndicators[idx];
    const meta = validateIndicatorMeta(req.body || {}, id);
    const updated = {
      ...prev,
      ...meta,
      status: prev.registered ? 'registered' : 'draft',
      updated_at: utcnow(),
    };
    store.state.studioIndicators[idx] = updated;
    store.clearSignalCache(prev.name);
    store.save();
    res.json(updated);
  } catch (err) {
    errDetail(res, 400, err.message);
  }
});

app.delete('/api/studio/indicators/:id', (req, res) => {
  const id = Number(req.params.id);
  const idx = store.state.studioIndicators.findIndex((i) => i.id === id);
  if (idx < 0) return errDetail(res, 404, 'Indicator not found');
  const [removed] = store.state.studioIndicators.splice(idx, 1);
  store.clearSignalCache(removed.name);
  addAgentLog('research_agent', 'info', `Indicator '${removed.name}' deleted and removed from catalogue`);
  store.save();
  res.json({ deleted: true, id, name: removed.name });
});

app.post('/api/studio/indicators/validate', async (req, res) => {
  try {
    const { code, params = [], dataset_id = null, syntax_only = false } = req.body || {};
    if (!code || !String(code).trim()) return errDetail(res, 400, 'Python code cannot be empty.');
    const normParams = normalizeParamSpecs(params);
    const defaults = Object.fromEntries(normParams.map((p) => [p.name, p.default]));
    const { candles, source } = syntax_only ? { candles: [], source: 'syntax check only' } : resolveValidationCandles(dataset_id);
    const result = await runPythonSandbox({
      mode: 'indicator',
      code: String(code),
      candles,
      params: defaults,
      syntax_only,
    });
    res.json({ ...result, data_source: source });
  } catch (err) {
    errDetail(res, 400, err.message);
  }
});

// Explicit Update Indicator flow: validate -> save -> register -> update Research Agent catalogue -> NEVER auto-start optimization
app.post('/api/studio/indicators/:id/update', async (req, res) => {
  try {
    const id = Number(req.params.id);
    const idx = store.state.studioIndicators.findIndex((i) => i.id === id);
    if (idx < 0) return errDetail(res, 404, 'Indicator not found');
    const prev = store.state.studioIndicators[idx];
    const meta = validateIndicatorMeta(req.body || {}, id);
    const defaults = Object.fromEntries(meta.params.map((p) => [p.name, p.default]));
    const datasetId = req.body?.dataset_id || null;
    const { candles, source } = resolveValidationCandles(datasetId);

    // 1. Validate code & execute in isolated Python sandbox
    const validation = await runPythonSandbox({
      mode: 'indicator',
      code: meta.code,
      candles,
      params: defaults,
      syntax_only: false,
    });

    if (!validation.ok) {
      const failed = {
        ...prev,
        ...meta,
        status: 'error',
        registered: false,
        available: false,
        status_reason: `${validation.error?.type || 'ValidationError'} (line ${validation.error?.line || 1}): ${validation.error?.message}`,
        last_validated_at: utcnow(),
        last_validation: { ...validation, data_source: source },
        updated_at: utcnow(),
      };
      store.state.studioIndicators[idx] = failed;
      store.save();
      return res.status(400).json({
        detail: failed.status_reason,
        validation: failed.last_validation,
        indicator: failed,
      });
    }

    // 2-5. Save, register in dynamic registry, update Research Agent catalogue, do NOT auto-start optimization
    const registeredItem = {
      ...prev,
      ...meta,
      status: 'registered',
      registered: true,
      available: true,
      status_reason: validation.causal ? 'ok (causal verified)' : 'registered (warning: prefix check flagged look-ahead)',
      last_validated_at: utcnow(),
      last_validation: { ...validation, data_source: source },
      updated_at: utcnow(),
    };
    store.state.studioIndicators[idx] = registeredItem;
    store.clearSignalCache(prev.name);
    store.clearSignalCache(meta.name);
    addAgentLog(
      'research_agent',
      'info',
      `Indicator '${meta.name}' v${meta.version} registered in dynamic catalogue (${getRegisteredIndicators().length} total available). No optimization started automatically.`
    );
    store.save();

    res.json({
      indicator: registeredItem,
      validation: registeredItem.last_validation,
      catalogue_size: getRegisteredIndicators().length,
      auto_started_optimization: false,
      message: `Indicator '${meta.name}' v${meta.version} validated, saved, and registered in the Research Agent catalogue. No optimization run was started automatically.`,
    });
  } catch (err) {
    errDetail(res, 400, err.message);
  }
});

// ---------------------------------------------------------------------------
// Strategy Studio API (Code-First, Unlimited Custom Strategies, Isolated Sandbox)
// ---------------------------------------------------------------------------
function validateStrategyMeta(body, existingId = null) {
  const name = String(body.name || '').trim();
  if (!/^[a-zA-Z][a-zA-Z0-9_\- ]{1,79}$/.test(name)) {
    throw new Error('Strategy name must be 2-80 characters and start with a letter.');
  }
  const dup = store.state.studioStrategies.find((s) => s.name.toLowerCase() === name.toLowerCase() && s.id !== existingId);
  if (dup) throw new Error(`A strategy named '${name}' already exists.`);
  const version = String(body.version || '1.0.0').trim() || '1.0.0';
  const description = String(body.description || '').trim();
  const direction = ['both', 'long', 'short'].includes(body.direction) ? body.direction : 'both';
  const stop_loss_pct = Math.max(0, Math.min(25, Number(body.stop_loss_pct ?? 0.5)));
  const target_pct = Math.max(0, Math.min(50, Number(body.target_pct ?? 1.0)));
  const entry_notes = String(body.entry_notes || '').trim();
  const exit_notes = String(body.exit_notes || '').trim();
  const linked_indicators = Array.isArray(body.linked_indicators) ? body.linked_indicators.map(String) : [];
  const code = String(body.code || '');
  if (!code.trim()) throw new Error('Python strategy code cannot be empty.');
  const params = normalizeParamSpecs(body.params || []);
  return {
    name, version, description, direction, stop_loss_pct, target_pct,
    entry_notes, exit_notes, linked_indicators, code, params,
  };
}

app.get('/api/studio/strategies', (_req, res) => {
  res.json({
    total: store.state.studioStrategies.length,
    registered_count: getRegisteredStrategies().length,
    items: store.state.studioStrategies,
  });
});

app.post('/api/studio/strategies', (req, res) => {
  try {
    const meta = validateStrategyMeta(req.body || {});
    const now = utcnow();
    const item = {
      id: store.state.nextIds.strategy++,
      ...meta,
      status: 'draft',
      registered: false,
      available: false,
      status_reason: 'draft — click Update Strategy to validate and register',
      last_validated_at: null,
      last_validation: null,
      created_at: now,
      updated_at: now,
    };
    store.state.studioStrategies.push(item);
    store.save();
    res.json(item);
  } catch (err) {
    errDetail(res, 400, err.message);
  }
});

app.put('/api/studio/strategies/:id', (req, res) => {
  try {
    const id = Number(req.params.id);
    const idx = store.state.studioStrategies.findIndex((s) => s.id === id);
    if (idx < 0) return errDetail(res, 404, 'Strategy not found');
    const prev = store.state.studioStrategies[idx];
    const meta = validateStrategyMeta(req.body || {}, id);
    const updated = {
      ...prev,
      ...meta,
      status: prev.registered ? 'registered' : 'draft',
      updated_at: utcnow(),
    };
    store.state.studioStrategies[idx] = updated;
    store.save();
    res.json(updated);
  } catch (err) {
    errDetail(res, 400, err.message);
  }
});

app.delete('/api/studio/strategies/:id', (req, res) => {
  const id = Number(req.params.id);
  const idx = store.state.studioStrategies.findIndex((s) => s.id === id);
  if (idx < 0) return errDetail(res, 404, 'Strategy not found');
  const [removed] = store.state.studioStrategies.splice(idx, 1);
  addAgentLog('research_agent', 'info', `Strategy '${removed.name}' deleted and removed from catalogue`);
  store.save();
  res.json({ deleted: true, id, name: removed.name });
});

async function executeStrategyValidation({ code, params = [], linked_indicators = [], direction = 'both', stop_loss_pct = 0.5, target_pct = 1.0, dataset_id = null, syntax_only = false }) {
  const normParams = normalizeParamSpecs(params);
  const defaults = Object.fromEntries(normParams.map((p) => [p.name, p.default]));
  const regMap = Object.fromEntries(getRegisteredIndicators().map((i) => [i.name, i]));
  const indicatorCodes = [];
  for (const indName of linked_indicators) {
    const indDef = regMap[indName];
    if (indDef) {
      indicatorCodes.push({ name: indDef.name, code: indDef.code, params: indicatorDefaults(indDef) });
    }
  }
  const { candles, source } = syntax_only ? { candles: [], source: 'syntax check only' } : resolveValidationCandles(dataset_id);
  const sandboxRes = await runPythonSandbox({
    mode: 'strategy',
    code: String(code),
    candles,
    params: defaults,
    indicator_codes: indicatorCodes,
    syntax_only,
  });
  if (!sandboxRes.ok || syntax_only) {
    return { ...sandboxRes, data_source: source, metrics: null };
  }
  const { runBacktest, computeMetrics } = await import('./server_engine.js');
  const bt = runBacktest(candles, sandboxRes.signals, {
    min_votes: 1,
    direction,
    stop_loss_pct: sandboxRes.stop_loss_pct ?? stop_loss_pct,
    target_pct: sandboxRes.target_pct ?? target_pct,
    exit_signals: sandboxRes.exit_signals,
  });
  const metrics = computeMetrics(bt.trades, bt.equity, DEFAULT_BACKTEST.initial_capital, bt.final_capital, MIN_TRADES, bt.skipped_entries);
  return {
    ...sandboxRes,
    signals: undefined,
    exit_signals: undefined,
    data_source: source,
    metrics,
  };
}

app.post('/api/studio/strategies/validate', async (req, res) => {
  try {
    const body = req.body || {};
    if (!body.code || !String(body.code).trim()) return errDetail(res, 400, 'Python strategy code cannot be empty.');
    const result = await executeStrategyValidation(body);
    res.json(result);
  } catch (err) {
    errDetail(res, 400, err.message);
  }
});

// Explicit Update Strategy flow: validate -> save -> register -> update Research Agent catalogue -> NEVER auto-start optimization
app.post('/api/studio/strategies/:id/update', async (req, res) => {
  try {
    const id = Number(req.params.id);
    const idx = store.state.studioStrategies.findIndex((s) => s.id === id);
    if (idx < 0) return errDetail(res, 404, 'Strategy not found');
    const prev = store.state.studioStrategies[idx];
    const meta = validateStrategyMeta(req.body || {}, id);
    const datasetId = req.body?.dataset_id || null;

    const validation = await executeStrategyValidation({
      ...meta,
      dataset_id: datasetId,
      syntax_only: false,
    });

    if (!validation.ok) {
      const failed = {
        ...prev,
        ...meta,
        status: 'error',
        registered: false,
        available: false,
        status_reason: `${validation.error?.type || 'ValidationError'} (line ${validation.error?.line || 1}): ${validation.error?.message}`,
        last_validated_at: utcnow(),
        last_validation: validation,
        updated_at: utcnow(),
      };
      store.state.studioStrategies[idx] = failed;
      store.save();
      return res.status(400).json({
        detail: failed.status_reason,
        validation: failed.last_validation,
        strategy: failed,
      });
    }

    const registeredItem = {
      ...prev,
      ...meta,
      status: 'registered',
      registered: true,
      available: true,
      status_reason: 'ok (validated in isolated sandbox)',
      last_validated_at: utcnow(),
      last_validation: validation,
      updated_at: utcnow(),
    };
    store.state.studioStrategies[idx] = registeredItem;
    addAgentLog(
      'research_agent',
      'info',
      `Strategy '${meta.name}' v${meta.version} registered in dynamic catalogue (${getRegisteredStrategies().length} total available). No optimization started automatically.`
    );
    store.save();

    res.json({
      strategy: registeredItem,
      validation,
      catalogue_size: getRegisteredStrategies().length,
      auto_started_optimization: false,
      message: `Strategy '${meta.name}' v${meta.version} validated, saved, and registered in the Research Agent catalogue. No optimization run was started automatically.`,
    });
  } catch (err) {
    errDetail(res, 400, err.message);
  }
});

// ---------------------------------------------------------------------------
// Strategy options & describe (without hardcoded templates)
// ---------------------------------------------------------------------------
app.get('/api/strategy/templates', (_req, res) => {
  res.json([]);
});

app.get('/api/strategy/options', (_req, res) => {
  res.json({
    directions: ['both', 'long', 'short'],
    stop_loss_choices: [0, 0.2, 0.3, 0.5, 0.8, 1.0],
    target_choices: [0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0],
    max_indicators: null, // unlimited
    defaults: { ...DEFAULT_BACKTEST, min_trades: MIN_TRADES },
  });
});

app.post('/api/strategy/describe', (req, res) => {
  try {
    const spec = buildStrategySpec(req.body || {});
    res.json({ hash: specHash(spec), description: describeSpec(spec), spec });
  } catch (err) {
    errDetail(res, 400, err.message);
  }
});

// ---------------------------------------------------------------------------
// Backtest & Out-of-Sample
// ---------------------------------------------------------------------------
function makeBacktestConfig(body) {
  return {
    initial_capital: Number(body.initial_capital || DEFAULT_BACKTEST.initial_capital),
    position_pct: Number(body.position_pct || DEFAULT_BACKTEST.position_pct),
    lot_size: DEFAULT_BACKTEST.lot_size,
    brokerage_per_order: body.brokerage_per_order !== undefined ? Number(body.brokerage_per_order) : DEFAULT_BACKTEST.brokerage_per_order,
    charges_pct: body.charges_pct !== undefined ? Number(body.charges_pct) : DEFAULT_BACKTEST.charges_pct,
    slippage_points: body.slippage_points !== undefined ? Number(body.slippage_points) : DEFAULT_BACKTEST.slippage_points,
    no_entry_after_minute: DEFAULT_BACKTEST.no_entry_after_minute,
    square_off_minute: DEFAULT_BACKTEST.square_off_minute,
  };
}

app.post('/api/backtest', async (req, res) => {
  try {
    const body = req.body || {};
    findDataset(body.dataset_id);
    if (!['train', 'validation'].includes(body.split || 'validation')) {
      return errDetail(res, 400, 'use /api/backtest/oos for the out-of-sample split');
    }
    const spec = body.strategy?.strategy_ids?.length
      ? {
          strategy_name: body.strategy.strategy_name || 'Custom Strategy',
          strategy_ids: body.strategy.strategy_ids,
          strategy_params: body.strategy.strategy_params || {},
          indicators: body.strategy.indicators || [],
          min_votes: body.strategy.min_votes || 1,
          direction: body.strategy.direction || 'both',
          stop_loss_pct: Number(body.strategy.stop_loss_pct ?? 0.5),
          target_pct: Number(body.strategy.target_pct ?? 1.0),
        }
      : buildStrategySpec(body.strategy || {});

    const cfg = makeBacktestConfig(body);
    const { bt, metrics, slice } = await evaluateSpecFull(body.dataset_id, spec, body.split || 'validation', { backtestCfg: cfg });
    const chart = resampleCandles(slice, body.chart_timeframe || '5min').slice(-1500);
    res.json({
      strategy: { hash: specHash(spec), description: describeSpec(spec), spec },
      split: body.split || 'validation',
      metrics,
      trades: bt.trades,
      equity: makeEquityPayload(bt, metrics),
      candles: chart,
    });
  } catch (err) {
    errDetail(res, err.status || 400, err.message);
  }
});

app.post('/api/backtest/oos', async (req, res) => {
  try {
    const body = req.body || {};
    findDataset(body.dataset_id);
    const spec = body.strategy?.strategy_ids?.length
      ? body.strategy
      : buildStrategySpec(body.strategy || {});
    const hash = specHash(spec);
    const existing = store.state.oosTests.find((o) => o.strategy_hash === hash && o.dataset_id === Number(body.dataset_id));
    if (existing) {
      return res.json({ already_run: true, created_at: existing.created_at, metrics: existing.metrics });
    }
    const cfg = makeBacktestConfig(body);
    const { metrics } = await evaluateSpecFull(body.dataset_id, spec, 'oos', { backtestCfg: cfg });
    const rec = { strategy_hash: hash, dataset_id: Number(body.dataset_id), created_at: utcnow(), metrics };
    store.state.oosTests.push(rec);
    store.save();
    res.json({ already_run: false, metrics });
  } catch (err) {
    errDetail(res, err.status || 400, err.message);
  }
});

// ---------------------------------------------------------------------------
// Agents
// ---------------------------------------------------------------------------
app.get('/api/agents', (_req, res) => {
  res.json(agentsStatus());
});

app.post('/api/agents/start', (req, res) => {
  try {
    findDataset(req.body?.dataset_id);
    res.json(startAutonomousAgents(req.body));
  } catch (err) {
    errDetail(res, err.status || 400, err.message);
  }
});

app.post('/api/agents/:name/:action', (req, res) => {
  try {
    res.json(controlAgent(req.params.name, req.params.action));
  } catch (err) {
    errDetail(res, 400, err.message);
  }
});

app.get('/api/agents/:name/logs', (req, res) => {
  const after = Number(req.query.after || 0);
  const limit = Math.min(1000, Number(req.query.limit || 200));
  const rows = store.state.agentLogs
    .filter((l) => l.agent === req.params.name && l.id > after)
    .slice(-limit);
  res.json(rows);
});

app.get('/api/agents/evaluation/snapshot', (_req, res) => {
  res.json(store.state.agentCheckpoints.evaluation_agent?.state || buildEvaluationSnapshot());
});

// ---------------------------------------------------------------------------
// Optimization
// ---------------------------------------------------------------------------
app.get('/api/optimization/options', (_req, res) => {
  res.json({
    samplers: ['tpe', 'ga', 'forward', 'backward', 'hill', 'nsga2', 'random', 'grid'],
    pruners: ['median', 'none'],
    objectives: ['net_profit', 'sharpe_ratio', 'profit_factor', 'recovery_factor', 'min_drawdown', 'multi_objective'],
    max_indicators: null,
    catalogue: {
      indicators: getRegisteredIndicators().length,
      strategies: getRegisteredStrategies().length,
    },
    notes: {
      grid: 'exhaustive over discretised choices; size is capped by trial budget',
      cmaes: 'continuous-only sampler; categorical choices fall back to independent sampling',
      nsga2: 'multi-objective: net profit and drawdown; pruning is disabled',
    },
  });
});

app.get('/api/optimization/runs', (_req, res) => {
  res.json(getOptimizationRuns());
});

app.post('/api/optimization/start', (req, res) => {
  try {
    findDataset(req.body?.dataset_id);
    res.json(startOptimizationStudy(req.body));
  } catch (err) {
    errDetail(res, err.status || 400, err.message);
  }
});

app.post('/api/optimization/:study/:action', (req, res) => {
  try {
    res.json(controlOptimizationStudy(req.params.study, req.params.action));
  } catch (err) {
    errDetail(res, 400, err.message);
  }
});

app.get('/api/optimization/studies/:study', (req, res) => {
  const name = req.params.study;
  const run = store.state.optimizationRuns.find((r) => r.study_name === name);
  if (!run) return errDetail(res, 404, `study '${name}' not found`);
  const trials = store.state.experiments.filter((e) => e.study_name === name);
  const complete = trials.filter((t) => t.status === 'completed');
  const best = complete.length
    ? complete.reduce((a, b) => ((b.metrics?.net_profit ?? -1e12) > (a.metrics?.net_profit ?? -1e12) ? b : a))
    : null;
  res.json({
    study_name: name,
    trials: trials.length,
    states: {
      complete: complete.length,
      pruned: trials.filter((t) => t.status === 'pruned').length,
      failed: trials.filter((t) => t.status === 'failed').length,
      running: 0,
    },
    best: best ? { value: best.metrics.net_profit, params: describeSpec(best.spec), strategy_hash: best.strategy_hash } : null,
  });
});

// ---------------------------------------------------------------------------
// Results & Experiments
// ---------------------------------------------------------------------------
app.get('/api/results/rankings', (req, res) => {
  try {
    const datasetId = Number(req.query.dataset_id);
    findDataset(datasetId);
    const top = Math.min(200, Math.max(1, Number(req.query.top || 50)));
    const split = String(req.query.split || 'train');
    const out = {};
    for (const key of Object.keys(RANKINGS)) out[key] = getLeaderboard(key, datasetId, { top, split });
    res.json(out);
  } catch (err) {
    errDetail(res, err.status || 400, err.message);
  }
});

app.get('/api/results/rankings/:key', (req, res) => {
  try {
    const datasetId = Number(req.query.dataset_id);
    findDataset(datasetId);
    const top = Math.min(200, Math.max(1, Number(req.query.top || 50)));
    const split = String(req.query.split || 'train');
    res.json(getLeaderboard(req.params.key, datasetId, { top, split }));
  } catch (err) {
    errDetail(res, err.status || 400, err.message);
  }
});

app.get('/api/results/strategies/:hash', (req, res) => {
  const datasetId = Number(req.query.dataset_id);
  const rows = store.state.experiments.filter(
    (e) => e.strategy_hash === req.params.hash && e.dataset_id === datasetId && e.status === 'completed'
  );
  if (!rows.length) return errDetail(res, 404, 'strategy not found');
  const validations = store.state.validations.filter((v) => v.strategy_hash === req.params.hash && v.dataset_id === datasetId).reverse();
  const oos = store.state.oosTests.find((o) => o.strategy_hash === req.params.hash && o.dataset_id === datasetId) || null;
  const dsr = store.state.statisticalTests.filter((s) => s.strategy_hash === req.params.hash && s.dataset_id === datasetId && s.kind === 'dsr').slice(-1)[0] || null;
  res.json({
    hash: req.params.hash,
    spec: rows[0].spec,
    experiments: rows.map((r) => ({ split: r.split, metrics: r.metrics, created_at: r.created_at, agent: r.agent, study: r.study_name })),
    validations,
    out_of_sample: oos ? { metrics: oos.metrics, created_at: oos.created_at } : null,
    dsr: dsr?.result || null,
  });
});

app.get('/api/experiments', (req, res) => {
  const datasetId = req.query.dataset_id ? Number(req.query.dataset_id) : null;
  const study = req.query.study || null;
  const status = req.query.status || null;
  const limit = Math.min(1000, Number(req.query.limit || 100));
  const offset = Number(req.query.offset || 0);

  let rows = [...store.state.experiments].reverse();
  if (datasetId !== null) rows = rows.filter((r) => r.dataset_id === datasetId);
  if (study) rows = rows.filter((r) => r.study_name === study);
  if (status) rows = rows.filter((r) => r.status === status);
  const total = rows.length;
  const paged = rows.slice(offset, offset + limit).map((r) => ({
    ...r,
    description: describeSpec(r.spec),
  }));
  res.json({ total, rows: paged });
});

app.get('/api/experiments/:id', (req, res) => {
  const row = store.state.experiments.find((e) => e.id === Number(req.params.id));
  if (!row) return errDetail(res, 404, 'experiment not found');
  res.json(row);
});

// ---------------------------------------------------------------------------
// Analytics & Statistics
// ---------------------------------------------------------------------------
app.get('/api/analytics/optuna/:study/:chart', (req, res) => {
  const { study, chart } = req.params;
  const trials = store.state.experiments
    .filter((e) => e.study_name === study && e.status === 'completed' && e.metrics)
    .sort((a, b) => (a.trial_number || a.id) - (b.trial_number || b.id));
  if (!trials.length) return errDetail(res, 400, `Study '${study}' has no completed trials yet`);

  if (chart === 'history') {
    const x = trials.map((t, i) => t.trial_number ?? i + 1);
    const y = trials.map((t) => t.metrics.net_profit);
    let bestSoFar = -Infinity;
    const bestCurve = y.map((v) => { bestSoFar = Math.max(bestSoFar, v); return bestSoFar; });
    return res.json({
      data: [
        { type: 'scatter', mode: 'markers', name: 'Trial Net Profit', x, y, marker: { color: '#4f8cff', size: 7 } },
        { type: 'scatter', mode: 'lines', name: 'Best Value', x, y: bestCurve, line: { color: '#22c55e', width: 2 } },
      ],
      layout: { title: `Optimization History (${study})`, xaxis: { title: 'Trial' }, yaxis: { title: 'Net Profit' } },
    });
  }
  if (chart === 'pareto') {
    return res.json({
      data: [{
        type: 'scatter', mode: 'markers', name: 'Trials',
        x: trials.map((t) => t.metrics.max_drawdown_pct),
        y: trials.map((t) => t.metrics.net_profit),
        text: trials.map((t) => describeSpec(t.spec)),
        marker: { color: '#4f8cff', size: 8 },
      }],
      layout: { title: 'Profit vs Max Drawdown %', xaxis: { title: 'Max Drawdown %' }, yaxis: { title: 'Net Profit' } },
    });
  }
  if (chart === 'importance') {
    const counts = new Map();
    for (const t of trials) {
      for (const ind of t.spec?.indicators || []) {
        const list = counts.get(ind.name) || [];
        list.push(t.metrics.net_profit);
        counts.set(ind.name, list);
      }
    }
    const entries = [...counts.entries()].map(([name, vals]) => [name, vals.reduce((a, b) => a + b, 0) / vals.length]);
    return res.json({
      data: [{ type: 'bar', orientation: 'h', y: entries.map((e) => e[0]), x: entries.map((e) => e[1]), marker: { color: '#4f8cff' } }],
      layout: { title: 'Mean Net Profit by Indicator Presence', xaxis: { title: 'Mean Net Profit' } },
    });
  }
  // Fallback slice/contour/parallel over trial numbers & sharpe
  return res.json({
    data: [{
      type: 'scatter', mode: 'markers',
      x: trials.map((t) => t.metrics.trades),
      y: trials.map((t) => t.metrics.sharpe_ratio ?? 0),
      text: trials.map((t) => describeSpec(t.spec)),
      marker: { color: trials.map((t) => t.metrics.net_profit), colorscale: 'Viridis', size: 9, showscale: true },
    }],
    layout: { title: `${chart} view (${study})`, xaxis: { title: 'Trades' }, yaxis: { title: 'Sharpe Ratio' } },
  });
});

app.get('/api/analytics/equity', async (req, res) => {
  try {
    const datasetId = Number(req.query.dataset_id);
    const hash = String(req.query.strategy_hash || '');
    const split = String(req.query.split || 'validation');
    if (!['train', 'validation'].includes(split)) return errDetail(res, 400, 'equity analytics supports train and validation only');
    const exp = store.state.experiments.find((e) => e.strategy_hash === hash && e.dataset_id === datasetId);
    if (!exp) return errDetail(res, 404, 'strategy not found');
    const { bt, metrics } = await evaluateSpecFull(datasetId, exp.spec, split);
    res.json({ split, ...makeEquityPayload(bt, metrics), trades: bt.trades });
  } catch (err) {
    errDetail(res, 400, err.message);
  }
});

app.get('/api/analytics/walk_forward', (req, res) => {
  const datasetId = Number(req.query.dataset_id);
  const seen = new Set();
  const out = [];
  for (const v of [...store.state.validations].reverse()) {
    if (v.dataset_id !== datasetId || seen.has(v.strategy_hash)) continue;
    seen.add(v.strategy_hash);
    out.push(v);
  }
  res.json(out);
});

app.get('/api/analytics/train_vs_validation', (req, res) => {
  const datasetId = Number(req.query.dataset_id);
  const vals = new Map();
  for (const v of store.state.validations) {
    if (v.dataset_id === datasetId) vals.set(v.strategy_hash, v);
  }
  const out = [];
  const seen = new Set();
  for (const e of [...store.state.experiments].reverse()) {
    if (e.dataset_id !== datasetId || e.split !== 'train' || e.status !== 'completed' || seen.has(e.strategy_hash)) continue;
    const v = vals.get(e.strategy_hash);
    if (!v) continue;
    seen.add(e.strategy_hash);
    out.push({
      strategy_hash: e.strategy_hash,
      train_net_profit: e.metrics.net_profit,
      validation_net_profit: v.validation_net_profit,
      validated: Boolean(v.validated),
    });
  }
  res.json(out);
});

app.post('/api/statistics/dsr', (req, res) => {
  try {
    findDataset(req.body?.dataset_id);
    if ((req.body?.split || 'train') !== 'train') {
      return errDetail(res, 400, 'DSR is computed on the train split only');
    }
    res.json(computeDsrForDataset(req.body.dataset_id, 'train'));
  } catch (err) {
    errDetail(res, err.status || 400, err.message);
  }
});

app.get('/api/statistics', (req, res) => {
  const datasetId = Number(req.query.dataset_id);
  const kind = req.query.kind || null;
  let rows = store.state.statisticalTests.filter((s) => s.dataset_id === datasetId);
  if (kind) rows = rows.filter((s) => s.kind === kind);
  res.json(rows);
});

// ---------------------------------------------------------------------------
// Frontend Hosting (Vite dev middleware or built dist) on 0.0.0.0:3000
// ---------------------------------------------------------------------------
async function startServer() {
  if (!IS_PROD) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      root: path.join(__dirname, 'frontend'),
      configFile: path.join(__dirname, 'frontend', 'vite.config.js'),
      server: {
        middlewareMode: true,
        hmr: false,
      },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.join(__dirname, 'frontend', 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distDir, 'index.html'));
    });
  }

  app.listen(PORT, HOST, () => {
    console.log(`[IndicatorLab] Server listening on http://${HOST}:${PORT} (${IS_PROD ? 'production' : 'development'})`);
  });
}

startServer();
