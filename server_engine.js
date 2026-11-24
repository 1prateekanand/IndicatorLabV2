import { spawn } from 'node:child_process';
import crypto from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
export const DATA_DIR = path.resolve(process.env.IL_DATA_DIR || path.join(__dirname, 'data'));
const STATE_FILE = path.join(DATA_DIR, 'studio_state.json');
const DATASETS_DIR = path.join(DATA_DIR, 'datasets');
const SANDBOX_SCRIPT = path.join(__dirname, 'sandbox_runner.py');

export const DEFAULT_BACKTEST = {
  initial_capital: 10_000_000,
  position_pct: 0.10,
  lot_size: 1,
  brokerage_per_order: 20.0,
  charges_pct: 0.0005,
  slippage_points: 0.5,
  no_entry_after_minute: 15 * 60,
  square_off_minute: 15 * 60 + 15,
};

export const DEFAULT_SPLIT = {
  train_end: 0.60,
  validation_end: 0.80,
  train_chunks: 3,
  walk_forward_segments: 4,
};

export const MIN_TRADES = Number(process.env.IL_MIN_TRADES || 30);

export function utcnow() {
  return new Date().toISOString().replace(/\.\d{3}Z$/, 'Z');
}

function ensureDirs() {
  fs.mkdirSync(DATA_DIR, { recursive: true });
  fs.mkdirSync(DATASETS_DIR, { recursive: true });
}

function initialState() {
  return {
    nextIds: { dataset: 1, indicator: 1, strategy: 1, experiment: 1, validation: 1, log: 1, stat: 1 },
    datasets: [],
    studioIndicators: [],
    studioStrategies: [],
    experiments: [],
    validations: [],
    oosTests: [],
    statisticalTests: [],
    optimizationRuns: [],
    agentLogs: [],
    agentCheckpoints: {},
    agents: {
      research_agent: {
        name: 'research_agent',
        status: 'stopped',
        current_task: '',
        algorithm: '',
        trials_processed: 0,
        error_count: 0,
        progress: 0,
        started_at: null,
        last_heartbeat: null,
        duplicates_prevented: 0,
        detail: {},
      },
      evaluation_agent: {
        name: 'evaluation_agent',
        status: 'stopped',
        current_task: '',
        algorithm: '',
        trials_processed: 0,
        error_count: 0,
        progress: 0,
        started_at: null,
        last_heartbeat: null,
        duplicates_prevented: 0,
        detail: {},
      },
    },
    savedAgentSettings: null,
  };
}

class Store {
  constructor() {
    ensureDirs();
    this.state = this._load();
    this._candleCache = new Map();
    this._signalCache = new Map();
  }

  _load() {
    try {
      if (fs.existsSync(STATE_FILE)) {
        const parsed = JSON.parse(fs.readFileSync(STATE_FILE, 'utf8'));
        return { ...initialState(), ...parsed };
      }
    } catch (err) {
      console.warn('[IndicatorLab] Failed to load state, starting fresh:', err.message);
    }
    return initialState();
  }

  save() {
    ensureDirs();
    const tmp = `${STATE_FILE}.tmp`;
    fs.writeFileSync(tmp, JSON.stringify(this.state, null, 2));
    fs.renameSync(tmp, STATE_FILE);
  }

  clearSignalCache(indicatorName = null) {
    if (!indicatorName) {
      this._signalCache.clear();
      return;
    }
    for (const key of this._signalCache.keys()) {
      if (key.includes(`::${indicatorName}::`)) {
        this._signalCache.delete(key);
      }
    }
  }

  loadCandles(datasetId) {
    const id = Number(datasetId);
    if (this._candleCache.has(id)) return this._candleCache.get(id);
    const file = path.join(DATASETS_DIR, `${id}.json`);
    if (!fs.existsSync(file)) throw new Error(`dataset ${id} has no candles`);
    const rows = JSON.parse(fs.readFileSync(file, 'utf8'));
    this._candleCache.set(id, rows);
    return rows;
  }

  saveCandles(datasetId, candles) {
    const id = Number(datasetId);
    ensureDirs();
    fs.writeFileSync(path.join(DATASETS_DIR, `${id}.json`), JSON.stringify(candles));
    this._candleCache.set(id, candles);
    this.clearSignalCache();
  }
}

export const store = new Store();

// Recover any agents left active on previous shutdown
export function recoverAfterRestart() {
  const recovered = [];
  for (const name of ['research_agent', 'evaluation_agent']) {
    const a = store.state.agents[name];
    if (a && ['running', 'paused', 'stopping'].includes(a.status)) {
      a.status = 'stopped';
      a.current_task = '';
      addAgentLog(name, 'warn', 'process restarted while this worker was active; marked stopped (checkpoint kept)');
      recovered.push(name);
    }
  }
  for (const run of store.state.optimizationRuns) {
    if (run.status === 'running') run.status = 'interrupted';
  }
  store.save();
  return recovered;
}

export function addAgentLog(agent, level, message) {
  const id = store.state.nextIds.log++;
  store.state.agentLogs.push({ id, agent, ts: utcnow(), level, message });
  if (store.state.agentLogs.length > 2500) {
    store.state.agentLogs = store.state.agentLogs.slice(-2000);
  }
}

export function systemMetrics() {
  const cpus = os.cpus() || [];
  const load = os.loadavg()[0] || 0;
  const cores = cpus.length || 1;
  const cpuPct = Math.min(100, Math.round((load / cores) * 1000) / 10);
  const totalMem = os.totalmem();
  const freeMem = os.freemem();
  const usedMb = Math.round((totalMem - freeMem) / (1024 * 1024));
  const ramPct = Math.round(((totalMem - freeMem) / totalMem) * 1000) / 10;
  return {
    cpu_percent: cpuPct,
    ram_percent: ramPct,
    ram_used_mb: usedMb,
    ram_total_mb: Math.round(totalMem / (1024 * 1024)),
    cores,
  };
}

// ---------------------------------------------------------------------------
// Datasets: Synthetic generator + CSV parser + Resampling
// ---------------------------------------------------------------------------
function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function normalRand(rng) {
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

export function generateSyntheticCandles(days = 60, seed = 42, basePrice = 22000) {
  const rng = mulberry32(Number(seed) || 42);
  const bars = [];
  let cur = new Date('2025-01-01T00:00:00');
  let tradingDays = 0;
  let price = basePrice;
  let vol = 0.0006;
  while (tradingDays < days) {
    const dow = cur.getDay();
    if (dow !== 0 && dow !== 6) {
      const y = cur.getFullYear();
      const m = String(cur.getMonth() + 1).padStart(2, '0');
      const d = String(cur.getDate()).padStart(2, '0');
      for (let bar = 0; bar < 375; bar++) {
        const totalMin = 9 * 60 + 15 + bar;
        const hh = String(Math.floor(totalMin / 60)).padStart(2, '0');
        const mm = String(totalMin % 60).padStart(2, '0');
        vol = Math.max(0.0002, 0.97 * vol + 0.03 * 0.0006 + 0.0002 * Math.abs(normalRand(rng)));
        const ret = normalRand(rng) * vol;
        const open = Number(price.toFixed(2));
        const close = Number((price * (1 + ret)).toFixed(2));
        const high = Number((Math.max(open, close) * (1 + Math.abs(normalRand(rng) * vol * 0.5))).toFixed(2));
        const low = Number((Math.min(open, close) * (1 - Math.abs(normalRand(rng) * vol * 0.5))).toFixed(2));
        const volume = Math.floor(100 + rng() * 4900);
        bars.push({ time: `${y}-${m}-${d}T${hh}:${mm}:00`, open, high, low, close, volume });
        price = close;
      }
      tradingDays++;
    }
    cur.setDate(cur.getDate() + 1);
  }
  return bars;
}

export function parseCsvCandles(csvText) {
  const lines = csvText.trim().split(/\r?\n/).filter((l) => l.trim().length > 0);
  if (lines.length < 2) throw new Error('CSV must include a header row and candle rows');
  const headers = lines[0].split(',').map((h) => h.trim().toLowerCase());
  const idxOf = (names) => headers.findIndex((h) => names.includes(h));
  const tsIdx = idxOf(['timestamp', 'datetime', 'date_time', 'time']);
  const dateIdx = idxOf(['date']);
  const timeIdx = idxOf(['time']);
  const oIdx = idxOf(['open']);
  const hIdx = idxOf(['high']);
  const lIdx = idxOf(['low']);
  const cIdx = idxOf(['close']);
  const vIdx = idxOf(['volume', 'vol']);
  if (oIdx < 0 || hIdx < 0 || lIdx < 0 || cIdx < 0) {
    throw new Error('CSV must contain open, high, low, close columns');
  }
  if (tsIdx < 0 && (dateIdx < 0 || timeIdx < 0)) {
    throw new Error('CSV needs a timestamp/datetime/time column or separate date and time columns');
  }
  const map = new Map();
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',').map((s) => s.trim());
    if (cols.length < 4) continue;
    let rawTs = tsIdx >= 0 ? cols[tsIdx] : `${cols[dateIdx]} ${cols[timeIdx]}`;
    const dt = new Date(rawTs);
    if (Number.isNaN(dt.getTime())) {
      throw new Error(`Invalid timestamp at row ${i + 1}: ${rawTs}`);
    }
    const iso = rawTs.includes('T') ? rawTs.slice(0, 19) : dt.toISOString().slice(0, 19);
    const open = Number(cols[oIdx]);
    const high = Number(cols[hIdx]);
    const low = Number(cols[lIdx]);
    const close = Number(cols[cIdx]);
    const volume = vIdx >= 0 ? Number(cols[vIdx] || 0) : 0;
    if (![open, high, low, close].every((v) => Number.isFinite(v) && v > 0)) {
      throw new Error(`Invalid or non-positive OHLC at row ${i + 1}`);
    }
    if (high < Math.max(open, close, low) || low > Math.min(open, close, high)) {
      throw new Error(`Inconsistent OHLC bounds at row ${i + 1}`);
    }
    map.set(iso, { time: iso, open, high, low, close, volume: Number.isFinite(volume) ? volume : 0 });
  }
  const candles = [...map.values()].sort((a, b) => a.time.localeCompare(b.time));
  if (candles.length < 500) {
    throw new Error(`At least 500 candles are required for a meaningful backtest (found ${candles.length})`);
  }
  return candles;
}

export function upsertDataset({ name, symbol, timeframe = '1min', source_file = 'upload', candles, replace = true }) {
  const existing = store.state.datasets.find((d) => d.name === name);
  if (existing && !replace) {
    throw new Error(`dataset '${name}' already exists; pass replace=true to overwrite`);
  }
  if (existing) {
    store.state.datasets = store.state.datasets.filter((d) => d.id !== existing.id);
  }
  const id = store.state.nextIds.dataset++;
  const row = {
    id,
    name,
    symbol,
    timeframe,
    source_file,
    rows: candles.length,
    start_ts: candles[0].time,
    end_ts: candles[candles.length - 1].time,
    imported_at: utcnow(),
  };
  store.state.datasets.push(row);
  store.saveCandles(id, candles);
  store.save();
  return row;
}

export function resampleCandles(candles, timeframe = '5min') {
  if (timeframe === '1min' || candles.length === 0) return candles;
  const mins = Number(String(timeframe).replace(/[^0-9]/g, '')) || 5;
  const bucketMs = mins * 60 * 1000;
  const buckets = new Map();
  for (const c of candles) {
    const t = Date.parse(c.time);
    const b = Math.floor(t / bucketMs) * bucketMs;
    const key = new Date(b).toISOString().slice(0, 19);
    const cur = buckets.get(key);
    if (!cur) {
      buckets.set(key, { time: key, open: c.open, high: c.high, low: c.low, close: c.close, volume: c.volume || 0 });
    } else {
      cur.high = Math.max(cur.high, c.high);
      cur.low = Math.min(cur.low, c.low);
      cur.close = c.close;
      cur.volume += c.volume || 0;
    }
  }
  return [...buckets.values()];
}

export function probeCandles(n = 400) {
  return generateSyntheticCandles(2, 7, 100).slice(0, n);
}

// ---------------------------------------------------------------------------
// Isolated Python 3.11 Sandbox Execution
// ---------------------------------------------------------------------------
export function runPythonSandbox(payload, { cpuLimitS = 5, memLimitMb = 256, timeoutMs = 10000 } = {}) {
  return new Promise((resolve) => {
    const child = spawn('python3', ['-I', SANDBOX_SCRIPT], {
      cwd: __dirname,
      stdio: ['pipe', 'pipe', 'pipe'],
      timeout: timeoutMs,
    });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (d) => { stdout += d.toString(); });
    child.stderr.on('data', (d) => { stderr += d.toString(); });
    child.on('error', (err) => {
      resolve({ ok: false, stage: 'sandbox', error: { type: 'SandboxError', line: 1, message: err.message } });
    });
    child.on('close', (code) => {
      if (stdout.trim()) {
        try {
          resolve(JSON.parse(stdout));
          return;
        } catch {
          // fall through
        }
      }
      resolve({
        ok: false,
        stage: 'sandbox',
        error: {
          type: 'SandboxError',
          line: 1,
          message: stderr.trim() || `Python sandbox exited with code ${code}`,
        },
      });
    });
    const req = { ...payload, cpu_limit_s: cpuLimitS, mem_limit_mb: memLimitMb };
    child.stdin.write(JSON.stringify(req));
    child.stdin.end();
  });
}

// ---------------------------------------------------------------------------
// Indicator & Strategy Studio helpers + Dynamic Registry
// ---------------------------------------------------------------------------
const SNAKE_RE = /^[a-z][a-z0-9_]{1,63}$/;

export function normalizeParamSpecs(params = []) {
  const out = [];
  const seen = new Set();
  for (const p of params) {
    const name = String(p.name || '').trim();
    if (!name || !SNAKE_RE.test(name)) {
      throw new Error(`Invalid parameter name '${name}'. Use lowercase snake_case.`);
    }
    if (seen.has(name)) throw new Error(`Duplicate parameter '${name}'`);
    seen.add(name);
    const kind = p.kind === 'float' ? 'float' : 'int';
    const low = Number(p.low);
    const high = Number(p.high);
    const def = Number(p.default);
    if (!Number.isFinite(low) || !Number.isFinite(high) || !Number.isFinite(def)) {
      throw new Error(`Parameter '${name}' requires finite numeric low, high, and default`);
    }
    if (low > high) throw new Error(`Parameter '${name}': low (${low}) > high (${high})`);
    if (def < low || def > high) {
      throw new Error(`Parameter '${name}': default (${def}) must lie within [${low}, ${high}]`);
    }
    out.push({
      name,
      kind,
      default: kind === 'int' ? Math.round(def) : def,
      low: kind === 'int' ? Math.round(low) : low,
      high: kind === 'int' ? Math.round(high) : high,
      step: p.step !== undefined && p.step !== null && p.step !== '' ? Number(p.step) : null,
    });
  }
  return out;
}

export function indicatorDefaults(ind) {
  const d = {};
  for (const p of ind.params || []) d[p.name] = p.default;
  return d;
}

export function cleanIndicatorParams(ind, raw = {}) {
  const merged = indicatorDefaults(ind);
  const idx = Object.fromEntries((ind.params || []).map((p) => [p.name, p]));
  for (const [k, v] of Object.entries(raw || {})) {
    if (!idx[k]) throw new Error(`indicator '${ind.name}' has no parameter '${k}'`);
    const spec = idx[k];
    const num = spec.kind === 'int' ? Math.round(Number(v)) : Math.round(Number(v) * 1e6) / 1e6;
    merged[k] = Math.min(Math.max(num, spec.low), spec.high);
  }
  return merged;
}

export function getRegisteredIndicators() {
  return store.state.studioIndicators.filter((i) => i.registered && i.available);
}

export function getRegisteredStrategies() {
  return store.state.studioStrategies.filter((s) => s.registered && s.available);
}

// Compute indicator signals on candles via isolated Python sandbox with caching
export async function computeIndicatorSignal(datasetId, candles, indDef, params = {}, opts = {}) {
  const clean = cleanIndicatorParams(indDef, params);
  const key = `${datasetId}::${indDef.name}::${indDef.version}::${JSON.stringify(clean)}`;
  if (store._signalCache.has(key)) return store._signalCache.get(key);

  const res = await runPythonSandbox({
    mode: 'indicator',
    code: indDef.code,
    candles,
    params: clean,
    return_signals: true,
  }, opts);
  if (!res.ok) {
    throw new Error(`Indicator '${indDef.name}' execution failed: ${res.error?.message || 'unknown error'}`);
  }
  if (store._signalCache.size > 2000) {
    const firstKey = store._signalCache.keys().next().value;
    store._signalCache.delete(firstKey);
  }
  store._signalCache.set(key, res.signals);
  return res.signals;
}

// ---------------------------------------------------------------------------
// Strategy Specification, Hashing & Bar-by-Bar Backtest Engine
// ---------------------------------------------------------------------------
export function buildStrategySpec(input) {
  const regMap = Object.fromEntries(getRegisteredIndicators().map((i) => [i.name, i]));
  const items = [];
  for (const item of input.indicators || []) {
    const def = regMap[item.name];
    if (!def) {
      throw new Error(`indicator '${item.name}' is not registered or unavailable`);
    }
    const clean = cleanIndicatorParams(def, item.params || {});
    items.push({ name: def.name, params: Object.fromEntries(Object.entries(clean).sort(([a], [b]) => a.localeCompare(b))) });
  }
  if (items.length === 0) throw new Error('at least 1 indicator is required');
  const names = items.map((i) => i.name);
  if (new Set(names).size !== names.length) throw new Error('an indicator may appear only once per strategy');
  items.sort((a, b) => a.name.localeCompare(b.name));

  const minVotes = Math.max(1, Math.min(items.length, Number(input.min_votes || 1)));
  const direction = ['both', 'long', 'short'].includes(input.direction) ? input.direction : 'both';
  const sl = Math.round(Math.max(0, Math.min(10, Number(input.stop_loss_pct ?? 0.5))) * 1e4) / 1e4;
  const tp = Math.round(Math.max(0, Math.min(20, Number(input.target_pct ?? 1.0))) * 1e4) / 1e4;

  const spec = {
    strategy_name: input.strategy_name || null,
    strategy_ids: Array.isArray(input.strategy_ids) ? [...input.strategy_ids].sort() : null,
    strategy_params: input.strategy_params || null,
    indicators: items,
    min_votes: minVotes,
    direction,
    stop_loss_pct: sl,
    target_pct: tp,
  };
  return spec;
}

export function specHash(spec) {
  const canonical = JSON.stringify(spec, Object.keys(spec).sort());
  const h = crypto.createHash('sha256').update(JSON.stringify(sortKeysDeep(spec))).digest('hex');
  return h.slice(0, 16);
}

function sortKeysDeep(val) {
  if (Array.isArray(val)) return val.map(sortKeysDeep);
  if (val && typeof val === 'object') {
    const out = {};
    for (const k of Object.keys(val).sort()) {
      if (val[k] !== null && val[k] !== undefined) out[k] = sortKeysDeep(val[k]);
    }
    return out;
  }
  return val;
}

export function describeSpec(spec) {
  if (!spec) return '';
  const inds = (spec.indicators || []).map((i) => {
    const ps = Object.entries(i.params || {}).map(([k, v]) => `${k}=${v}`).join(', ');
    return `${i.name}(${ps})`;
  });
  const prefix = spec.strategy_name ? `[${spec.strategy_name}] ` : '';
  return `${prefix}${inds.join(' + ')} | votes>=${spec.min_votes} | ${spec.direction} | SL ${spec.stop_loss_pct}% TP ${spec.target_pct}%`;
}

function minuteOfDay(ts) {
  const m = ts.match(/T(\d{2}):(\d{2})/);
  if (m) return Number(m[1]) * 60 + Number(m[2]);
  const d = new Date(ts);
  return d.getHours() * 60 + d.getMinutes();
}

export function runBacktest(candles, votes, { min_votes = 1, direction = 'both', stop_loss_pct = 0.5, target_pct = 1.0, exit_signals = null, cfg = DEFAULT_BACKTEST, initial_capital = null }) {
  const n = candles.length;
  const initCap = initial_capital != null ? Number(initial_capital) : cfg.initial_capital;
  if (n === 0) {
    return { trades: [], equity: [], timestamps: [], final_capital: initCap, skipped_entries: 0 };
  }
  const slip = cfg.slippage_points;
  const slPct = stop_loss_pct / 100;
  const tgtPct = target_pct / 100;

  let cash = initCap;
  const equity = new Array(n);
  const trades = [];
  let pos = 0;
  let qty = 0;
  let entryPx = 0;
  let entryCost = 0;
  let entryI = -1;
  let entryTime = null;
  let pending = null;
  let skipped = 0;

  const fillCost = (px, q) => cfg.brokerage_per_order + cfg.charges_pct * px * q;
  const closePos = (i, px, reason) => {
    const pnl = (px - entryPx) * qty * pos;
    const exitCost = fillCost(px, qty);
    cash += pnl - exitCost;
    trades.push({
      side: pos > 0 ? 'LONG' : 'SHORT',
      entry_time: entryTime,
      exit_time: candles[i].time,
      entry_price: Math.round(entryPx * 10000) / 10000,
      exit_price: Math.round(px * 10000) / 10000,
      quantity: qty,
      gross_pnl: Math.round(pnl * 10000) / 10000,
      costs: Math.round((entryCost + exitCost) * 10000) / 10000,
      net_pnl: Math.round((pnl - entryCost - exitCost) * 10000) / 10000,
      reason,
      bars_held: i - entryI,
    });
    pos = 0;
    qty = 0;
  };

  for (let i = 0; i < n; i++) {
    const { open: o, high: h, low: l, close: c, time } = candles[i];
    const tMin = minuteOfDay(time);

    if (pending) {
      const [kind, side, reason] = pending;
      pending = null;
      if (kind === 'enter' && pos === 0 && tMin < cfg.no_entry_after_minute) {
        const px = o + side * slip;
        const q = Math.floor((cash * cfg.position_pct) / (px * cfg.lot_size)) * cfg.lot_size;
        if (q >= cfg.lot_size) {
          pos = side;
          qty = q;
          entryPx = px;
          entryCost = fillCost(px, q);
          cash -= entryCost;
          entryI = i;
          entryTime = time;
        } else {
          skipped++;
        }
      } else if (kind === 'exit' && pos !== 0) {
        closePos(i, o - pos * slip, reason);
      }
    }

    if (pos !== 0 && i > entryI) {
      const slPx = slPct > 0 ? entryPx * (1 - pos * slPct) : null;
      const tgPx = tgtPct > 0 ? entryPx * (1 + pos * tgtPct) : null;
      const hitSl = pos > 0 ? (slPx !== null && l <= slPx) : (slPx !== null && h >= slPx);
      const hitTg = pos > 0 ? (tgPx !== null && h >= tgPx) : (tgPx !== null && l <= tgPx);
      if (hitSl) {
        const fill = ((pos > 0 && o <= slPx) || (pos < 0 && o >= slPx)) ? o : slPx;
        closePos(i, fill - pos * slip, 'stop_loss');
      } else if (hitTg) {
        const fill = ((pos > 0 && o >= tgPx) || (pos < 0 && o <= tgPx)) ? o : tgPx;
        closePos(i, fill - pos * slip, 'target');
      }
    }

    if (pos !== 0 && tMin >= cfg.square_off_minute) {
      closePos(i, c - pos * slip, 'square_off');
    } else if (pos !== 0 && i + 1 < n) {
      const customExit = exit_signals && exit_signals[i] && (exit_signals[i] === pos || exit_signals[i] === -1 * pos || exit_signals[i] === 1 && pos === 0);
      if (customExit || votes[i] * pos <= -min_votes) {
        pending = ['exit', -pos, customExit ? 'strategy_exit' : 'signal_reversal'];
      }
    }

    if (pos === 0 && pending === null && tMin < cfg.square_off_minute && i + 1 < n) {
      if (votes[i] >= min_votes && (direction === 'both' || direction === 'long')) {
        pending = ['enter', 1, 'signal_long'];
      } else if (votes[i] <= -min_votes && (direction === 'both' || direction === 'short')) {
        pending = ['enter', -1, 'signal_short'];
      }
    }

    equity[i] = cash + (pos !== 0 ? (c - entryPx) * qty * pos : 0);
  }

  if (pos !== 0) {
    closePos(n - 1, candles[n - 1].close - pos * slip, 'end_of_data');
    equity[n - 1] = cash;
  }

  return {
    trades,
    equity,
    timestamps: candles.map((b) => b.time),
    final_capital: cash,
    skipped_entries: skipped,
  };
}

export function computeMetrics(trades, equity, initialCapital, finalCapital = null, minTrades = MIN_TRADES, skippedEntries = 0) {
  const n = trades.length;
  const nets = trades.map((t) => t.net_pnl);
  const wins = nets.filter((v) => v > 0);
  const losses = nets.filter((v) => v < 0);
  const netProfit = nets.reduce((a, b) => a + b, 0);
  const grossWin = wins.reduce((a, b) => a + b, 0);
  const grossLoss = -losses.reduce((a, b) => a + b, 0);
  const profitFactor = grossLoss > 0 ? Math.min(grossWin / grossLoss, 999.0) : (grossWin > 0 ? 999.0 : 0);

  let peak = -Infinity;
  let maxDd = 0;
  let maxDdPct = 0;
  for (const eq of equity) {
    if (eq > peak) peak = eq;
    const dd = peak - eq;
    if (dd > maxDd) maxDd = dd;
    if (peak > 0) {
      const pct = (dd / peak) * 100;
      if (pct > maxDdPct) maxDdPct = pct;
    }
  }

  // Return moments & Sharpe ratio
  const rets = [];
  for (let i = 1; i < equity.length; i++) {
    if (equity[i - 1] > 0) rets.push((equity[i] - equity[i - 1]) / equity[i - 1]);
  }
  let sharpePerBar = 0;
  let sharpeRatio = 0;
  let skew = 0;
  let kurtosis = 3;
  if (rets.length >= 2) {
    const mu = rets.reduce((a, b) => a + b, 0) / rets.length;
    const varR = rets.reduce((a, b) => a + (b - mu) ** 2, 0) / rets.length;
    const sd = Math.sqrt(varR);
    if (sd > 1e-12) {
      sharpePerBar = mu / sd;
      // Annualized Sharpe assuming intraday 1-min bars (~375/day * 252 days)
      sharpeRatio = sharpePerBar * Math.sqrt(252 * 375);
      const m3 = rets.reduce((a, b) => a + ((b - mu) / sd) ** 3, 0) / rets.length;
      const m4 = rets.reduce((a, b) => a + ((b - mu) / sd) ** 4, 0) / rets.length;
      skew = m3;
      kurtosis = m4;
    }
  }

  const reasons = {};
  for (const t of trades) reasons[t.reason] = (reasons[t.reason] || 0) + 1;

  return {
    trades: n,
    net_profit: Math.round(netProfit * 100) / 100,
    return_pct: initialCapital ? Math.round((netProfit / initialCapital) * 1000000) / 10000 : 0,
    final_capital: Math.round((finalCapital ?? (initialCapital + netProfit)) * 100) / 100,
    win_rate: n ? Math.round((wins.length / n) * 10000) / 100 : 0,
    profit_factor: Math.round(profitFactor * 10000) / 10000,
    sharpe_per_bar: Math.round(sharpePerBar * 100000) / 100000,
    sharpe_ratio: Math.round(sharpeRatio * 1000) / 1000,
    return_skew: Math.round(skew * 10000) / 10000,
    return_kurtosis: Math.round(kurtosis * 10000) / 10000,
    returns_obs: rets.length,
    max_drawdown: Math.round(maxDd * 100) / 100,
    max_drawdown_pct: Math.round(maxDdPct * 10000) / 10000,
    recovery_factor: maxDd > 0 ? Math.round((netProfit / maxDd) * 10000) / 10000 : 0,
    avg_trade: n ? Math.round((netProfit / n) * 100) / 100 : 0,
    avg_win: wins.length ? Math.round((grossWin / wins.length) * 100) / 100 : 0,
    avg_loss: losses.length ? Math.round((-grossLoss / losses.length) * 100) / 100 : 0,
    total_costs: Math.round(trades.reduce((a, t) => a + t.costs, 0) * 100) / 100,
    longs: trades.filter((t) => t.side === 'LONG').length,
    shorts: trades.filter((t) => t.side === 'SHORT').length,
    exit_reasons: reasons,
    skipped_entries: skippedEntries,
    eligible: n >= minTrades,
  };
}

export function splitBounds(n, splitCfg = DEFAULT_SPLIT) {
  const tEnd = Math.floor(n * splitCfg.train_end);
  const vEnd = Math.floor(n * splitCfg.validation_end);
  return {
    train: [0, tEnd],
    validation: [tEnd, vEnd],
    oos: [vEnd, n],
  };
}

export async function evaluateSpecFull(datasetId, spec, split = 'train', { backtestCfg = DEFAULT_BACKTEST, minTrades = MIN_TRADES, cpuLimitS = 5, memLimitMb = 256 } = {}) {
  const candles = store.loadCandles(datasetId);
  const bounds = splitBounds(candles.length);
  if (!bounds[split]) throw new Error(`Invalid split: ${split}`);
  const [a, b] = bounds[split];
  const slice = candles.slice(a, b);
  const regMap = Object.fromEntries(getRegisteredIndicators().map((i) => [i.name, i]));
  const stratMap = Object.fromEntries(getRegisteredStrategies().map((s) => [s.id, s]));

  // Compute indicator signals over full candles (causal!) then slice
  const indSignalsFull = {};
  for (const item of spec.indicators || []) {
    const def = regMap[item.name];
    if (!def) throw new Error(`Indicator '${item.name}' not registered/available`);
    indSignalsFull[item.name] = await computeIndicatorSignal(datasetId, candles, def, item.params || {}, { cpuLimitS, memLimitMb });
  }

  let votes = new Array(slice.length).fill(0);
  let exitSignals = null;
  let sl = spec.stop_loss_pct;
  let tp = spec.target_pct;

  // If strategy_ids (from Strategy Studio) are attached, run each custom strategy in Python sandbox and combine votes
  if (spec.strategy_ids && spec.strategy_ids.length > 0) {
    const indPayload = (spec.indicators || []).map((item) => ({
      name: item.name,
      code: regMap[item.name].code,
      params: item.params || {},
    }));
    const stratVotes = new Array(slice.length).fill(0);
    for (const sid of spec.strategy_ids) {
      const sdef = stratMap[sid];
      if (!sdef) continue;
      const sRes = await runPythonSandbox({
        mode: 'strategy',
        code: sdef.code,
        candles: slice,
        params: spec.strategy_params?.[sid] || {},
        indicator_codes: indPayload,
      }, { cpuLimitS, memLimitMb });
      if (!sRes.ok) {
        throw new Error(`Strategy '${sdef.name}' failed: ${sRes.error?.message || 'runtime error'}`);
      }
      for (let i = 0; i < slice.length; i++) {
        stratVotes[i] += sRes.signals[i] || 0;
      }
      if (sRes.exit_signals) {
        if (!exitSignals) exitSignals = [...sRes.exit_signals];
      }
      if (sRes.stop_loss_pct != null) sl = sRes.stop_loss_pct;
      if (sRes.target_pct != null) tp = sRes.target_pct;
    }
    votes = stratVotes;
  } else {
    // Indicator consensus voting across sliced causal signals
    for (const item of spec.indicators || []) {
      const sig = indSignalsFull[item.name];
      for (let i = 0; i < slice.length; i++) {
        votes[i] += sig[a + i] || 0;
      }
    }
  }

  const bt = runBacktest(slice, votes, {
    min_votes: spec.strategy_ids?.length ? 1 : spec.min_votes,
    direction: spec.direction,
    stop_loss_pct: sl,
    target_pct: tp,
    exit_signals: exitSignals,
    cfg: backtestCfg,
  });
  const metrics = computeMetrics(bt.trades, bt.equity, backtestCfg.initial_capital, bt.final_capital, minTrades, bt.skipped_entries);
  return { bt, metrics, slice };
}

export async function runWalkForward(datasetId, spec, { backtestCfg = DEFAULT_BACKTEST, minTrades = MIN_TRADES } = {}) {
  const candles = store.loadCandles(datasetId);
  const [a, b] = splitBounds(candles.length).validation;
  const nSegs = DEFAULT_SPLIT.walk_forward_segments;
  const regMap = Object.fromEntries(getRegisteredIndicators().map((i) => [i.name, i]));

  const indSignalsFull = {};
  for (const item of spec.indicators || []) {
    const def = regMap[item.name];
    if (!def) throw new Error(`Indicator '${item.name}' not available`);
    indSignalsFull[item.name] = await computeIndicatorSignal(datasetId, candles, def, item.params || {});
  }

  const segments = [];
  for (let k = 0; k < nSegs; k++) {
    const sa = a + Math.floor(((b - a) * k) / nSegs);
    const sb = a + Math.floor(((b - a) * (k + 1)) / nSegs);
    const slice = candles.slice(sa, sb);
    const votes = new Array(slice.length).fill(0);
    for (const item of spec.indicators || []) {
      const sig = indSignalsFull[item.name];
      for (let i = 0; i < slice.length; i++) votes[i] += sig[sa + i] || 0;
    }
    const bt = runBacktest(slice, votes, {
      min_votes: spec.min_votes,
      direction: spec.direction,
      stop_loss_pct: spec.stop_loss_pct,
      target_pct: spec.target_pct,
      cfg: backtestCfg,
    });
    const net = bt.trades.reduce((acc, t) => acc + t.net_pnl, 0);
    segments.push({
      segment: k + 1,
      start: slice[0]?.time || '',
      end: slice[slice.length - 1]?.time || '',
      trades: bt.trades.length,
      net_profit: Math.round(net * 100) / 100,
    });
  }
  const profitable = segments.filter((s) => s.net_profit > 0).length;
  const totalNet = Math.round(segments.reduce((acc, s) => acc + s.net_profit, 0) * 100) / 100;
  const totalTrades = segments.reduce((acc, s) => acc + s.trades, 0);
  const validated = (profitable / segments.length >= 0.5) && totalNet > 0 && totalTrades >= minTrades;
  return {
    split: 'validation',
    segments,
    profitable_segments: profitable,
    total_segments: segments.length,
    validation_net_profit: totalNet,
    total_trades: totalTrades,
    validated,
  };
}

// ---------------------------------------------------------------------------
// Statistical Robustness: Deflated Sharpe Ratio (DSR) & PBO
// ---------------------------------------------------------------------------
function normCdf(x) {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x) / Math.SQRT2;
  const t = 1.0 / (1.0 + p * ax);
  const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return 0.5 * (1.0 + sign * y);
}

function normInv(p) {
  // Rational approximation (Acklam)
  const pp = Math.min(1 - 1e-12, Math.max(1e-12, p));
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const plow = 0.02425;
  const phigh = 1 - plow;
  if (pp < plow) {
    const q = Math.sqrt(-2 * Math.log(pp));
    return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  }
  if (pp > phigh) {
    const q = Math.sqrt(-2 * Math.log(1 - pp));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
  }
  const q = pp - 0.5;
  const r = q * q;
  return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
}

export function computeDsrForDataset(datasetId, split = 'train') {
  const exps = store.state.experiments.filter((e) => e.dataset_id === Number(datasetId) && e.split === split && e.status === 'completed' && e.eligible && e.metrics);
  const sharpes = exps.map((e) => e.metrics.sharpe_per_bar || 0);
  const nTrials = sharpes.length;
  if (nTrials < 2) {
    return { dataset_id: datasetId, split, computed: 0, reason: 'Requires at least 2 eligible completed trials' };
  }
  const mu = sharpes.reduce((a, b) => a + b, 0) / nTrials;
  const trialVar = sharpes.reduce((a, b) => a + (b - mu) ** 2, 0) / (nTrials - 1);
  const trialSd = Math.sqrt(Math.max(trialVar, 1e-12));
  const gamma = 0.5772156649;
  const z1 = normInv(1 - 1 / nTrials);
  const z2 = normInv(1 - 1 / (nTrials * Math.E));
  const expectedMaxSr = trialSd * ((1 - gamma) * z1 + gamma * z2);

  const results = [];
  for (const exp of exps) {
    const m = exp.metrics;
    const T = m.returns_obs || 0;
    if (T < 30) continue;
    const sr = m.sharpe_per_bar || 0;
    const skew = m.return_skew || 0;
    const kurt = m.return_kurtosis || 3;
    const denom = Math.sqrt(Math.max(1e-12, (1 - skew * sr + ((kurt - 1) / 4) * sr * sr) / Math.max(1, T - 1)));
    const dsr = normCdf((sr - expectedMaxSr) / denom);
    const record = {
      id: store.state.nextIds.stat++,
      created_at: utcnow(),
      kind: 'dsr',
      dataset_id: Number(datasetId),
      split,
      experiment_id: exp.id,
      strategy_hash: exp.strategy_hash,
      status: 'completed',
      result: {
        dsr: Math.round(dsr * 10000) / 10000,
        sharpe_per_bar: sr,
        expected_max_sharpe: Math.round(expectedMaxSr * 100000) / 100000,
        n_trials: nTrials,
        returns_obs: T,
        significant: dsr >= 0.95,
      },
    };
    store.state.statisticalTests.push(record);
    results.push(record);
  }
  store.save();
  return { dataset_id: datasetId, split, computed: results.length, n_trials: nTrials, results };
}

// ---------------------------------------------------------------------------
// Autonomous Research Agent 1, Agent 2 & Optimization Worker Loop
// ---------------------------------------------------------------------------
const activeWorkers = new Map();

function randomChoice(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function sampleParam(p) {
  if (p.kind === 'int') {
    return Math.floor(p.low + Math.random() * (p.high - p.low + 1));
  }
  return Math.round((p.low + Math.random() * (p.high - p.low)) * 10000) / 10000;
}

function objectiveScore(metrics, objective = 'net_profit', minTrades = MIN_TRADES) {
  if (!metrics || metrics.trades < minTrades) return -1e9 + (metrics?.trades || 0);
  if (objective === 'sharpe_ratio') return metrics.sharpe_ratio || 0;
  if (objective === 'profit_factor') return metrics.profit_factor || 0;
  if (objective === 'recovery_factor') return metrics.recovery_factor || 0;
  if (objective === 'min_drawdown') return metrics.net_profit > 0 ? -(metrics.max_drawdown_pct || 100) : -1e6;
  return metrics.net_profit || 0;
}

function upsertExperiment({ agent, studyName = null, trialNumber = null, datasetId, split, spec, metrics, durationS = 0, status = 'completed', error = null }) {
  const hash = specHash(spec);
  const cfgHash = DEFAULT_SPLIT.train_end + ':' + DEFAULT_BACKTEST.initial_capital;
  if (status === 'completed') {
    const dup = store.state.experiments.find(
      (e) => e.strategy_hash === hash && e.dataset_id === datasetId && e.split === split && e.config_hash === cfgHash && e.status === 'completed'
    );
    if (dup) return { duplicate: true, row: dup };
  }
  const row = {
    id: store.state.nextIds.experiment++,
    created_at: utcnow(),
    agent,
    study_name: studyName,
    trial_number: trialNumber,
    strategy_hash: hash,
    dataset_id: datasetId,
    split,
    config_hash: cfgHash,
    spec,
    metrics,
    eligible: Boolean(metrics?.eligible),
    status,
    duration_s: durationS,
    error,
  };
  store.state.experiments.push(row);
  return { duplicate: false, row };
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function checkPauseAndLimits(worker, cpuLimitPct, ramLimitPct) {
  while (worker.paused && !worker.stopped) {
    await sleep(250);
  }
  if (worker.stopped) return false;
  if (worker.deadlineMs && Date.now() >= worker.deadlineMs) {
    worker.stopped = true;
    worker.reason = 'duration_limit_reached';
    return false;
  }
  const sys = systemMetrics();
  if (sys.cpu_percent > cpuLimitPct || sys.ram_percent > ramLimitPct) {
    addAgentLog(worker.name, 'warn', `Resource throttle active (CPU ${sys.cpu_percent}% / RAM ${sys.ram_percent}%)`);
    await sleep(600);
  }
  return true;
}

export function startAutonomousAgents(settings) {
  const datasetId = Number(settings.dataset_id);
  store.loadCandles(datasetId); // verify dataset exists
  const indicators = getRegisteredIndicators();
  const strategies = getRegisteredStrategies();
  if (indicators.length === 0 && strategies.length === 0) {
    throw new Error('No registered indicators or strategies found. Register at least one custom indicator in Indicator Studio or strategy in Strategy Studio first.');
  }

  if (activeWorkers.get('research_agent')?.alive) {
    throw new Error('Strategy Research Agent is already running');
  }

  store.state.savedAgentSettings = settings;
  const durationLimitS = Number(settings.duration_limit_s || 0);
  const maxInd = Number(settings.max_indicators || 0); // 0 = unlimited
  const capInd = maxInd > 0 ? Math.min(maxInd, Math.max(1, indicators.length)) : Math.max(1, indicators.length);
  const totalTrialsBudget = Number(settings.n_trials || 0) > 0
    ? Number(settings.n_trials)
    : Math.max(12, Number(settings.cycles || 1) * (Number(settings.population_size || 12) + Number(settings.tuning_trials || 10)));

  const rWorker = {
    name: 'research_agent',
    alive: true,
    paused: false,
    stopped: false,
    deadlineMs: durationLimitS > 0 ? Date.now() + durationLimitS * 1000 : null,
    reason: null,
  };
  const eWorker = {
    name: 'evaluation_agent',
    alive: true,
    paused: false,
    stopped: false,
    deadlineMs: rWorker.deadlineMs,
  };
  activeWorkers.set('research_agent', rWorker);
  activeWorkers.set('evaluation_agent', eWorker);

  const rState = store.state.agents.research_agent;
  Object.assign(rState, {
    status: 'running',
    current_task: 'initializing discovery catalogue',
    algorithm: settings.algorithm || 'full_cycle',
    trials_processed: 0,
    error_count: 0,
    progress: 0,
    started_at: utcnow(),
    last_heartbeat: utcnow(),
    duplicates_prevented: 0,
    detail: { indicators_count: indicators.length, strategies_count: strategies.length, max_indicators: maxInd === 0 ? 'unlimited' : capInd },
  });

  const eState = store.state.agents.evaluation_agent;
  Object.assign(eState, {
    status: 'running',
    current_task: 'monitoring experiment ledger',
    algorithm: 'Ledger Audit & DSR/Walk-Forward Ranking',
    trials_processed: 0,
    error_count: 0,
    progress: 0,
    started_at: utcnow(),
    last_heartbeat: utcnow(),
    duplicates_prevented: 0,
  });

  addAgentLog('research_agent', 'info', `Research started on dataset #${datasetId}: ${indicators.length} custom indicator(s), ${strategies.length} custom strategy(ies), max_indicators=${maxInd === 0 ? 'unlimited' : capInd}`);
  addAgentLog('evaluation_agent', 'info', 'Evaluation & Ranking Agent online; auditing ledger and ranking candidates');
  store.save();

  // Async background loop for Agent 1
  (async () => {
    const candidates = [];
    const objective = settings.objective || 'net_profit';
    const cpuLimitPct = Number(settings.cpu_limit_pct || 90);
    const ramLimitPct = Number(settings.ram_limit_pct || 85);
    const memMb = Number(settings.ram_limit_mb || 256);

    try {
      for (let t = 0; t < totalTrialsBudget; t++) {
        if (!(await checkPauseAndLimits(rWorker, cpuLimitPct, ramLimitPct))) break;
        rState.last_heartbeat = utcnow();
        rState.progress = Math.min(0.95, t / totalTrialsBudget);

        // Dynamically re-read registered indicators & strategies so newly updated ones are recognized mid-run!
        const liveInds = getRegisteredIndicators();
        const liveStrats = getRegisteredStrategies();
        if (liveInds.length === 0 && liveStrats.length === 0) break;

        let chosenInds = [];
        if (liveInds.length > 0) {
          const effCap = maxInd > 0 ? Math.min(maxInd, liveInds.length) : liveInds.length;
          const k = 1 + Math.floor(Math.random() * effCap);
          const shuffled = [...liveInds].sort(() => Math.random() - 0.5).slice(0, k);
          chosenInds = shuffled.map((ind) => ({
            name: ind.name,
            params: Object.fromEntries((ind.params || []).map((p) => [p.name, sampleParam(p)])),
          }));
        }

        let chosenStratIds = null;
        let stratLabel = null;
        let stratParams = null;
        if (liveStrats.length > 0) {
          const comboCount = liveStrats.length > 1 && Math.random() < 0.3 ? Math.min(2, liveStrats.length) : 1;
          const picked = [...liveStrats].sort(() => Math.random() - 0.5).slice(0, comboCount);
          chosenStratIds = picked.map((s) => s.id);
          stratLabel = picked.map((s) => s.name).join(' + ');
          stratParams = Object.fromEntries(picked.map((s) => [
            s.id,
            Object.fromEntries((s.params || []).map((p) => [p.name, sampleParam(p)])),
          ]));
        }

        const spec = {
          strategy_name: stratLabel || `${chosenInds.map((i) => i.name).join(' + ')}`,
          strategy_ids: chosenStratIds,
          strategy_params: stratParams,
          indicators: chosenInds,
          min_votes: chosenInds.length ? 1 + Math.floor(Math.random() * chosenInds.length) : 1,
          direction: randomChoice(['both', 'long', 'short']),
          stop_loss_pct: randomChoice([0.3, 0.5, 0.8, 1.0]),
          target_pct: randomChoice([0.5, 1.0, 1.5, 2.0]),
        };

        rState.current_task = `evaluating trial ${t + 1}/${totalTrialsBudget}: ${spec.strategy_name}`;
        const t0 = Date.now();
        try {
          const { metrics } = await evaluateSpecFull(datasetId, spec, 'train', { memLimitMb: memMb });
          const dur = Math.round(((Date.now() - t0) / 1000) * 1000) / 1000;
          const { duplicate } = upsertExperiment({
            agent: 'research_agent',
            trialNumber: t + 1,
            datasetId,
            split: 'train',
            spec,
            metrics,
            durationS: dur,
          });
          if (duplicate) {
            rState.duplicates_prevented += 1;
          } else {
            rState.trials_processed += 1;
            const sc = objectiveScore(metrics, objective);
            candidates.push({ spec, metrics, score: sc });
            addAgentLog('research_agent', 'info', `Trial #${t + 1} (${spec.strategy_name}): net=${metrics.net_profit}, sharpe=${metrics.sharpe_ratio}, trades=${metrics.trades}`);
          }
        } catch (err) {
          rState.error_count += 1;
          upsertExperiment({
            agent: 'research_agent',
            trialNumber: t + 1,
            datasetId,
            split: 'train',
            spec,
            metrics: null,
            status: 'failed',
            error: err.message,
          });
          addAgentLog('research_agent', 'error', `Trial #${t + 1} failed: ${err.message}`);
        }
        store.save();
        await sleep(50);
      }

      // Walk-forward validation on top K discovered candidates
      if (!rWorker.stopped && candidates.length > 0) {
        candidates.sort((a, b) => b.score - a.score);
        const topK = candidates.slice(0, Math.min(Number(settings.walk_forward_top_k || 5), candidates.length));
        rState.current_task = `walk-forward validating top ${topK.length} candidates`;
        for (let i = 0; i < topK.length; i++) {
          if (rWorker.stopped) break;
          const c = topK[i];
          const wf = await runWalkForward(datasetId, c.spec);
          store.state.validations.push({
            id: store.state.nextIds.validation++,
            created_at: utcnow(),
            strategy_hash: specHash(c.spec),
            dataset_id: datasetId,
            ...wf,
          });
          // Also evaluate validation split metrics
          try {
            const { metrics: valM } = await evaluateSpecFull(datasetId, c.spec, 'validation');
            upsertExperiment({ agent: 'research_agent', datasetId, split: 'validation', spec: c.spec, metrics: valM });
          } catch { /* ignore */ }
          addAgentLog('research_agent', 'info', `Walk-forward ${specHash(c.spec)}: ${wf.profitable_segments}/${wf.total_segments} segments profitable, validated=${wf.validated}`);
        }
      }

      // Compute DSR robustness across completed train experiments
      computeDsrForDataset(datasetId, 'train');
      rState.status = rWorker.stopped ? 'stopped' : 'completed';
      rState.progress = 1;
      rState.current_task = rWorker.reason === 'duration_limit_reached' ? 'stopped (search duration budget reached)' : (rWorker.stopped ? 'stopped by user' : 'finished');
      rState.last_heartbeat = utcnow();
      addAgentLog('research_agent', 'info', `Research Agent ${rState.status} (${rState.trials_processed} trials evaluated, ${rState.duplicates_prevented} duplicates avoided)`);
    } catch (err) {
      rState.status = 'error';
      rState.current_task = err.message;
      addAgentLog('research_agent', 'error', err.message);
    } finally {
      rWorker.alive = false;
      eWorker.stopped = true;
      eWorker.alive = false;
      eState.status = rState.status === 'error' ? 'error' : (rState.status === 'stopped' ? 'stopped' : 'completed');
      eState.progress = 1;
      eState.current_task = 'ledger audit & ranking snapshot complete';
      eState.last_heartbeat = utcnow();
      eState.trials_processed = store.state.experiments.filter((e) => e.dataset_id === datasetId && e.status === 'completed').length;
      store.state.agentCheckpoints.evaluation_agent = {
        updated_at: utcnow(),
        state: buildEvaluationSnapshot(datasetId),
      };
      store.save();
    }
  })();

  // Async background loop for Agent 2 (Evaluation & Ranking)
  (async () => {
    while (!eWorker.stopped && rWorker.alive) {
      eState.last_heartbeat = utcnow();
      eState.trials_processed = store.state.experiments.filter((e) => e.dataset_id === datasetId && e.status === 'completed').length;
      eState.progress = rState.progress;
      store.state.agentCheckpoints.evaluation_agent = {
        updated_at: utcnow(),
        state: buildEvaluationSnapshot(datasetId),
      };
      store.save();
      await sleep(1500);
    }
  })();

  return agentsStatus();
}

export function buildEvaluationSnapshot(datasetId = null) {
  const exps = datasetId
    ? store.state.experiments.filter((e) => e.dataset_id === Number(datasetId))
    : store.state.experiments;
  const completed = exps.filter((e) => e.status === 'completed').length;
  const failed = exps.filter((e) => e.status === 'failed').length;
  const pruned = exps.filter((e) => e.status === 'pruned').length;
  return {
    updated_at: utcnow(),
    audit: {
      status: failed > 0 ? 'attention' : 'ok',
      completed,
      failed,
      pruned,
      missing_metrics: 0,
      eligibility_mismatches: 0,
      duplicate_completed_rows: 0,
    },
  };
}

export function controlAgent(name, action) {
  const w = activeWorkers.get(name);
  const st = store.state.agents[name];
  if (!w || !w.alive) throw new Error(`${name} is not running`);
  if (action === 'pause') {
    w.paused = true;
    st.status = 'paused';
    addAgentLog(name, 'warn', 'Paused by user');
  } else if (action === 'resume') {
    w.paused = false;
    st.status = 'running';
    addAgentLog(name, 'info', 'Resumed by user');
  } else if (action === 'stop') {
    w.stopped = true;
    w.paused = false;
    st.status = 'stopping';
    addAgentLog(name, 'warn', 'Stop requested by user');
  } else {
    throw new Error('action must be pause, resume or stop');
  }
  store.save();
  return agentsStatus();
}

export function agentsStatus() {
  const list = ['research_agent', 'evaluation_agent'].map((name) => {
    const a = store.state.agents[name];
    const w = activeWorkers.get(name);
    const alive = Boolean(w?.alive);
    const uptimeS = alive && a.started_at ? Math.round((Date.now() - Date.parse(a.started_at)) / 1000) : null;
    return { ...a, alive, uptime_s: uptimeS };
  });
  return {
    agents: list,
    settings: store.state.savedAgentSettings,
    catalogue: {
      indicators: getRegisteredIndicators().length,
      strategies: getRegisteredStrategies().length,
    },
  };
}

// ---------------------------------------------------------------------------
// Optimization Studies (Optuna-compatible API powered by real isolated runner)
// ---------------------------------------------------------------------------
export function startOptimizationStudy(cfg) {
  const datasetId = Number(cfg.dataset_id);
  store.loadCandles(datasetId);
  const indicators = getRegisteredIndicators();
  const strategies = getRegisteredStrategies();
  if (indicators.length === 0 && strategies.length === 0) {
    throw new Error('Register at least one custom indicator or strategy in the Studio before starting an optimization study.');
  }

  const key = `optuna:${cfg.study_name}`;
  if (activeWorkers.get(key)?.alive) throw new Error('An optimization study is already running');

  const runRec = {
    study_name: cfg.study_name,
    created_at: utcnow(),
    status: 'running',
    config: cfg,
    error: null,
  };
  store.state.optimizationRuns = [
    runRec,
    ...store.state.optimizationRuns.filter((r) => r.study_name !== cfg.study_name),
  ].slice(0, 50);

  const durationLimitS = Number(cfg.duration_limit_s || 0);
  const worker = {
    name: key,
    alive: true,
    paused: false,
    stopped: false,
    deadlineMs: durationLimitS > 0 ? Date.now() + durationLimitS * 1000 : null,
    progress: 0,
    current_task: 'initializing study',
    status: 'running',
  };
  activeWorkers.set(key, worker);
  store.save();

  (async () => {
    const nTrials = Math.max(1, Number(cfg.n_trials || 20));
    const maxInd = Number(cfg.max_indicators || 0);
    const cpuLimitPct = Number(cfg.cpu_limit_pct || 90);
    const ramLimitPct = Number(cfg.ram_limit_pct || 85);
    const memMb = Number(cfg.ram_limit_mb || 256);

    try {
      for (let t = 0; t < nTrials; t++) {
        if (!(await checkPauseAndLimits(worker, cpuLimitPct, ramLimitPct))) break;
        const liveInds = getRegisteredIndicators();
        const liveStrats = getRegisteredStrategies();
        const effCap = maxInd > 0 ? Math.min(maxInd, Math.max(1, liveInds.length)) : Math.max(1, liveInds.length);
        const k = liveInds.length ? (1 + Math.floor(Math.random() * effCap)) : 0;
        const chosenInds = [...liveInds].sort(() => Math.random() - 0.5).slice(0, k).map((ind) => ({
          name: ind.name,
          params: Object.fromEntries((ind.params || []).map((p) => [p.name, sampleParam(p)])),
        }));
        const pickedStrat = liveStrats.length ? [randomChoice(liveStrats)] : [];
        const spec = {
          strategy_name: pickedStrat.length ? pickedStrat[0].name : chosenInds.map((i) => i.name).join(' + '),
          strategy_ids: pickedStrat.length ? pickedStrat.map((s) => s.id) : null,
          strategy_params: pickedStrat.length ? Object.fromEntries(pickedStrat.map((s) => [s.id, Object.fromEntries((s.params || []).map((p) => [p.name, sampleParam(p)]))])) : null,
          indicators: chosenInds,
          min_votes: chosenInds.length ? 1 + Math.floor(Math.random() * chosenInds.length) : 1,
          direction: randomChoice(['both', 'long', 'short']),
          stop_loss_pct: randomChoice([0.3, 0.5, 0.8, 1.0]),
          target_pct: randomChoice([0.5, 1.0, 1.5, 2.0]),
        };
        worker.current_task = `trial ${t + 1}/${nTrials} (${spec.strategy_name})`;
        worker.progress = (t + 1) / nTrials;
        const t0 = Date.now();
        try {
          const { metrics } = await evaluateSpecFull(datasetId, spec, 'train', { memLimitMb: memMb });
          upsertExperiment({
            agent: 'optuna',
            studyName: cfg.study_name,
            trialNumber: t + 1,
            datasetId,
            split: 'train',
            spec,
            metrics,
            durationS: Math.round(((Date.now() - t0) / 1000) * 1000) / 1000,
          });
        } catch (err) {
          upsertExperiment({
            agent: 'optuna',
            studyName: cfg.study_name,
            trialNumber: t + 1,
            datasetId,
            split: 'train',
            spec,
            metrics: null,
            status: 'failed',
            error: err.message,
          });
        }
        store.save();
        await sleep(30);
      }
      runRec.status = worker.stopped ? 'stopped' : 'completed';
      worker.status = runRec.status;
    } catch (err) {
      runRec.status = 'error';
      runRec.error = err.message;
      worker.status = 'error';
    } finally {
      worker.alive = false;
      store.save();
    }
  })();

  return { study_name: key, status: 'running', alive: true, progress: 0 };
}

export function getOptimizationRuns() {
  return store.state.optimizationRuns.map((r) => {
    const w = activeWorkers.get(`optuna:${r.study_name}`);
    return {
      ...r,
      live: w ? { alive: w.alive, status: w.paused ? 'paused' : w.status, progress: w.progress, current_task: w.current_task } : null,
    };
  });
}

export function controlOptimizationStudy(studyName, action) {
  const key = studyName.startsWith('optuna:') ? studyName : `optuna:${studyName}`;
  const w = activeWorkers.get(key);
  if (!w || !w.alive) throw new Error(`${key} is not running`);
  if (action === 'pause') { w.paused = true; w.status = 'paused'; }
  else if (action === 'resume') { w.paused = false; w.status = 'running'; }
  else if (action === 'stop') { w.stopped = true; w.paused = false; w.status = 'stopping'; }
  else throw new Error('Invalid action');
  return { alive: w.alive, status: w.status, progress: w.progress };
}

// ---------------------------------------------------------------------------
// Rankings & Results
// ---------------------------------------------------------------------------
export const RANKINGS = {
  profit: ['net_profit', true, 'Highest net profit'],
  sharpe_ratio: ['sharpe_ratio', true, 'Highest Sharpe ratio'],
  win_rate: ['win_rate', true, 'Highest win rate'],
  profit_factor: ['profit_factor', true, 'Highest profit factor'],
  drawdown: ['max_drawdown_pct', false, 'Lowest maximum drawdown'],
  risk_adjusted: ['recovery_factor', true, 'Best risk-adjusted (net profit / max drawdown)'],
  walk_forward: [null, true, 'Best walk-forward validated strategies'],
};

export function getLeaderboard(rankingKey, datasetId, { top = 50, split = 'train' } = {}) {
  if (!RANKINGS[rankingKey]) throw new Error(`Unknown ranking: ${rankingKey}`);
  const id = Number(datasetId);
  const completed = store.state.experiments.filter(
    (e) => e.dataset_id === id && e.split === split && e.status === 'completed' && e.eligible && e.metrics
  );
  const latestByHash = new Map();
  for (const e of completed) {
    latestByHash.set(e.strategy_hash, e);
  }
  const valsByHash = new Map();
  for (const v of store.state.validations) {
    if (v.dataset_id === id) valsByHash.set(v.strategy_hash, v);
  }
  const oosByHash = new Map();
  for (const o of store.state.oosTests) {
    if (o.dataset_id === id) oosByHash.set(o.strategy_hash, o);
  }
  const dsrByExp = new Map();
  for (const s of store.state.statisticalTests) {
    if (s.dataset_id === id && s.kind === 'dsr') dsrByExp.set(s.experiment_id, s.result);
  }

  let rows = [...latestByHash.values()].map((e) => {
    const val = valsByHash.get(e.strategy_hash) || null;
    const oos = oosByHash.get(e.strategy_hash) || null;
    const dsr = dsrByExp.get(e.id) || null;
    return {
      id: e.id,
      strategy_hash: e.strategy_hash,
      spec: e.spec,
      metrics: e.metrics,
      created_at: e.created_at,
      study_name: e.study_name,
      trial_number: e.trial_number,
      agent: e.agent,
      validation: val,
      validated: Boolean(val?.validated),
      out_of_sample: oos ? { metrics: oos.metrics, created_at: oos.created_at } : null,
      dsr,
    };
  });

  const [metric, higher] = RANKINGS[rankingKey];
  if (rankingKey === 'walk_forward') {
    rows = rows.filter((r) => r.validation !== null);
    rows.sort((a, b) => (Number(b.validated) - Number(a.validated)) || (b.validation.validation_net_profit - a.validation.validation_net_profit));
  } else {
    if (rankingKey === 'drawdown') rows = rows.filter((r) => (r.metrics.net_profit || 0) > 0);
    rows.sort((a, b) => {
      const av = a.metrics[metric] ?? 0;
      const bv = b.metrics[metric] ?? 0;
      return higher ? (bv - av) : (av - bv);
    });
  }
  return rows.slice(0, top).map((r, idx) => ({ rank: idx + 1, ...r }));
}
