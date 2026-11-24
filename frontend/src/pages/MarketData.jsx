import { useEffect, useMemo, useState } from 'react';
import { useApp } from '../context.js';
import { get } from '../api.js';
import Plot from '../components/Plot.jsx';
import { ErrorBox } from '../components/ui.jsx';

// Timeframes shown in the toolbar. ``api`` is the value the backend resample accepts.
const TIMEFRAMES = [
  { label: '1m', api: '1min', minutes: 1 },
  { label: '5m', api: '5min', minutes: 5 },
  { label: '10m', api: '10min', minutes: 10 },
  { label: '1H', api: '1h', minutes: 60 },
  { label: '4H', api: '4h', minutes: 240 },
];
const PRESETS = [
  { label: 'Full History', years: null },
  { label: 'Last 1Y', years: 1 },
  { label: 'Last 3Y', years: 3 },
  { label: 'Last 5Y', years: 5 },
];
const MAX_BARS = 5000;          // must match MAX_CHART_BARS in backend/indicatorlab/api.py
const RESET_BARS = 200;         // Reset View shows about this many bars of the chosen timeframe
const TWO_DAYS_MS = 2 * 86400000;
const DAY_MS = 86400000;
const MIN_SPAN_MIN = 1;

// Dataset timestamps are naive exchange-local times. They are parsed as UTC so arithmetic is exact.
const toMs = (s) => Date.parse(String(s).replace(' ', 'T').slice(0, 19) + 'Z');
const toIso = (ms) => new Date(ms).toISOString().slice(0, 19);
const fmt = (n, d = 2) => (n === null || n === undefined ? '—' : Number(n).toFixed(d));
const fmtInt = (n) => (n === null || n === undefined ? '—' : Number(n).toLocaleString('en-IN'));

export default function MarketData() {
  const { datasets, datasetId } = useApp();
  const ds = datasets.find((d) => d.id === datasetId) || null;
  const [tf, setTf] = useState(TIMEFRAMES[0]);
  const [win, setWin] = useState(null);             // {start, end} as epoch ms
  const [preset, setPreset] = useState('Full History');
  const [chart, setChart] = useState(null);         // backend payload
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [customStart, setCustomStart] = useState('');
  const [customEnd, setCustomEnd] = useState('');

  const bounds = useMemo(() => (ds ? { lo: toMs(ds.start_ts), hi: toMs(ds.end_ts) } : null), [ds]);

  // Reset to the latest bars whenever the dataset or timeframe changes.
  useEffect(() => {
    if (!bounds) { setWin(null); return; }
    setWin(resetWindow(bounds, tf));
    setPreset('Reset');
  }, [bounds, tf]); // eslint-disable-line react-hooks/exhaustive-deps

  // Load candles for the selected window. Stale responses are ignored.
  useEffect(() => {
    if (!datasetId || !win) return undefined;
    let cancelled = false;
    setBusy(true);
    setError(null);
    const q = new URLSearchParams({ timeframe: tf.api, start: toIso(win.start), end: toIso(win.end), limit: String(MAX_BARS) });
    get(`/api/datasets/${datasetId}/candles?${q}`)
      .then((r) => { if (!cancelled) setChart(r); })
      .catch((e) => { if (!cancelled) { setChart(null); setError(e.message); } })
      .finally(() => { if (!cancelled) setBusy(false); });
    return () => { cancelled = true; };
  }, [datasetId, tf, win?.start, win?.end]); // eslint-disable-line react-hooks/exhaustive-deps

  const apply = (start, end, label = 'Custom') => {
    if (!bounds) return;
    let s = Math.max(bounds.lo, Math.min(start, bounds.hi));
    let e = Math.max(bounds.lo, Math.min(end, bounds.hi));
    if (e - s < tf.minutes * 60000) e = Math.min(bounds.hi, s + tf.minutes * 60000);
    setWin({ start: s, end: e });
    setPreset(label);
  };

  const span = win ? win.end - win.start : 0;
  const shift = (direction) => { if (win) apply(win.start + direction * span / 2, win.end + direction * span / 2); };
  const latest = () => { if (bounds) apply(bounds.hi - span, bounds.hi, 'Latest'); };
  const zoom = (factor) => {
    if (!win || !bounds) return;
    const mid = (win.start + win.end) / 2;
    const next = Math.max(tf.minutes * 60000 * MIN_SPAN_MIN, Math.min(bounds.hi - bounds.lo, span * factor));
    apply(mid - next / 2, mid + next / 2, preset);
  };
  const resetView = () => { if (bounds) setWin(resetWindow(bounds, tf)); setPreset('Reset'); };
  const pickPreset = (p) => {
    if (!bounds) return;
    if (p.years === null) return apply(bounds.lo, bounds.hi, p.label);
    const start = Math.max(bounds.lo, bounds.hi - p.years * 365 * DAY_MS);
    apply(start, bounds.hi, p.label);
  };
  const applyCustom = () => {
    if (!customStart || !customEnd) return;
    apply(Date.parse(`${customStart}T00:00:00Z`), Date.parse(`${customEnd}T23:59:59Z`), 'Custom');
  };

  // Zoom and pan with the mouse: Plotly reports the new x range, which is fetched from the backend.
  const onRelayout = (ev) => {
    if (!bounds || !ev) return;
    if (ev['xaxis.autorange']) return resetView();
    const a = ev['xaxis.range[0]'];
    const b = ev['xaxis.range[1]'];
    if (a && b) apply(toMs(a), toMs(b), 'Custom');
  };

  const bars = chart?.bars || [];
  const last = bars[bars.length - 1];
  const prev = bars[bars.length - 2];
  const change = last && prev ? last.close - prev.close : null;
  const changePct = change !== null && prev.close ? (change / prev.close) * 100 : null;
  const tone = change === null ? '' : change >= 0 ? 'pos' : 'neg';

  const figure = useMemo(() => {
    const x = bars.map((b) => b.time);
    return {
      data: [{
        type: 'candlestick', name: `${ds?.symbol || ''} ${tf.label}`, x,
        open: bars.map((b) => b.open), high: bars.map((b) => b.high),
        low: bars.map((b) => b.low), close: bars.map((b) => b.close),
        increasing: { line: { color: '#22c55e' }, fillcolor: '#22c55e' },
        decreasing: { line: { color: '#ef4444' }, fillcolor: '#ef4444' },
        hovertext: bars.map((b) => `O ${fmt(b.open)}  H ${fmt(b.high)}  L ${fmt(b.low)}  C ${fmt(b.close)}`),
        hoverinfo: 'x+text',
      }],
      layout: {
        dragmode: 'zoom', hovermode: 'x',
        showlegend: false,
        xaxis: {
          rangeslider: { visible: false }, showspikes: true, spikemode: 'across', spikesnap: 'cursor',
          spikecolor: '#3b82f6', spikethickness: 1, title: { text: 'Time (IST)' },
          type: 'date', tickformat: tf.minutes >= 60 ? '%d-%b %H:%M' : '%d-%b %H:%M',
          // NSE cash market: no weekend trading and no trading outside 09:15-15:30 IST. Hide those gaps.
          rangebreaks: [{ bounds: ['sat', 'mon'] }, { bounds: [15.5, 9.25], pattern: 'hour' }],
        },
        yaxis: { showspikes: true, spikemode: 'across', spikecolor: '#3b82f6', spikethickness: 1, side: 'right', fixedrange: false },
        margin: { l: 16, r: 64, t: 12, b: 44 },
      },
    };
  }, [chart, tf, ds]);

  if (!ds) {
    return (
      <div className="page market-data">
        <h1>Market Data</h1>
        <div className="empty">No dataset selected. Import or select a dataset in the top bar.</div>
      </div>
    );
  }

  return (
    <div className="page market-data">
      <div className="md-header">
        <h1>Market Data</h1>
        <div className="md-dataset">
          <span className="md-symbol">{ds.symbol}</span>
          <span className="md-name">{ds.name}</span>
        </div>
      </div>

      <div className="md-toolbar">
        <div className="md-group" role="group" aria-label="Timeframe">
          {TIMEFRAMES.map((t) => (
            <button key={t.label} className={tf.label === t.label ? 'chip active' : 'chip'} onClick={() => setTf(t)}>{t.label}</button>
          ))}
        </div>
        <div className="md-group" role="group" aria-label="Range">
          {PRESETS.map((p) => (
            <button key={p.label} className={preset === p.label ? 'chip active' : 'chip'} onClick={() => pickPreset(p)}>{p.label}</button>
          ))}
        </div>
        <div className="md-group md-dates">
          <input type="date" aria-label="Start date" value={customStart} onChange={(e) => setCustomStart(e.target.value)} />
          <span>→</span>
          <input type="date" aria-label="End date" value={customEnd} onChange={(e) => setCustomEnd(e.target.value)} />
          <button className="chip" onClick={applyCustom} disabled={!customStart || !customEnd}>Apply</button>
        </div>
      </div>

      <div className="md-toolbar">
        <div className="md-group">
          <button className="chip" onClick={() => shift(-1)}>‹ Earlier</button>
          <button className="chip" onClick={() => shift(1)}>Newer ›</button>
          <button className="chip" onClick={latest}>Latest</button>
        </div>
        <div className="md-group">
          <button className="chip" onClick={() => zoom(0.5)}>Zoom In</button>
          <button className="chip" onClick={() => zoom(2)}>Zoom Out</button>
          <button className="chip" onClick={resetView}>Reset View</button>
        </div>
      </div>

      <div className="md-chart-card">
        <div className="md-ohlc">
          <span className="md-tag">{ds.symbol} · {tf.label}</span>
          {last ? (
            <>
              <span>{last.time.replace('T', ' ')} IST</span>
              <span>O <b>{fmt(last.open)}</b></span>
              <span>H <b>{fmt(last.high)}</b></span>
              <span>L <b>{fmt(last.low)}</b></span>
              <span>C <b>{fmt(last.close)}</b></span>
              <span className={tone}>{change !== null ? `${change >= 0 ? '+' : ''}${fmt(change)} (${changePct >= 0 ? '+' : ''}${fmt(changePct)}%)` : ''}</span>
            </>
          ) : <span className="muted">{busy ? 'Loading…' : 'No bars in this range'}</span>}
        </div>

        {error && <ErrorBox error={error} />}
        <Plot data={figure.data} layout={figure.layout} height={520} onRelayout={onRelayout} />

        <div className="md-status">
          <span>Range: {chart?.start ? `${chart.start.replace('T', ' ')} → ${chart.end.replace('T', ' ')}` : '—'}</span>
          <span>Candles in range: {fmtInt(chart?.total_bars)}</span>
          <span>Candles shown: {fmtInt(bars.length)}</span>
          <span>Dataset: {ds.start_ts.replace('T', ' ')} → {ds.end_ts.replace('T', ' ')}</span>
        </div>
        {chart?.aggregated && (
          <div className="md-note">
            This range has {fmtInt(chart.total_bars)} source bars. To fit the {fmtInt(MAX_BARS)}-candle display limit,
            each candle groups {fmtInt(chart.source_bars_per_candle)} consecutive {tf.label} bars (OHLC kept exact).
            Choose a shorter range to see individual candles.
          </div>
        )}
        {busy && <div className="md-note">Loading candles…</div>}
      </div>
    </div>
  );
}

function resetWindow(bounds, tf) {
  const span = Math.min(bounds.hi - bounds.lo, Math.max(TWO_DAYS_MS, RESET_BARS * tf.minutes * 60000));
  return { start: bounds.hi - span, end: bounds.hi };
}
