import { useEffect, useState } from 'react';
import { get, post, fmtMoney, fmtPct, fmtNum, specText } from '../api.js';
import { useApp, loadStrategy, saveStrategy } from '../context.js';
import { Card, Stat, Field, ErrorBox, Table, useAsync } from '../components/ui.jsx';
import Plot from '../components/Plot.jsx';

const DEFAULT_STRATEGY = { indicators: [], min_votes: 1, direction: 'both', stop_loss_pct: 0.5, target_pct: 1.0 };

export default function Backtest() {
  const { datasetId } = useApp();
  const [strategy, setStrategy] = useState(() => loadStrategy() || DEFAULT_STRATEGY);
  const [studioStrategies, setStudioStrategies] = useState([]);
  const [regIndicators, setRegIndicators] = useState([]);
  const [split, setSplit] = useState('validation');
  const [costs, setCosts] = useState({ initial_capital: '', position_pct: '', brokerage_per_order: '', charges_pct: '', slippage_points: '' });
  const [chart, setChart] = useState('5min');
  const [result, setResult] = useState(null);
  const [oos, setOos] = useState(null);
  const [confirmOos, setConfirmOos] = useState(false);
  const { error, busy, run } = useAsync();

  useEffect(() => {
    const s = loadStrategy();
    if (s) setStrategy(s);
    get('/api/studio/strategies').then((r) => setStudioStrategies((r.items || []).filter((x) => x.registered))).catch(() => {});
    get('/api/indicators').then((r) => setRegIndicators((r.indicators || []).filter((x) => x.available))).catch(() => {});
  }, []);

  const pickStudioStrategy = (sid) => {
    if (!sid) return;
    const found = studioStrategies.find((s) => s.id === Number(sid));
    if (!found) return;
    const next = {
      strategy_name: found.name,
      strategy_ids: [found.id],
      strategy_params: { [found.id]: Object.fromEntries((found.params || []).map((p) => [p.name, p.default])) },
      indicators: (found.linked_indicators || []).map((name) => ({ name, params: {} })),
      min_votes: 1,
      direction: found.direction || 'both',
      stop_loss_pct: found.stop_loss_pct ?? 0.5,
      target_pct: found.target_pct ?? 1.0,
    };
    setStrategy(next);
    saveStrategy(next);
  };

  const addIndicatorCombo = (name) => {
    if (!name || strategy.indicators.some((i) => i.name === name)) return;
    const def = regIndicators.find((i) => i.name === name);
    const next = { ...strategy, indicators: [...strategy.indicators, { name, params: { ...(def?.defaults || {}) } }] };
    setStrategy(next);
    saveStrategy(next);
  };

  const hasRunnableTarget = Boolean(strategy.indicators?.length || strategy.strategy_ids?.length);

  const body = () => {
    const payload = { dataset_id: datasetId, strategy, split, chart_timeframe: chart };
    for (const [k, v] of Object.entries(costs)) if (v !== '') payload[k] = Number(v);
    return payload;
  };

  const runBacktest = () => run(async () => {
    setOos(null);
    setResult(await post('/api/backtest', body()));
  });

  const runOos = () => run(async () => {
    setOos(await post('/api/backtest/oos', body()));
    setConfirmOos(false);
  });

  const m = result?.metrics;
  const candles = result?.candles || [];
  const markers = result?.trades || [];
  const candleData = candles.length ? [{
    type: 'candlestick', x: candles.map((c) => c.time), open: candles.map((c) => c.open),
    high: candles.map((c) => c.high), low: candles.map((c) => c.low), close: candles.map((c) => c.close),
    name: 'Price', increasing: { line: { color: '#22c55e' } }, decreasing: { line: { color: '#ef4444' } },
  }, {
    type: 'scatter', mode: 'markers', name: 'Entry', x: markers.map((t) => t.entry_time), y: markers.map((t) => t.entry_price),
    marker: { symbol: markers.map((t) => (t.side === 'LONG' ? 'triangle-up' : 'triangle-down')), size: 10, color: '#4f8cff' },
  }, {
    type: 'scatter', mode: 'markers', name: 'Exit', x: markers.map((t) => t.exit_time), y: markers.map((t) => t.exit_price),
    marker: { symbol: 'x', size: 9, color: '#f5b942' },
  }] : null;

  const eq = result?.equity;
  const equityData = eq && eq.timestamps.length ? [
    { type: 'scatter', mode: 'lines', name: 'Equity', x: eq.timestamps, y: eq.equity, line: { color: '#4f8cff' } },
    { type: 'scatter', mode: 'lines', name: 'Drawdown', x: eq.timestamps, y: eq.drawdown.map((v) => -v), yaxis: 'y2', line: { color: '#ef4444' } },
  ] : null;

  const setCost = (k) => (e) => setCosts({ ...costs, [k]: e.target.value });

  return (
    <>
      <div className="page-head"><div><h1>Backtest</h1><div className="muted">Simulates registered custom strategies and indicator combinations bar-by-bar with explicit execution rules.</div></div></div>
      <ErrorBox error={error} />
      <Card title="Setup" actions={<button className="primary" disabled={busy || !datasetId || !hasRunnableTarget} onClick={runBacktest}>Run backtest</button>}>
        <div className="form-grid" style={{ marginBottom: 12 }}>
          <Field label="Load Registered Strategy">
            <select value={strategy.strategy_ids?.[0] || ''} onChange={(e) => pickStudioStrategy(e.target.value)}>
              <option value="">Select from Strategy Studio…</option>
              {studioStrategies.map((s) => <option key={s.id} value={s.id}>{s.name} (v{s.version})</option>)}
            </select>
          </Field>
          <Field label="Add Registered Indicator">
            <select value="" onChange={(e) => addIndicatorCombo(e.target.value)}>
              <option value="">Add indicator to combination…</option>
              {regIndicators.map((i) => <option key={i.name} value={i.name}>{i.name}</option>)}
            </select>
          </Field>
          <Field label="Split">
            <select value={split} onChange={(e) => setSplit(e.target.value)}>
              <option value="train">Train</option>
              <option value="validation">Validation</option>
            </select>
          </Field>
          <Field label="Chart timeframe">
            <select value={chart} onChange={(e) => setChart(e.target.value)}>
              {['1min', '5min', '15min', '30min', '60min'].map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </Field>
          <Field label="Initial capital"><input type="number" placeholder="default" value={costs.initial_capital} onChange={setCost('initial_capital')} /></Field>
          <Field label="Position size (fraction)"><input type="number" step="0.01" placeholder="default" value={costs.position_pct} onChange={setCost('position_pct')} /></Field>
          <Field label="Brokerage per order"><input type="number" step="0.5" placeholder="default" value={costs.brokerage_per_order} onChange={setCost('brokerage_per_order')} /></Field>
          <Field label="Taxes &amp; charges (fraction)"><input type="number" step="0.0001" placeholder="default" value={costs.charges_pct} onChange={setCost('charges_pct')} /></Field>
          <Field label="Slippage (points)"><input type="number" step="0.1" placeholder="default" value={costs.slippage_points} onChange={setCost('slippage_points')} /></Field>
        </div>
        <div className="info-box">
          Active Target: {hasRunnableTarget
            ? `${strategy.strategy_name ? `[${strategy.strategy_name}] ` : ''}${specText(strategy)}`
            : 'none selected. Register an indicator/strategy in the Studios or select one above.'}
        </div>
      </Card>
      {m && (
        <>
          <div className="grid stats">
            <Stat label="Final balance" value={fmtMoney(m.final_capital)} tone={m.net_profit >= 0 ? 'good' : 'bad'} />
            <Stat label="Net profit" value={fmtMoney(m.net_profit)} tone={m.net_profit >= 0 ? 'good' : 'bad'} hint={fmtPct(m.return_pct)} />
            <Stat label="Sharpe ratio" value={fmtNum(m.sharpe_ratio ?? 0)} />
            <Stat label="Win rate" value={fmtPct(m.win_rate)} hint={`${m.trades} trades`} />
            <Stat label="Profit factor" value={fmtNum(m.profit_factor)} />
            <Stat label="Max drawdown" value={fmtMoney(m.max_drawdown)} hint={fmtPct(m.max_drawdown_pct)} tone="bad" />
            <Stat label="Costs paid" value={fmtMoney(m.total_costs)} />
            <Stat label="Skipped entries" value={m.skipped_entries} />
          </div>
          <Card title={`Price, entries and exits (${result.split})`}>
            {candleData ? <Plot data={candleData} layout={{ xaxis: { rangeslider: { visible: false } } }} height={420} /> : <div className="empty">No candles in this split.</div>}
          </Card>
          <Card title="Equity and drawdown">
            {equityData ? <Plot data={equityData} layout={{ yaxis2: { overlaying: 'y', side: 'right', showgrid: false, title: 'Drawdown' } }} height={300} /> : <div className="empty">No equity curve.</div>}
          </Card>
          <Card title={`Trade history (${markers.length})`}>
            <Table
              rowKey={(_t, i) => i}
              columns={[
                { key: 'side', label: 'Side' },
                { key: 'entry_time', label: 'Entry' },
                { key: 'entry_price', label: 'Entry px', align: 'right', render: (t) => fmtNum(t.entry_price) },
                { key: 'exit_time', label: 'Exit' },
                { key: 'exit_price', label: 'Exit px', align: 'right', render: (t) => fmtNum(t.exit_price) },
                { key: 'quantity', label: 'Qty', align: 'right' },
                { key: 'gross_pnl', label: 'Gross', align: 'right', render: (t) => fmtMoney(t.gross_pnl) },
                { key: 'costs', label: 'Costs', align: 'right', render: (t) => fmtMoney(t.costs) },
                { key: 'net_pnl', label: 'Net', align: 'right', render: (t) => <span style={{ color: t.net_pnl >= 0 ? 'var(--good)' : 'var(--bad)' }}>{fmtMoney(t.net_pnl)}</span> },
                { key: 'reason', label: 'Reason' },
              ]}
              rows={markers.slice(0, 500)}
              empty="No trades in this split."
            />
          </Card>
        </>
      )}
      <Card title="Out-of-sample test">
        <p className="muted" style={{ marginTop: 0 }}>
          The out-of-sample tail is never used by the agents. Running it is a one-time action per strategy and dataset, and the
          result is stored permanently.
        </p>
        {!confirmOos
          ? <button className="danger" disabled={busy || !hasRunnableTarget || !datasetId} onClick={() => setConfirmOos(true)}>Run out-of-sample (one-time)</button>
          : <div className="pill-row">
              <span>Confirm: this result will be stored and cannot be re-run.</span>
              <button className="danger" disabled={busy} onClick={runOos}>Yes, run it</button>
              <button onClick={() => setConfirmOos(false)}>Cancel</button>
            </div>}
        {oos && <div className="info-box">
          {oos.already_run ? `Already run on ${new Date(oos.created_at).toLocaleString()}. ` : 'Run now. '}
          Net profit {fmtMoney(oos.metrics.net_profit)} · Sharpe {fmtNum(oos.metrics.sharpe_ratio ?? 0)} · {oos.metrics.trades} trades · win rate {fmtPct(oos.metrics.win_rate)} · max drawdown {fmtPct(oos.metrics.max_drawdown_pct)}
        </div>}
      </Card>
    </>
  );
}
