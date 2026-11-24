import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { OTHER_PAGES } from '../navigation.js';
import { get, post } from '../api.js';
import { useApp } from '../context.js';
import { Card, Stat, Badge, ErrorBox, Table } from '../components/ui.jsx';
import Plot from '../components/Plot.jsx';
import IndicatorTable from '../components/IndicatorTable.jsx';
import Robustness from '../components/Robustness.jsx';
import { ParetoCard, ClustersCard } from '../components/ParetoClusters.jsx';

const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));
const fmtMoney = (v) => (v === null || v === undefined ? '—' : Number(v).toLocaleString('en-IN', { maximumFractionDigits: 2, minimumFractionDigits: 2 }));
const names = (members) => (members || []).map((m) => m.name).join(' + ');

export default function Dashboard() {
  const { datasetId, datasets } = useApp();
  const dataset = datasets.find((x) => x.id === datasetId);
  const [d, setD] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState(null);
  const [oosBusy, setOosBusy] = useState(false);

  const load = useCallback(async () => {
    if (!datasetId) return;
    try {
      const [dash, an] = await Promise.all([get(`/api/research/dashboard?dataset_id=${datasetId}`), get(`/api/research/analysis?dataset_id=${datasetId}`)]);
      setD(dash); setAnalysis(an); setError(null);
    } catch (e) { setError(e.message); }
  }, [datasetId]);

  useEffect(() => { setD(null); load(); const t = setInterval(load, 5000); return () => clearInterval(t); }, [load]);

  const runOos = async (runId) => {
    if (!window.confirm('Run the one-shot out-of-sample test for this run\'s frozen combination?\n\nIt can be run once per run. The result cannot be repeated or re-tuned.')) return;
    setOosBusy(true); setError(null);
    try { await post(`/api/research/oos/${runId}`, {}); await load(); } catch (e) { setError(e.message); } finally { setOosBusy(false); }
  };

  if (!datasetId) return <div className="page"><h1>Dashboard</h1><div className="empty">Import or select a dataset in the top bar.</div></div>;
  if (!d) return <div className="page"><h1>Dashboard</h1><ErrorBox error={error} />{!error && <div className="muted">Loading…</div>}</div>;

  const ind = d.individual;
  const comb = d.combination;
  const top = comb?.rankings || [];
  const best = top[0];
  const oos = comb?.oos;

  return (
    <div className="page research-dashboard">
      <div className="page-head">
        <div>
          <h1>Dashboard</h1>
          <div className="muted">All figures are calculated by the backend from stored runs. Train split unless marked OOS.</div>
        </div>
        <Link to="/indicator" className="ghost-link">Open Indicator research →</Link>
      </div>
      <ErrorBox error={error} />
      {dataset?.is_synthetic ? <div className="warn-box">Synthetic dataset selected: every figure here is a controlled test result, not market performance.</div> : null}

      <div className="grid stats">
        <Stat label="Indicators discovered" value={ind.rows.length} hint={`${ind.counts.unavailable ?? 0} unavailable`} />
        <Stat label="Evaluated" value={ind.counts.evaluated ?? 0} hint={ind.run ? `run #${ind.run.id}` : 'no run yet'} tone="good" />
        <Stat label="Best Wilson LB" value={best ? fmt(best.metrics.wilson_lb, 4) : '—'} hint={best ? names(best.members) : 'no combination yet'} />
        <Stat label="Combinations ranked" value={comb?.selection?.coverage?.combinations_ranked ?? '—'} hint={comb?.run ? `run #${comb.run.id}` : 'no combination run yet'} />
        <Stat label="OOS" value={oos ? fmtMoney(oos.metrics.net_profit) : (comb?.frozen ? 'not run' : '—')} hint={oos ? `${oos.metrics.trades} trades, net profit` : 'one-shot, untouched tail'} />
      </div>

      <div className="grid two">
        <Card title="Individual evaluation">
          {ind.run ? <RunLine run={ind.run} /> : <div className="empty">No individual run yet. Start one on the Indicator page.</div>}
        </Card>
        <Card title="Combination research">
          {comb?.run ? <RunLine run={comb.run} /> : <div className="empty">No combination run yet. Start one on the Indicator page.</div>}
        </Card>
      </div>

      <Card title="Indicator ranking" actions={<span className="muted">sorted by Wilson lower bound (default)</span>}>
        {ind.rows.length ? <IndicatorTable rows={ind.rows} csvName="indicator_ranking.csv" /> : <div className="empty">No indicators discovered.</div>}
      </Card>

      <Card title="Top combinations (train selection)">
        <Table
          rowKey={(r) => r.combo_key}
          empty="No combination has reached the minimum trades yet."
          columns={[
            { key: 'rank', label: 'Rank' },
            { key: 'members', label: 'Members', render: (r) => names(r.members) },
            { key: 'size', label: 'Size' },
            { key: 'trades', label: 'Trades', align: 'num', render: (r) => r.metrics.trades },
            { key: 'win', label: 'Win %', align: 'num', render: (r) => fmt(r.metrics.raw_win_rate) },
            { key: 'wilson', label: 'Wilson LB', align: 'num', render: (r) => fmt(r.metrics.wilson_lb, 4) },
            { key: 'shrunk', label: 'Shrunk score', align: 'num', render: (r) => fmt(r.metrics.shrunk_score, 4) },
            { key: 'net', label: 'Net profit', align: 'num', render: (r) => <span className={r.metrics.net_profit >= 0 ? 'pos' : 'neg'}>{fmtMoney(r.metrics.net_profit)}</span> },
            { key: 'pf', label: 'Profit factor', align: 'num', render: (r) => fmt(r.metrics.profit_factor) },
            { key: 'dd', label: 'Max DD %', align: 'num', render: (r) => fmt(r.metrics.max_drawdown_pct) },
          ]}
          rows={top}
        />
        <small className="muted">Ranking is Wilson lower bound on the win rate, then shrunk score, then net profit. Net profit is shown but does not change the order.</small>
      </Card>

      <div className="grid two">
        <CorrelationCard corr={comb?.correlation} />
        <WalkForwardCard folds={comb?.folds || []} />
      </div>

      <div className="grid two">
        <OosCard comb={comb} oos={oos} busy={oosBusy} onRun={runOos} />
        <Card title="Coverage">
          {comb?.selection ? <Coverage sel={comb.selection} /> : <div className="empty">Run a combination search to see coverage.</div>}
        </Card>
      </div>

      <div className="grid two">
        <Robustness combinationRunId={comb?.run?.id} datasetSynthetic={!!dataset?.is_synthetic} />
        <ParetoCard analysis={analysis} />
      </div>

      <ClustersCard analysis={analysis} />

      <EquityCard comb={comb} />

      <Card title="Other pages">
        <div className="other-links">{OTHER_PAGES.map(([to, label]) => <Link key={to} to={to}>{label}</Link>)}</div>
      </Card>
    </div>
  );
}

function RunLine({ run }) {
  return (
    <div className="run-line">
      <Badge status={run.status} />
      <span className="muted">{run.message || ''}</span>
      <span className="muted">{run.progress != null ? `${Math.round(run.progress * 100)}%` : ''}</span>
      {run.error && <div className="error-box">{run.error}</div>}
    </div>
  );
}

function CorrelationCard({ corr }) {
  const figure = useMemo(() => {
    if (!corr || !corr.names?.length) return null;
    return {
      data: [{
        type: 'heatmap', z: corr.matrix, x: corr.names, y: corr.names, zmin: -1, zmax: 1,
        colorscale: [[0, '#2563eb'], [0.5, '#0f172a'], [1, '#f59e0b']],
        texttemplate: '%{z:.2f}', hovertemplate: '%{y} vs %{x}: %{z:.3f}<extra></extra>',
        colorbar: { title: 'Spearman ρ', thickness: 12 },
      }],
      layout: { margin: { l: 120, r: 40, t: 12, b: 120 }, xaxis: { tickangle: -40 } },
    };
  }, [corr]);
  return (
    <Card title="Correlation heatmap (Spearman, filter pool)">
      {figure ? <Plot data={figure.data} layout={figure.layout} height={Math.max(300, 40 * corr.names.length + 160)} />
        : <div className="empty">Available after a combination run finishes.</div>}
      {corr && <small className="muted">Indicators at or above |ρ| {corr.threshold} were dropped from the pool.</small>}
    </Card>
  );
}

function WalkForwardCard({ folds }) {
  return (
    <Card title="Walk-forward folds">
      <Table
        rowKey={(f) => f.fold}
        empty="Available after a combination run starts folds."
        rows={folds}
        columns={[
          { key: 'fold', label: 'Fold' },
          { key: 'sel', label: 'Selected on', render: (f) => <span className="nowrap">{f.selection_range.start.slice(0, 7)} → {f.selection_range.end.slice(0, 7)}</span> },
          { key: 'test', label: 'Tested on', render: (f) => <span className="nowrap">{f.test_range.start.slice(0, 7)} → {f.test_range.end.slice(0, 7)}</span> },
          { key: 'sel_members', label: 'Selected combination', render: (f) => names(f.selected_members) || '—' },
          { key: 'test_trades', label: 'Test trades', align: 'num', render: (f) => f.test_metrics?.trades ?? '—' },
          { key: 'test_net', label: 'Test net profit', align: 'num', render: (f) => (f.test_metrics ? fmtMoney(f.test_metrics.net_profit) : '—') },
          { key: 'status', label: 'Status', render: (f) => <Badge status={f.status} /> },
        ]}
      />
      <small className="muted">Each fold chooses on data before its test chunk, with a purge gap. The test chunk never influences that choice.</small>
    </Card>
  );
}

function OosCard({ comb, oos, busy, onRun }) {
  if (!comb?.run) return <Card title="Out-of-sample (one-shot)"><div className="empty">No combination run yet.</div></Card>;
  if (!comb.frozen) return <Card title="Out-of-sample (one-shot)"><div className="empty">The run froze no combination (none reached the minimum trades).</div></Card>;
  const canRun = comb.run.status === 'completed' && !oos;
  return (
    <Card title="Out-of-sample (one-shot)" actions={canRun ? <button className="primary" disabled={busy} onClick={() => onRun(comb.run.id)}>Run OOS test</button> : null}>
      <div className="kv-grid">
        <div><b>Frozen combination</b> {names(comb.frozen.members)}</div>
        <div><b>Size</b> {comb.frozen.size}</div>
        <div><b>Frozen at</b> {comb.frozen.frozen_at}</div>
      </div>
      {oos ? (
        <>
          <div className="kv-grid">
            <div><b>OOS window</b> {oos.window.start} → {oos.window.end}</div>
            <div><b>Bars</b> {oos.window.bars}</div>
            <div><b>Trades</b> {oos.metrics.trades}</div>
            <div><b>Win %</b> {fmt(oos.metrics.raw_win_rate)}</div>
            <div><b>Net profit</b> <span className={oos.metrics.net_profit >= 0 ? 'pos' : 'neg'}>{fmtMoney(oos.metrics.net_profit)}</span></div>
            <div><b>Profit factor</b> {fmt(oos.metrics.profit_factor)}</div>
            <div><b>Max drawdown %</b> {fmt(oos.metrics.max_drawdown_pct)}</div>
            <div><b>Wilson LB</b> {fmt(oos.metrics.wilson_lb, 4)}</div>
          </div>
          <small className="muted">Evaluated once on {oos.created_at}. It cannot be repeated for this run.</small>
        </>
      ) : <small className="muted">Not evaluated yet. The test runs once and cannot be repeated.</small>}
    </Card>
  );
}

function Coverage({ sel }) {
  const c = sel.coverage;
  return (
    <div className="kv-grid">
      <div><b>Indicators available</b> {c.indicators_available}</div>
      <div><b>Eligible</b> {c.eligible}</div>
      <div><b>Dropped (Spearman)</b> {c.dropped_by_correlation}</div>
      <div><b>Pool</b> {c.pool_size}</div>
      <div><b>Pairs</b> {c.pairs_evaluated} / {c.pairs_possible}</div>
      <div><b>Triples</b> {c.triples_evaluated} / {c.triples_possible}</div>
      <div><b>SFFS evaluations</b> {c.sffs_evaluations}</div>
      <div><b>Budget reached</b> {c.budget_reached ? 'yes (partial)' : 'no'}</div>
    </div>
  );
}

function EquityCard({ comb }) {
  const train = comb?.train_curve;
  const oos = comb?.oos?.curve;
  const figure = useMemo(() => {
    const traces = [];
    if (train) {
      traces.push({ type: 'scatter', mode: 'lines', name: 'Equity (train, frozen)', x: train.time, y: train.equity, line: { color: '#3b82f6' }, xaxis: 'x', yaxis: 'y' });
      traces.push({ type: 'scatter', mode: 'lines', name: 'Drawdown % (train)', x: train.time, y: train.drawdown_pct, line: { color: '#ef4444' }, fill: 'tozeroy', xaxis: 'x2', yaxis: 'y2' });
    }
    if (oos) {
      traces.push({ type: 'scatter', mode: 'lines', name: 'Equity (OOS)', x: oos.time, y: oos.equity, line: { color: '#22c55e' }, xaxis: 'x3', yaxis: 'y3' });
      traces.push({ type: 'scatter', mode: 'lines', name: 'Drawdown % (OOS)', x: oos.time, y: oos.drawdown_pct, line: { color: '#f97316' }, fill: 'tozeroy', xaxis: 'x4', yaxis: 'y4' });
    }
    return {
      data: traces,
      layout: {
        xaxis: { domain: [0, 0.48], anchor: 'y' }, yaxis: { domain: [0.55, 1], title: 'Equity' },
        xaxis2: { domain: [0, 0.48], anchor: 'y2' }, yaxis2: { domain: [0, 0.4], title: 'Drawdown %' },
        xaxis3: { domain: [0.52, 1], anchor: 'y3' }, yaxis3: { domain: [0.55, 1], anchor: 'x3' },
        xaxis4: { domain: [0.52, 1], anchor: 'y4' }, yaxis4: { domain: [0, 0.4], anchor: 'x4' },
        showlegend: true, margin: { t: 40 }, annotations: [
          { text: 'Train (frozen combination)', x: 0.24, y: 1.02, xref: 'paper', yref: 'paper', showarrow: false, font: { size: 12 } },
          { text: 'Out-of-sample', x: 0.76, y: 1.02, xref: 'paper', yref: 'paper', showarrow: false, font: { size: 12 } },
        ],
      },
    };
  }, [train, oos]);
  return (
    <Card title="Equity curve and drawdown">
      {figure.data.length > 0 ? <Plot data={figure.data} layout={figure.layout} height={520} />
        : <div className="empty">Available after a combination run finishes with a frozen combination.</div>}
      <small className="muted">Train curve: the frozen combination on the full train span. OOS curve appears only after the one-shot test.</small>
    </Card>
  );
}
