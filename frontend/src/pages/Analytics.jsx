import { useEffect, useState } from 'react';
import { get, fmtMoney, fmtPct, fmtTime } from '../api.js';
import { useApp } from '../context.js';
import { Card, Table, Badge, ErrorBox, useAsync, Field } from '../components/ui.jsx';
import Plot from '../components/Plot.jsx';

const OPTUNA_CHARTS = [
  ['history', 'Optimization history'], ['importance', 'Parameter importance'],
  ['contour', 'Contour'], ['slice', 'Slice'], ['parallel', 'Parallel coordinates'], ['pareto', 'Pareto front'],
];

export default function Analytics() {
  const { datasetId } = useApp();
  const [runs, setRuns] = useState([]);
  const [study, setStudy] = useState('');
  const [chart, setChart] = useState('history');
  const [figure, setFigure] = useState(null);
  const [chartError, setChartError] = useState(null);
  const [rankings, setRankings] = useState(null);
  const [wf, setWf] = useState([]);
  const [tvv, setTvv] = useState([]);
  const [equity, setEquity] = useState(null);
  const [strategyHash, setStrategyHash] = useState('');
  const [split, setSplit] = useState('validation');
  const { error, run } = useAsync();

  useEffect(() => { get('/api/optimization/runs').then((r) => { setRuns(r); if (r.length && !study) setStudy(r[0].study_name); }).catch(() => {}); }, []);

  useEffect(() => {
    if (!study) return;
    setChartError(null);
    get(`/api/analytics/optuna/${study}/${chart}`).then(setFigure).catch((e) => { setFigure(null); setChartError(e.message); });
  }, [study, chart]);

  useEffect(() => {
    if (!datasetId) return;
    run(async () => {
      const [b, w, t] = await Promise.all([
        get(`/api/results/rankings/profit?dataset_id=${datasetId}&top=10`),
        get(`/api/analytics/walk_forward?dataset_id=${datasetId}`),
        get(`/api/analytics/train_vs_validation?dataset_id=${datasetId}`),
      ]);
      setRankings(b); setWf(w); setTvv(t);
      if (!strategyHash && b.length) setStrategyHash(b[0].strategy_hash);
    });
  }, [datasetId]);

  useEffect(() => {
    if (!datasetId || !strategyHash) return;
    run(async () => setEquity(await get(`/api/analytics/equity?dataset_id=${datasetId}&strategy_hash=${strategyHash}&split=${split}`)));
  }, [datasetId, strategyHash, split]);

  const scatter = tvv.length ? [{
    type: 'scatter', mode: 'markers', x: tvv.map((r) => r.train_net_profit), y: tvv.map((r) => r.validation_net_profit),
    text: tvv.map((r) => r.strategy_hash), marker: { color: tvv.map((r) => (r.validated ? '#22c55e' : '#f5b942')), size: 9 },
    name: 'Strategies',
  }] : null;

  const eqData = equity && equity.timestamps.length ? [
    { type: 'scatter', mode: 'lines', name: 'Equity', x: equity.timestamps, y: equity.equity, line: { color: '#4f8cff' } },
    { type: 'scatter', mode: 'lines', name: 'Drawdown', x: equity.timestamps, y: equity.drawdown.map((v) => -v), yaxis: 'y2', line: { color: '#ef4444' } },
  ] : null;

  return (
    <>
      <div className="page-head"><div><h1>Analytics</h1><div className="muted">Optuna visualizations, equity and drawdown, walk-forward results, and training versus validation.</div></div></div>
      <ErrorBox error={error} />
      <Card title="Optuna visualizations" actions={
        <>
          <select value={study} onChange={(e) => setStudy(e.target.value)}>
            <option value="">Choose a study</option>
            {runs.map((r) => <option key={r.study_name} value={r.study_name}>{r.study_name}</option>)}
          </select>
          <select value={chart} onChange={(e) => setChart(e.target.value)}>
            {OPTUNA_CHARTS.map(([k, label]) => <option key={k} value={k}>{label}</option>)}
          </select>
        </>}>
        {!study && <div className="empty">Run an optimization study to see Optuna charts.</div>}
        {chartError && <div className="info-box">{chartError}</div>}
        {figure && figure.data && <Plot data={figure.data} layout={figure.layout} height={380} />}
      </Card>
      <div className="split-2">
        <Card title="Equity and drawdown" actions={
          <>
            <select value={strategyHash} onChange={(e) => setStrategyHash(e.target.value)}>
              {(rankings || []).map((r) => <option key={r.strategy_hash} value={r.strategy_hash}>#{r.rank} {r.strategy_hash}</option>)}
            </select>
            <select value={split} onChange={(e) => setSplit(e.target.value)}>
              <option value="train">Train</option><option value="validation">Validation</option>
            </select>
          </>}>
          {eqData ? <Plot data={eqData} layout={{ yaxis2: { overlaying: 'y', side: 'right', showgrid: false } }} height={300} />
            : <div className="empty">Select a strategy from the top-10 profit ranking.</div>}
        </Card>
        <Card title="Training vs validation performance">
          {scatter ? <Plot data={scatter} layout={{ xaxis: { title: 'Train net profit' }, yaxis: { title: 'Validation net profit' } }} height={300} />
            : <div className="empty">Needs strategies that have been walk-forward validated.</div>}
        </Card>
      </div>
      <Card title="Walk-forward validation">
        <Table
          rowKey={(r) => r.strategy_hash}
          columns={[
            { key: 'strategy_hash', label: 'Strategy', render: (r) => <span className="mono">{r.strategy_hash}</span> },
            { key: 'created_at', label: 'Checked', render: (r) => fmtTime(r.created_at) },
            { key: 'profitable_segments', label: 'Profitable segments', align: 'right', render: (r) => `${r.profitable_segments} / ${r.total_segments}` },
            { key: 'validation_net_profit', label: 'Validation net', align: 'right', render: (r) => fmtMoney(r.validation_net_profit) },
            { key: 'validated', label: 'Status', render: (r) => <Badge status={r.validated ? 'validated' : 'failed'} /> },
          ]}
          rows={wf}
          empty="No walk-forward checks yet."
        />
      </Card>
    </>
  );
}
