import { useEffect, useState } from 'react';
import { get, fmtMoney, fmtPct, fmtNum, fmtTime } from '../api.js';
import { useApp } from '../context.js';
import { Card, Badge, Table, ErrorBox, useAsync, Field } from '../components/ui.jsx';

export default function Experiments() {
  const { datasetId } = useApp();
  const [status, setStatus] = useState('');
  const [study, setStudy] = useState('');
  const [data, setData] = useState({ total: 0, rows: [] });
  const [page, setPage] = useState(0);
  const [compare, setCompare] = useState([]);
  const [runs, setRuns] = useState([]);
  const { error, run } = useAsync();
  const PAGE = 50;

  useEffect(() => { get('/api/optimization/runs').then(setRuns).catch(() => {}); }, []);

  useEffect(() => {
    const q = new URLSearchParams({ limit: String(PAGE), offset: String(page * PAGE) });
    if (datasetId) q.set('dataset_id', String(datasetId));
    if (status) q.set('status', status);
    if (study) q.set('study', study);
    run(async () => setData(await get(`/api/experiments?${q.toString()}`)));
  }, [datasetId, status, study, page]);

  const toggle = (row) => setCompare((c) => (c.some((x) => x.id === row.id)
    ? c.filter((x) => x.id !== row.id) : [...c.slice(-2), row]));

  const pages = Math.max(1, Math.ceil(data.total / PAGE));
  return (
    <>
      <div className="page-head"><div><h1>Experiments</h1><div className="muted">Every evaluation is stored once. Select up to three rows to compare.</div></div></div>
      <ErrorBox error={error} />
      <Card>
        <div className="form-grid" style={{ marginBottom: 12 }}>
          <Field label="Status">
            <select value={status} onChange={(e) => { setPage(0); setStatus(e.target.value); }}>
              <option value="">All</option>
              {['completed', 'pruned', 'failed'].map((s) => <option key={s} value={s}>{s}</option>)}
            </select>
          </Field>
          <Field label="Study">
            <select value={study} onChange={(e) => { setPage(0); setStudy(e.target.value); }}>
              <option value="">All studies</option>
              {runs.map((r) => <option key={r.study_name} value={r.study_name}>{r.study_name}</option>)}
            </select>
          </Field>
        </div>
        <Table
          rowKey={(r) => r.id}
          onRowClick={toggle}
          selectedKey={null}
          columns={[
            { key: 'sel', label: '', render: (r) => <input type="checkbox" readOnly checked={compare.some((c) => c.id === r.id)} /> },
            { key: 'id', label: 'ID', align: 'right' },
            { key: 'trial_number', label: 'Trial', align: 'right', render: (r) => r.trial_number ?? '—' },
            { key: 'study_name', label: 'Study', render: (r) => r.study_name || '—' },
            { key: 'description', label: 'Combination & parameters', render: (r) => <span className="mono">{r.description}</span> },
            { key: 'status', label: 'Status', render: (r) => <Badge status={r.status} /> },
            { key: 'net', label: 'Net profit', align: 'right', render: (r) => r.metrics ? fmtMoney(r.metrics.net_profit) : '—' },
            { key: 'trades', label: 'Trades', align: 'right', render: (r) => r.metrics?.trades ?? '—' },
            { key: 'duration_s', label: 'Duration', align: 'right', render: (r) => r.duration_s == null ? '—' : `${fmtNum(r.duration_s, 2)}s` },
            { key: 'split', label: 'Split' },
            { key: 'created_at', label: 'Recorded', render: (r) => fmtTime(r.created_at) },
          ]}
          rows={data.rows}
          empty="No experiments match these filters."
        />
        <div className="pill-row" style={{ marginTop: 10, justifyContent: 'space-between' }}>
          <span className="muted">{data.total} experiments</span>
          <span className="pill-row">
            <button disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button>
            <span className="muted">Page {page + 1} of {pages}</span>
            <button disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>Next</button>
          </span>
        </div>
      </Card>
      {compare.length > 0 && (
        <Card title="Comparison" actions={<button className="ghost" onClick={() => setCompare([])}>Clear</button>}>
          <Table
            rowKey={(r) => r.id}
            columns={[
              { key: 'id', label: 'ID' },
              { key: 'description', label: 'Strategy', render: (r) => <span className="mono">{r.description}</span> },
              { key: 'net', label: 'Net', align: 'right', render: (r) => fmtMoney(r.metrics?.net_profit) },
              { key: 'win', label: 'Win %', align: 'right', render: (r) => fmtPct(r.metrics?.win_rate) },
              { key: 'pf', label: 'PF', align: 'right', render: (r) => fmtNum(r.metrics?.profit_factor) },
              { key: 'dd', label: 'Max DD %', align: 'right', render: (r) => fmtPct(r.metrics?.max_drawdown_pct) },
              { key: 'tr', label: 'Trades', align: 'right', render: (r) => r.metrics?.trades },
              { key: 'costs', label: 'Costs', align: 'right', render: (r) => fmtMoney(r.metrics?.total_costs) },
            ]}
            rows={compare}
          />
        </Card>
      )}
    </>
  );
}
