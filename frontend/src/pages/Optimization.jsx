import { useEffect, useState } from 'react';
import { get, post, fmtTime } from '../api.js';
import { useApp } from '../context.js';
import { Card, Badge, Field, ErrorBox, Table, useAsync } from '../components/ui.jsx';
import Plot from '../components/Plot.jsx';

export default function Optimization() {
  const { datasetId } = useApp();
  const [options, setOptions] = useState(null);
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState(null);
  const [summary, setSummary] = useState(null);
  const [history, setHistory] = useState(null);
  const [importance, setImportance] = useState(null);
  const [form, setForm] = useState({
    study_name: 'study_1',
    n_trials: 40,
    duration_limit_s: 0,
    sampler: 'tpe',
    pruner: 'median',
    objective: 'net_profit',
    n_jobs: 1,
    cpu_limit_pct: 85,
    ram_limit_mb: 256,
    max_indicators: 0,
    choices_per_param: 5,
    seed: 42,
  });
  const { error, busy, run } = useAsync();

  const refresh = async () => {
    try {
      setRuns(await get('/api/optimization/runs'));
    } catch { /* keep last list */ }
  };

  useEffect(() => {
    get('/api/optimization/options').then(setOptions).catch(() => {});
    refresh();
    const t = setInterval(refresh, 3000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!selected) return;
    setSummary(null); setHistory(null); setImportance(null);
    get(`/api/optimization/studies/${selected}`).then(setSummary).catch(() => {});
    get(`/api/analytics/optuna/${selected}/history`).then(setHistory).catch(() => {});
    get(`/api/analytics/optuna/${selected}/importance`).then(setImportance).catch(() => setImportance(null));
  }, [selected]);

  const start = () => run(async () => {
    const r = await post('/api/optimization/start', {
      dataset_id: datasetId,
      ...form,
      n_trials: Number(form.n_trials),
      duration_limit_s: Number(form.duration_limit_s),
      max_indicators: Number(form.max_indicators),
      choices_per_param: Number(form.choices_per_param),
      n_jobs: Number(form.n_jobs),
      cpu_limit_pct: Number(form.cpu_limit_pct),
      ram_limit_mb: Number(form.ram_limit_mb),
      seed: Number(form.seed),
    });
    setSelected(r.study_name.replace('optuna:', ''));
    await refresh();
  });
  const control = (study, action) => run(async () => { await post(`/api/optimization/${study}/${action}`); await refresh(); });
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value });
  const active = runs.some((r) => r.live && r.live.alive);
  const cat = options?.catalogue || { indicators: 0, strategies: 0 };

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Optimization</h1>
          <div className="muted">
            Systematic combination &amp; parameter search over all registered indicators ({cat.indicators}) and strategies ({cat.strategies}) with pruning and resource budgets.
          </div>
        </div>
      </div>
      <ErrorBox error={error} />
      <Card title="New study" actions={<button className="primary" disabled={busy || active || !datasetId} onClick={start}>Start study</button>}>
        <div className="form-grid">
          <Field label="Study name"><input value={form.study_name} onChange={set('study_name')} /></Field>
          <Field label="Trials"><input type="number" min="1" value={form.n_trials} onChange={set('n_trials')} /></Field>
          <Field label="Duration budget (sec)" hint="0 = unlimited"><input type="number" min="0" value={form.duration_limit_s} onChange={set('duration_limit_s')} /></Field>
          <Field label="Sampler / Algorithm">
            <select value={form.sampler} onChange={set('sampler')}>
              {(options?.samplers || ['tpe']).map((s) => <option key={s} value={s}>{s.toUpperCase()}</option>)}
            </select>
          </Field>
          <Field label="Pruner">
            <select value={form.pruner} onChange={set('pruner')}>
              {(options?.pruners || ['median']).map((p) => <option key={p} value={p}>{p}</option>)}
            </select>
          </Field>
          <Field label="Objective">
            <select value={form.objective} onChange={set('objective')}>
              {(options?.objectives || ['net_profit']).map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
          </Field>
          <Field label="Max indicators / combo" hint="0 = unlimited (all registered)"><input type="number" min="0" value={form.max_indicators} onChange={set('max_indicators')} /></Field>
          <Field label="CPU limit (%)"><input type="number" min="10" max="100" value={form.cpu_limit_pct} onChange={set('cpu_limit_pct')} /></Field>
          <Field label="Memory limit (MB)"><input type="number" min="64" max="4096" step="64" value={form.ram_limit_mb} onChange={set('ram_limit_mb')} /></Field>
          <Field label="Parallel workers (n_jobs)"><input type="number" min="1" value={form.n_jobs} onChange={set('n_jobs')} /></Field>
          <Field label="Choices per parameter"><input type="number" min="2" max="20" value={form.choices_per_param} onChange={set('choices_per_param')} /></Field>
          <Field label="Seed"><input type="number" value={form.seed} onChange={set('seed')} /></Field>
        </div>
        {options?.notes && (
          <div className="info-box">
            Grid: {options.notes.grid}. CMA-ES: {options.notes.cmaes}. NSGA-II: {options.notes.nsga2}.
          </div>
        )}
      </Card>
      <Card title="Study runs">
        <Table
          rowKey={(r) => r.study_name}
          selectedKey={selected}
          onRowClick={(r) => setSelected(r.study_name)}
          columns={[
            { key: 'study_name', label: 'Study' },
            { key: 'status', label: 'Status', render: (r) => <Badge status={r.live?.alive ? r.live.status : r.status} /> },
            { key: 'sampler', label: 'Sampler', render: (r) => (r.config.sampler || '').toUpperCase() },
            { key: 'objective', label: 'Objective', render: (r) => r.config.objective },
            { key: 'n_trials', label: 'Trials', align: 'right', render: (r) => r.config.n_trials },
            { key: 'created_at', label: 'Created', render: (r) => fmtTime(r.created_at) },
            {
              key: 'controls',
              label: '',
              render: (r) => r.live?.alive ? (
                <div className="pill-row" onClick={(e) => e.stopPropagation()}>
                  <button disabled={busy || r.live.status === 'paused'} onClick={() => control(r.study_name, 'pause')}>Pause</button>
                  <button disabled={busy || r.live.status !== 'paused'} onClick={() => control(r.study_name, 'resume')}>Resume</button>
                  <button className="danger" disabled={busy} onClick={() => control(r.study_name, 'stop')}>Stop</button>
                </div>
              ) : null,
            },
          ]}
          rows={runs}
          empty="No optimization runs yet."
        />
        {runs.find((r) => r.study_name === selected)?.live && (
          <div style={{ marginTop: 12 }}>
            <div className="muted">Progress {Math.round((runs.find((r) => r.study_name === selected).live.progress || 0) * 100)}% · {runs.find((r) => r.study_name === selected).live.current_task}</div>
            <div className="progress"><div style={{ width: `${Math.round((runs.find((r) => r.study_name === selected).live.progress || 0) * 100)}%` }} /></div>
          </div>
        )}
      </Card>
      {selected && summary && (
        <div className="split-2">
          <Card title={`Best result · ${selected}`}>
            <dl className="kv">
              <dt>Trials</dt><dd>{summary.trials}</dd>
              <dt>Complete</dt><dd>{summary.states.complete || 0}</dd>
              <dt>Pruned</dt><dd>{summary.states.pruned || 0}</dd>
              <dt>Failed</dt><dd>{summary.states.failed || 0}</dd>
              <dt>Running</dt><dd>{summary.states.running || 0}</dd>
              <dt>Best value</dt><dd>{summary.best ? Number(summary.best.value).toFixed(2) : '—'}</dd>
              <dt>Strategy</dt><dd className="mono">{summary.best?.params || '—'}</dd>
            </dl>
          </Card>
          <Card title="Optimization history">
            {history && history.data ? <Plot data={history.data} layout={history.layout} height={300} /> : <div className="empty">Needs completed trials.</div>}
          </Card>
        </div>
      )}
      {selected && importance && importance.data && (
        <Card title="Indicator impact">
          <Plot data={importance.data} layout={importance.layout} height={320} />
        </Card>
      )}
    </>
  );
}
