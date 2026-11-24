import { useEffect, useState } from 'react';
import { get, post } from '../api.js';
import { Card, Badge } from './ui.jsx';

// Statistical robustness for one finished combination run. Runs on the train span only; OOS is never used.
const pv = (v) => (v === null || v === undefined ? '—' : Number(v).toFixed(4));
const money = (v) => (v === null || v === undefined ? '—' : Number(v).toLocaleString('en-IN', { maximumFractionDigits: 0 }));

export default function Robustness({ combinationRunId, datasetSynthetic }) {
  const [state, setState] = useState(null);     // {run_id, status, error, result}
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const load = async () => {
    if (!combinationRunId) return null;
    const r = await get(`/api/research/robustness?combination_run_id=${combinationRunId}`);
    setState(r);
    return r;
  };
  useEffect(() => { setState(null); load().catch((e) => setError(e.message)); /* eslint-disable-next-line */ }, [combinationRunId]);
  useEffect(() => {
    if (!state || (state.status !== 'running' && state.status !== 'paused')) return undefined;
    const t = setInterval(() => load().catch(() => {}), 2000);
    return () => clearInterval(t);
    // eslint-disable-next-line
  }, [state?.status, combinationRunId]);

  const start = async () => {
    setBusy(true); setError(null);
    try {
      await post('/api/research/robustness/start', { combination_run_id: combinationRunId, candidates: 10, permutations: 200, mc_iterations: 5000, spa_draws: 500 });
      await load();
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  };

  const running = state && (state.status === 'running' || state.status === 'paused');
  const r = state?.result;
  return (
    <Card title="Statistical robustness" actions={
      <button className="primary" onClick={start} disabled={!combinationRunId || busy || running}>Run robustness tests</button>}>
      {error && <div className="error-box">{error}</div>}
      {!combinationRunId && <div className="empty">Needs a finished combination run.</div>}
      {state && <div className="run-line"><Badge status={state.status || 'none'} /><span className="muted">run #{state.run_id ?? '—'}</span>{state.error && <span className="error-box">{state.error}</span>}</div>}
      {!r && combinationRunId && !running && <div className="muted">No robustness result for this run yet.</div>}
      {r && (
        <>
          {r.dataset?.is_synthetic && <div className="warn-box">{r.dataset.label}</div>}
          <div className="kv-grid">
            <div><b>Span</b> {r.window?.start?.slice(0, 10)} → {r.window?.end?.slice(0, 10)} ({r.window?.span})</div>
            <div><b>Candidates</b> {r.candidates?.length ?? 0}</div>
          </div>
          <div className="grid two">
            <Test title="Hansen SPA" p={r.spa?.p_value} note="Does any candidate beat buy-and-hold on these bars? Small p-value = evidence of an edge.">
              <div>Statistic {pv(r.spa?.statistic)} · draws {r.spa?.bootstrap_draws ?? '—'}</div>
            </Test>
            <Test title="Permutation test" p={r.permutation?.p_value} note="Shifted signals with the same count: is the real timing better than chance?">
              <div>Observed net {money(r.permutation?.observed)} · draws {r.permutation?.draws ?? '—'}</div>
            </Test>
            <Test title="Monte Carlo (trade bootstrap)" note="Resampled trade order and luck. Shows how much the result depends on sequence.">
              {r.monte_carlo?.status ? <div>{r.monte_carlo.reason}</div> : (
                <div>
                  Probability of loss {pv(r.monte_carlo?.probability_of_loss)} · net p5 {money(r.monte_carlo?.net_profit?.p5)} / p50 {money(r.monte_carlo?.net_profit?.p50)} / p95 {money(r.monte_carlo?.net_profit?.p95)}
                  <br />Max drawdown p95 {pv(r.monte_carlo?.max_drawdown?.p95)}% of capital
                </div>)}
            </Test>
            <Test title="PBO (CSCV)" note="Probability that the best in-sample candidate ranks below median out of sample.">
              <KeyValues obj={r.pbo} />
            </Test>
            <Test title="Deflated Sharpe" note="Sharpe corrected for how many candidates were tried.">
              <KeyValues obj={r.dsr} />
            </Test>
          </div>
          <small className="muted">Generated {r.generated_at}. Train span only. Results are reported as calculated; no threshold is applied for you.</small>
        </>
      )}
    </Card>
  );
}

function Test({ title, p, note, children }) {
  return (
    <div className="test-box">
      <div className="test-head"><b>{title}</b>{p !== undefined && p !== null && <span className="pv">p = {pv(p)}</span>}</div>
      <div className="muted small">{note}</div>
      <div className="test-body">{children}</div>
    </div>
  );
}

function KeyValues({ obj }) {
  if (!obj) return <div className="muted">Not available.</div>;
  const entries = Object.entries(obj).filter(([, v]) => v === null || typeof v !== 'object');
  if (!entries.length) return <div className="muted">Not available.</div>;
  return <div className="kv-grid compact">{entries.map(([k, v]) => <div key={k}><b>{k.replace(/_/g, ' ')}</b> {typeof v === 'number' ? Number(v).toFixed(4) : String(v ?? '—')}</div>)}</div>;
}
