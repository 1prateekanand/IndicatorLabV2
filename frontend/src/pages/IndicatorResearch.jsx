import { useEffect, useState } from 'react';
import { get, post } from '../api.js';
import { useApp } from '../context.js';
import { Card, Stat, Badge, ErrorBox, Field } from '../components/ui.jsx';
import IndicatorTable from '../components/IndicatorTable.jsx';
import RulesForm, { DEFAULT_RULES, rulesPayload } from '../components/RulesForm.jsx';
import { useRun } from '../components/useRun.js';
import SignalRules from '../components/SignalRules.jsx';
import SyntheticData from '../components/SyntheticData.jsx';

const TIMEFRAMES = ['1min', '5min', '10min', '1h', '4h'];
const isActive = (r) => !!r && (r.status === 'running' || r.status === 'paused');
const DEFAULT_COMBO = { segments: 4, pool_size: 12, corr_threshold: 0.8, purge_bars: 60, max_evaluations: 4000, top_n: 10 };

export default function IndicatorResearch() {
  const { datasetId, setDatasetId, reloadDatasets } = useApp();
  const [rules, setRules] = useState(DEFAULT_RULES);
  const [timeframe, setTimeframe] = useState('1min');
  const [combo, setCombo] = useState(DEFAULT_COMBO);
  const [indRunId, setIndRunId] = useState(null);
  const [table, setTable] = useState(null);
  const [comboRunId, setComboRunId] = useState(null);
  const [combo_, setCombo_] = useState(null);      // latest combination payload
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const indRun = useRun(indRunId);
  const comboStatus = useRun(comboRunId);

  // Re-attach to a run that is still active (for example after a page reload), so controls stay correct.
  const loadTable = async () => {
    if (!datasetId) return;
    try {
      const r = await get(`/api/research/individual?dataset_id=${datasetId}&split=train`);
      setTable(r);
      if (isActive(r.run)) setIndRunId(r.run.id);
    } catch (e) { setError(e.message); }
  };
  const loadCombo = async () => {
    if (!datasetId) return;
    try {
      const r = await get(`/api/research/combination?dataset_id=${datasetId}`);
      setCombo_(r);
      if (isActive(r.run)) setComboRunId(r.run.id);
    } catch (e) { setError(e.message); }
  };

  // On dataset change: show the latest stored results (no run is started).
  useEffect(() => { setIndRunId(null); setComboRunId(null); loadTable(); loadCombo(); /* eslint-disable-next-line */ }, [datasetId]);
  // Refresh the table when a run ends
  useEffect(() => { if (indRun && indRun.status !== 'running' && indRun.status !== 'paused') loadTable(); /* eslint-disable-next-line */ }, [indRun?.status]);
  useEffect(() => { if (comboStatus && comboStatus.status !== 'running' && comboStatus.status !== 'paused') loadCombo(); /* eslint-disable-next-line */ }, [comboStatus?.status]);

  const anyActive = [indRun, comboStatus].some(isActive);

  const startIndividual = async () => {
    setBusy(true); setError(null);
    try {
      const r = await post('/api/research/individual/start', { dataset_id: datasetId, split: 'train', timeframe, rules: rulesPayload(rules) });
      setIndRunId(r.id);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  };
  const startCombination = async () => {
    setBusy(true); setError(null);
    try {
      const r = await post('/api/research/combination/start', {
        dataset_id: datasetId, timeframe, rules: rulesPayload(rules),
        segments: Number(combo.segments), pool_size: Number(combo.pool_size), corr_threshold: Number(combo.corr_threshold),
        purge_bars: Number(combo.purge_bars), max_evaluations: Number(combo.max_evaluations), top_n: Number(combo.top_n),
      });
      setComboRunId(r.id);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  };
  const control = async (runId, action) => {
    setError(null);
    try { await post(`/api/research/runs/${runId}/${action}`, {}); } catch (e) { setError(e.message); }
  };

  if (!datasetId) {
    return <div className="page"><h1>Indicator</h1><div className="empty">Import or select a dataset in the top bar.</div></div>;
  }

  const counts = table?.counts || {};
  const rows = table?.rows || [];
  const latestComboRun = comboStatus || combo_?.run;

  return (
    <div className="page indicator-research">
      <div className="page-head">
        <div>
          <h1>Indicator</h1>
          <div className="muted">Every indicator registered in Indicator Studio is discovered automatically. Nothing is hardcoded.</div>
        </div>
      </div>
      <ErrorBox error={error} />

      <Card title="Backtest rules" actions={<span className="muted">Used by individual and combination runs</span>}>
        <div className="form-grid">
          <Field label="Timeframe">
            <select value={timeframe} onChange={(e) => setTimeframe(e.target.value)} disabled={anyActive}>
              {TIMEFRAMES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
          </Field>
        </div>
        <RulesForm value={rules} onChange={setRules} disabled={anyActive} />
        <small className="muted">Costs and capital are placeholders until you set your broker schedule. Entries fill at the next bar's open.</small>
      </Card>

      <div className="grid stats">
        <Stat label="Discovered" value={rows.length} hint="available and unavailable" />
        <Stat label="Evaluated" value={counts.evaluated ?? 0} tone="good" />
        <Stat label="Not evaluated" value={counts.not_evaluated ?? 0} hint="registered after the last run" />
        <Stat label="Changed since run" value={counts.changed_since_run ?? 0} hint="edited; re-run to evaluate" />
        <Stat label="Unavailable" value={counts.unavailable ?? 0} hint="see row details" />
      </div>

      <div className="grid two">
        <Card title="Individual evaluation" actions={
          <button className="primary" onClick={startIndividual} disabled={busy || anyActive}>Evaluate all indicators</button>}>
          <p className="muted">Trades, wins, win rate, Beta-Binomial score, Wilson lower bound and costs on the train split.</p>
          <RunBar run={indRun} onControl={control} />
        </Card>
        <Card title="Combination research" actions={
          <button className="primary" onClick={startCombination} disabled={busy || anyActive}>Start combination search</button>}>
          <div className="form-grid">
            <Field label="Walk-forward segments"><input type="number" min="2" max="10" value={combo.segments} onChange={(e) => setCombo({ ...combo, segments: e.target.value })} /></Field>
            <Field label="Candidate pool"><input type="number" min="3" max="25" value={combo.pool_size} onChange={(e) => setCombo({ ...combo, pool_size: e.target.value })} /></Field>
            <Field label="Spearman cut-off"><input type="number" step="0.05" min="0.5" max="0.99" value={combo.corr_threshold} onChange={(e) => setCombo({ ...combo, corr_threshold: e.target.value })} /></Field>
            <Field label="Purge (bars)"><input type="number" min="0" value={combo.purge_bars} onChange={(e) => setCombo({ ...combo, purge_bars: e.target.value })} /></Field>
            <Field label="Evaluation budget"><input type="number" min="100" value={combo.max_evaluations} onChange={(e) => setCombo({ ...combo, max_evaluations: e.target.value })} /></Field>
            <Field label="Top N"><input type="number" min="1" max="50" value={combo.top_n} onChange={(e) => setCombo({ ...combo, top_n: e.target.value })} /></Field>
          </div>
          <RunBar run={comboStatus} onControl={control} />
          {latestComboRun && !comboStatus && <div className="muted">Latest run #{latestComboRun.id}: {latestComboRun.status}</div>}
        </Card>
      </div>

      {combo_?.run && <CoverageCard payload={combo_} />}

      <div className="grid two">
        <SignalRules onSaved={loadTable} />
        <SyntheticData onCreated={async (made) => { await reloadDatasets?.(); setDatasetId?.(made.id); }} />
      </div>

      <Card title="Individual results" actions={<span className="muted">{table?.run ? `run #${table.run.id} · ${table.run.status}` : 'no run yet'}</span>}>
        {rows.length ? <IndicatorTable rows={rows} csvName="individual_indicators.csv" />
          : <div className="empty">No indicators discovered yet. Register them in Indicator Studio.</div>}
      </Card>
    </div>
  );
}

function RunBar({ run, onControl }) {
  if (!run) return <div className="muted">No active run.</div>;
  const active = isActive(run);
  return (
    <div className="run-bar">
      <Badge status={run.status} />
      <span className="muted">{run.message || ''}</span>
      <div className="progress"><div style={{ width: `${Math.round((run.progress || 0) * 100)}%` }} /></div>
      {active && (
        <div className="card-actions">
          {run.status === 'paused'
            ? <button onClick={() => onControl(run.id, 'resume')}>Resume</button>
            : <button onClick={() => onControl(run.id, 'pause')}>Pause</button>}
          <button className="ghost" onClick={() => onControl(run.id, 'stop')}>Stop</button>
        </div>
      )}
      {run.error && <div className="error-box">{run.error}</div>}
    </div>
  );
}

function CoverageCard({ payload }) {
  const sel = payload.selection;
  if (!sel) return null;
  const c = sel.coverage;
  return (
    <Card title="Combination coverage" actions={<span className="muted">run #{payload.run.id}</span>}>
      <div className="kv-grid">
        <div><b>Indicators available</b> {c.indicators_available}</div>
        <div><b>Eligible (min trades)</b> {c.eligible}</div>
        <div><b>Dropped by Spearman filter</b> {c.dropped_by_correlation}</div>
        <div><b>Candidate pool</b> {c.pool_size}</div>
        <div><b>Pairs</b> {c.pairs_evaluated} / {c.pairs_possible}</div>
        <div><b>Triples</b> {c.triples_evaluated} / {c.triples_possible}</div>
        <div><b>SFFS evaluations</b> {c.sffs_evaluations} (max size {c.max_size})</div>
        <div><b>Ranked combinations</b> {c.combinations_ranked}</div>
        <div><b>Budget reached</b> {c.budget_reached ? 'yes — results are partial' : 'no'}</div>
      </div>
      {sel.dropped?.length > 0 && (
        <details><summary>Dropped indicators ({sel.dropped.length})</summary>
          <ul>{sel.dropped.map((d) => <li key={d.key}>{d.name} — rho {d.rho} with {d.kept_partner}</li>)}</ul>
        </details>
      )}
    </Card>
  );
}
