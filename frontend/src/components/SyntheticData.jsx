import { useEffect, useState } from 'react';
import { get, post } from '../api.js';
import { Card, Field, ErrorBox } from './ui.jsx';

// Controlled test markets. Every dataset created here is flagged synthetic and labelled everywhere it is shown.
export default function SyntheticData({ onCreated }) {
  const [scenarios, setScenarios] = useState([]);
  const [scenario, setScenario] = useState('');
  const [days, setDays] = useState(60);
  const [seed, setSeed] = useState(1);
  const [error, setError] = useState(null);
  const [made, setMade] = useState(null);

  useEffect(() => {
    get('/api/research/synthetic/scenarios').then((r) => { setScenarios(r.scenarios); setScenario(r.scenarios[0]?.scenario || ''); }).catch((e) => setError(e.message));
  }, []);

  const create = async () => {
    setError(null); setMade(null);
    try {
      const r = await post('/api/research/synthetic/create', { scenario, days: Number(days), seed: Number(seed) });
      setMade(r);
      onCreated?.(r);
    } catch (e) { setError(e.message); }
  };

  const desc = scenarios.find((s) => s.scenario === scenario)?.description;
  return (
    <Card title="Synthetic test markets">
      <div className="warn-box">Synthetic data is for controlled testing only. Its results are never market performance.</div>
      <ErrorBox error={error} />
      <div className="form-grid">
        <Field label="Scenario">
          <select value={scenario} onChange={(e) => setScenario(e.target.value)}>
            {scenarios.map((s) => <option key={s.scenario} value={s.scenario}>{s.scenario.replace('_', ' ')}</option>)}
          </select>
        </Field>
        <Field label="Trading days (5–365)"><input type="number" min="5" max="365" value={days} onChange={(e) => setDays(e.target.value)} /></Field>
        <Field label="Seed"><input type="number" min="0" value={seed} onChange={(e) => setSeed(e.target.value)} /></Field>
      </div>
      {desc && <div className="muted small">{desc}</div>}
      <div className="card-actions" style={{ marginTop: 10 }}><button className="primary" onClick={create} disabled={!scenario}>Create synthetic dataset</button></div>
      {made && <div className="info-box">Created <b>{made.name}</b> (synthetic). Select it in the top bar to test on it.</div>}
    </Card>
  );
}
