import { useEffect, useMemo, useState } from 'react';
import { get, post } from '../api.js';
import { Card, Badge, ErrorBox, useAsync, Stat } from '../components/ui.jsx';

export default function Indicators() {
  const [data, setData] = useState(null);
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('all');
  const { error, busy, run } = useAsync();

  const load = () => run(async () => setData(await get('/api/indicators')));
  useEffect(() => { load(); }, []);

  const reload = () => run(async () => {
    await post('/api/indicators/reload');
    setData(await get('/api/indicators'));
  });

  const filtered = useMemo(() => {
    if (!data) return [];
    const q = query.trim().toLowerCase();
    return data.indicators.filter((i) =>
      (category === 'all' || i.category === category) &&
      (!q || i.name.includes(q) || i.description.toLowerCase().includes(q)));
  }, [data, query, category]);

  const categories = data ? Object.keys(data.summary.categories).sort() : [];
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Indicators</h1>
          <div className="muted">Discovered at runtime from the built-in package and the plugin folder. Nothing here is hardcoded.</div>
        </div>
        <button className="primary" disabled={busy} onClick={reload}>Reload registry</button>
      </div>
      <ErrorBox error={error} />
      {data && (
        <div className="grid stats" style={{ marginBottom: 14 }}>
          <Stat label="Registered" value={data.summary.total} />
          <Stat label="Available" value={data.summary.available} tone="good" hint="passed the self-test" />
          <Stat label="Unavailable" value={data.summary.unavailable} tone={data.summary.unavailable ? 'bad' : ''} />
          <Stat label="Categories" value={categories.length} />
        </div>
      )}
      <Card>
        <div className="form-grid" style={{ marginBottom: 12 }}>
          <label className="field"><span>Search</span>
            <input placeholder="name or description" value={query} onChange={(e) => setQuery(e.target.value)} />
          </label>
          <label className="field"><span>Category</span>
            <select value={category} onChange={(e) => setCategory(e.target.value)}>
              <option value="all">All categories</option>
              {categories.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
          </label>
        </div>
        {filtered.length === 0 && <div className="empty">No indicators match.</div>}
        <div className="grid two">
          {filtered.map((i) => (
            <div key={i.name} className={`ind-card ${i.available ? '' : 'unavailable'}`}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                <strong className="mono">{i.name}</strong>
                <Badge status={i.available ? 'ok' : 'unavailable'} />
              </div>
              <div className="muted">{i.category}{i.description ? ` · ${i.description}` : ''}</div>
              {!i.available && <div className="error-box">{i.status_reason}</div>}
              <table>
                <thead><tr><th>Parameter</th><th className="right">Default</th><th className="right">Range</th></tr></thead>
                <tbody>
                  {i.params.map((p) => (
                    <tr key={p.name}><td className="mono">{p.name}</td><td className="right">{p.default}</td>
                      <td className="right">{p.low}–{p.high}</td></tr>
                  ))}
                  {i.params.length === 0 && <tr><td colSpan="3" className="muted">No parameters</td></tr>}
                </tbody>
              </table>
              <small className="mono">{i.module}</small>
            </div>
          ))}
        </div>
      </Card>
      {data && data.summary.load_errors.length > 0 && (
        <Card title="Load errors">
          {data.summary.load_errors.map((e, idx) => (
            <div key={idx} className="error-box"><strong>{e.source}</strong>: {e.error}</div>
          ))}
        </Card>
      )}
      <Card title="Adding an indicator">
        <p className="muted" style={{ margin: 0 }}>
          Create a Python file in the plugin folder that decorates a causal function with <code>@indicator(...)</code>,
          return a signal in {'{-1, 0, 1}'}, then press <em>Reload registry</em>. The agents pick it up without code changes.
        </p>
      </Card>
    </>
  );
}
