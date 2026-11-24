import { useEffect, useState } from 'react';
import { get, put } from '../api.js';
import { Card, Field, ErrorBox } from './ui.jsx';

// Configurable signal rules for one indicator. Continuous outputs can be turned into long/short votes
// with thresholds, crossovers and state changes, combined with AND or OR. Legacy mode keeps -1/0/+1 as is.
const OPS = [
  ['gt', '>  greater than'], ['ge', '≥  at least'], ['lt', '<  less than'], ['le', '≤  at most'],
  ['cross_above', 'crosses above'], ['cross_below', 'crosses below'], ['equals_change', 'changes to state'],
];
const EMPTY_BLOCK = { logic: 'AND', conditions: [] };
const EMPTY_CONDITION = { op: 'gt', value: 0, other: '' };

export default function SignalRules({ onSaved }) {
  const [list, setList] = useState([]);
  const [key, setKey] = useState('');
  const [mode, setMode] = useState('legacy');
  const [long, setLong] = useState(EMPTY_BLOCK);
  const [short, setShort] = useState(EMPTY_BLOCK);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);

  const load = async () => {
    try {
      const r = await get('/api/research/signals');
      setList(r.indicators);
      setKey((k) => k || (r.indicators[0]?.source_key ?? ''));
    } catch (e) { setError(e.message); }
  };
  useEffect(() => { load(); }, []);

  const current = list.find((i) => i.source_key === key);
  useEffect(() => {
    const spec = current?.spec;
    if (!spec || spec.mode !== 'conditions') { setMode('legacy'); setLong(EMPTY_BLOCK); setShort(EMPTY_BLOCK); return; }
    setMode('conditions'); setLong(spec.long || EMPTY_BLOCK); setShort(spec.short || EMPTY_BLOCK);
  }, [key, current?.spec]);

  const others = list.filter((i) => i.source_key !== key && i.available).map((i) => i.name);

  const save = async (spec) => {
    setError(null); setNotice(null);
    try {
      await put(`/api/research/signals/${encodeURIComponent(key)}`, { spec });
      setNotice(spec ? 'Saved. Future runs use these rules; earlier results for this indicator are not reused.' : 'Reset to the indicator\'s own -1/0/+1 votes.');
      await load();
      onSaved?.();
    } catch (e) { setError(e.message); }
  };

  const block = (name, value, setValue) => (
    <div className="rule-block">
      <div className="rule-head">
        <b>{name === 'long' ? 'Long when' : 'Short when'}</b>
        <select value={value.logic} onChange={(e) => setValue({ ...value, logic: e.target.value })}>
          <option value="AND">all conditions (AND)</option><option value="OR">any condition (OR)</option>
        </select>
        <button className="ghost" onClick={() => setValue({ ...value, conditions: [...value.conditions, { ...EMPTY_CONDITION }] })}>+ condition</button>
      </div>
      {value.conditions.length === 0 && <div className="muted small">No conditions.</div>}
      {value.conditions.map((c, i) => (
        <div className="rule-row" key={i}>
          <select value={c.op} onChange={(e) => setValue({ ...value, conditions: value.conditions.map((x, j) => (j === i ? { ...x, op: e.target.value, other: e.target.value === 'equals_change' ? '' : x.other } : x)) })}>
            {OPS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
          </select>
          {c.op !== 'equals_change' && (
            <select value={c.other || ''} onChange={(e) => setValue({ ...value, conditions: value.conditions.map((x, j) => (j === i ? { ...x, other: e.target.value } : x)) })}>
              <option value="">a constant</option>
              {others.map((n) => <option key={n} value={n}>indicator {n}</option>)}
            </select>
          )}
          {(!c.other || c.op === 'equals_change') && (
            <input type="number" step="any" value={c.value ?? ''} onChange={(e) => setValue({ ...value, conditions: value.conditions.map((x, j) => (j === i ? { ...x, value: e.target.value === '' ? null : Number(e.target.value) } : x)) })} />
          )}
          <button className="ghost" onClick={() => setValue({ ...value, conditions: value.conditions.filter((_, j) => j !== i) })}>Remove</button>
        </div>
      ))}
    </div>
  );

  const buildSpec = () => (mode === 'legacy' ? null : {
    mode: 'conditions',
    long: { logic: long.logic, conditions: long.conditions.map(clean) },
    short: { logic: short.logic, conditions: short.conditions.map(clean) },
  });
  const clean = (c) => ({ op: c.op, value: c.other ? null : c.value, other: c.other || null });

  if (!list.length) return <Card title="Signal rules"><div className="empty">No indicators discovered. Register one in Indicator Studio.</div></Card>;
  return (
    <Card title="Signal rules (continuous outputs)">
      <ErrorBox error={error} />
      {notice && <div className="info-box">{notice}</div>}
      <div className="form-grid">
        <Field label="Indicator">
          <select value={key} onChange={(e) => setKey(e.target.value)}>
            {list.map((i) => <option key={i.source_key} value={i.source_key}>{i.name}{i.available ? '' : ' (unavailable)'}</option>)}
          </select>
        </Field>
        <Field label="Mode">
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            <option value="legacy">Own -1 / 0 / +1 votes</option><option value="conditions">Conditions on raw values</option>
          </select>
        </Field>
      </div>
      {current && !current.available && <div className="warn-box">{current.status_reason}</div>}
      {mode === 'conditions' && (
        <>
          {block('long', long, setLong)}
          {block('short', short, setShort)}
          <small className="muted">A bar votes long when only the long block holds, short when only the short block holds, and neutral when both or neither hold. Conditions read only the current and previous bar.</small>
        </>
      )}
      <div className="card-actions" style={{ marginTop: 10 }}>
        <button className="primary" disabled={!current?.available} onClick={() => save(buildSpec())}>Save rules</button>
      </div>
    </Card>
  );
}
