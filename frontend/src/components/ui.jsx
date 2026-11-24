import { useState } from 'react';

export function Card({ title, actions, children, className = '' }) {
  return (
    <section className={`card ${className}`}>
      {(title || actions) && (
        <header className="card-head">
          {title && <h3>{title}</h3>}
          {actions && <div className="card-actions">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint, tone }) {
  return (
    <div className={`stat ${tone || ''}`}>
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {hint && <div className="stat-hint">{hint}</div>}
    </div>
  );
}

export function Badge({ status }) {
  const s = String(status || 'unknown').toLowerCase();
  const tone = ['running', 'complete', 'completed', 'ok', 'validated'].includes(s) ? 'good'
    : ['paused', 'pending', 'waiting', 'attention', 'pruned'].includes(s) ? 'warn'
    : ['error', 'failed', 'fail', 'unavailable', 'interrupted'].includes(s) ? 'bad' : 'muted';
  return <span className={`badge ${tone}`}>{status}</span>;
}

export function Table({ columns, rows, onRowClick, rowKey, empty = 'No data yet.', selectedKey }) {
  if (!rows || rows.length === 0) return <div className="empty">{empty}</div>;
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>{columns.map((c) => <th key={c.key} className={c.align || ''}>{c.label}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, i) => {
            const key = rowKey ? rowKey(row) : i;
            return (
              <tr key={key} className={`${onRowClick ? 'clickable' : ''} ${selectedKey === key ? 'selected' : ''}`}
                onClick={onRowClick ? () => onRowClick(row) : undefined}>
                {columns.map((c) => (
                  <td key={c.key} className={c.align || ''}>{c.render ? c.render(row) : row[c.key]}</td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function Field({ label, children, hint }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint && <small>{hint}</small>}
    </label>
  );
}

export function ErrorBox({ error }) {
  if (!error) return null;
  return <div className="error-box">{String(error)}</div>;
}

export function useAsync() {
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const run = async (fn) => {
    setBusy(true);
    setError(null);
    try {
      return await fn();
    } catch (e) {
      setError(e.message || String(e));
      return undefined;
    } finally {
      setBusy(false);
    }
  };
  return { error, busy, run, setError };
}
