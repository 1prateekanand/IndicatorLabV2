import { Fragment, useMemo, useState } from 'react';
import { Badge } from './ui.jsx';

// Columns of an individual evaluation row. ``num`` columns sort numerically; nulls always sort last.
const COLUMNS = [
  { key: 'name', label: 'Indicator' },
  { key: 'source', label: 'Source' },
  { key: 'category', label: 'Category' },
  { key: 'status', label: 'Status' },
  { key: 'trades', label: 'Trades', num: true },
  { key: 'wins', label: 'Wins', num: true },
  { key: 'losses', label: 'Losses', num: true },
  { key: 'raw_win_rate', label: 'Win %', num: true, fmt: (v) => fmt(v, 2) },
  { key: 'wilson_lb', label: 'Wilson LB', num: true, fmt: (v) => fmt(v, 4) },
  { key: 'shrunk_score', label: 'Shrunk score', num: true, fmt: (v) => fmt(v, 4) },
  { key: 'net_profit', label: 'Net profit', num: true, fmt: (v) => fmt(v, 2), signed: true },
  { key: 'profit_factor', label: 'Profit factor', num: true, fmt: (v) => fmt(v, 2) },
  { key: 'max_drawdown_pct', label: 'Max DD %', num: true, fmt: (v) => fmt(v, 2) },
  { key: 'total_costs', label: 'Costs', num: true, fmt: (v) => fmt(v, 2) },
];

const fmt = (v, d) => (v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d));

function cmp(a, b, key, dir) {
  const x = a[key], y = b[key];
  const xn = x === null || x === undefined, yn = y === null || y === undefined;
  if (xn || yn) return xn === yn ? 0 : xn ? 1 : -1;  // nulls last in both directions
  if (typeof x === 'number' && typeof y === 'number') return (x - y) * dir;
  return String(x).localeCompare(String(y)) * dir;
}

export function toCsv(rows) {
  const head = ['name', 'source', 'category', 'status', 'trades', 'wins', 'losses', 'raw_win_rate', 'wilson_lb',
    'shrunk_score', 'net_profit', 'net_return_pct', 'profit_factor', 'max_drawdown_pct', 'total_costs', 'reason'];
  const esc = (v) => (v === null || v === undefined ? '' : `"${String(v).replace(/"/g, '""')}"`);
  return [head.join(','), ...rows.map((r) => head.map((h) => esc(r[h])).join(','))].join('\n');
}

export default function IndicatorTable({ rows, csvName = 'indicators.csv' }) {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('all');
  const [status, setStatus] = useState('all');
  const [sort, setSort] = useState({ key: 'wilson_lb', dir: -1 });
  const [open, setOpen] = useState(null);

  const categories = useMemo(() => ['all', ...Array.from(new Set(rows.map((r) => r.category).filter(Boolean))).sort()], [rows]);
  const statuses = useMemo(() => ['all', ...Array.from(new Set(rows.map((r) => r.status))).sort()], [rows]);

  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    const filtered = rows.filter((r) =>
      (category === 'all' || r.category === category) &&
      (status === 'all' || r.status === status) &&
      (!q || String(r.name).toLowerCase().includes(q) || String(r.category || '').toLowerCase().includes(q)));
    return [...filtered].sort((a, b) => cmp(a, b, sort.key, sort.dir));
  }, [rows, query, category, status, sort]);

  const clickSort = (key) => setSort((s) => (s.key === key ? { key, dir: -s.dir } : { key, dir: -1 }));

  const download = () => {
    const blob = new Blob([toCsv(shown)], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = csvName;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="ind-table">
      <div className="ind-toolbar">
        <input placeholder="Search name or category" value={query} onChange={(e) => setQuery(e.target.value)} />
        <select value={category} onChange={(e) => setCategory(e.target.value)} aria-label="Category">
          {categories.map((c) => <option key={c} value={c}>{c === 'all' ? 'All categories' : c}</option>)}
        </select>
        <select value={status} onChange={(e) => setStatus(e.target.value)} aria-label="Status">
          {statuses.map((s) => <option key={s} value={s}>{s === 'all' ? 'All statuses' : s}</option>)}
        </select>
        <span className="muted">{shown.length} of {rows.length} shown</span>
        <button className="ghost" onClick={download} disabled={!shown.length}>Export CSV</button>
      </div>
      <div className="table-wrap">
        <table className="sortable">
          <thead>
            <tr>
              {COLUMNS.map((c) => (
                <th key={c.key} className={c.num ? 'num' : ''}>
                  <button className="th-sort" onClick={() => clickSort(c.key)}>
                    {c.label}{sort.key === c.key ? (sort.dir > 0 ? ' ▲' : ' ▼') : ''}
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {shown.length === 0 && <tr><td colSpan={COLUMNS.length} className="empty">No indicators match.</td></tr>}
            {shown.map((r) => (
              <Fragment key={r.source_key}>
                <tr className="clickable" onClick={() => setOpen(open === r.source_key ? null : r.source_key)}>
                  {COLUMNS.map((c) => {
                    const v = r[c.key];
                    if (c.key === 'status') return <td key={c.key}><Badge status={v} /></td>;
                    const text = c.fmt ? c.fmt(v) : (v ?? '—');
                    const tone = c.signed && typeof v === 'number' ? (v >= 0 ? 'pos' : 'neg') : '';
                    return <td key={c.key} className={`${c.num ? 'num' : ''} ${tone}`}>{text}</td>;
                  })}
                </tr>
                {open === r.source_key && (
                  <tr className="detail-row">
                    <td colSpan={COLUMNS.length}><RowDetail row={r} /></td>
                  </tr>
                )}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RowDetail({ row }) {
  if (row.status !== 'evaluated') {
    return (
      <div className="kv-grid">
        <div><b>Status</b> {row.status}</div>
        <div><b>Reason</b> {row.reason || '—'}</div>
      </div>
    );
  }
  const reasons = row.exit_reasons || {};
  return (
    <div className="kv-grid">
      <div><b>Longs / shorts</b> {row.longs} / {row.shorts}</div>
      <div><b>Net return</b> {fmt(row.net_return_pct, 2)}%</div>
      <div><b>Average trade</b> {fmt(row.avg_trade, 2)}</div>
      <div><b>Skipped entries</b> {row.skipped_entries}</div>
      <div><b>Bars evaluated</b> {row.bars_evaluated}</div>
      <div><b>Evaluated range</b> {row.eval_start} → {row.eval_end}</div>
      <div><b>Exit reasons</b> {Object.keys(reasons).length ? Object.entries(reasons).map(([k, v]) => `${k}: ${v}`).join(', ') : '—'}</div>
      <div><b>Scored</b> {row.eligible_for_score ? 'yes' : 'no (below minimum trades)'}</div>
    </div>
  );
}
