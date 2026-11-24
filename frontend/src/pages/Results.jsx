import { useEffect, useMemo, useState } from 'react';
import { get, fmtMoney, fmtPct, fmtNum, fmtTime, specText } from '../api.js';
import { useApp, saveStrategy } from '../context.js';
import { Card, Badge, Table, ErrorBox, Field, useAsync } from '../components/ui.jsx';
import { useNavigate } from 'react-router-dom';

const TABS = [
  ['profit', 'Highest net profit'],
  ['sharpe_ratio', 'Highest Sharpe ratio'],
  ['win_rate', 'Highest win rate'],
  ['profit_factor', 'Profit factor'],
  ['drawdown', 'Lowest drawdown'],
  ['risk_adjusted', 'Risk-adjusted'],
  ['walk_forward', 'Walk-forward validated'],
];

function formatParams(spec) {
  const indParams = (spec?.indicators || []).map((i) => {
    const ps = Object.entries(i.params || {}).map(([k, v]) => `${k}=${v}`).join(', ');
    return ps ? `${i.name}(${ps})` : i.name;
  });
  const stratParams = spec?.strategy_params
    ? Object.values(spec.strategy_params).flatMap((p) => Object.entries(p || {}).map(([k, v]) => `${k}=${v}`))
    : [];
  const risk = `SL ${spec?.stop_loss_pct ?? 0}% · TP ${spec?.target_pct ?? 0}% · ${spec?.direction || 'both'}`;
  return [...indParams, ...stratParams, risk].filter(Boolean).join(' | ');
}

export default function Results() {
  const { datasetId } = useApp();
  const navigate = useNavigate();
  const [tab, setTab] = useState('profit');
  const [top, setTop] = useState(50);
  const [split, setSplit] = useState('train');
  const [rows, setRows] = useState([]);
  const [detail, setDetail] = useState(null);
  const [compareIds, setCompareIds] = useState([]);
  const [search, setSearch] = useState('');
  const [robustFilter, setRobustFilter] = useState('all');
  const [sortCol, setSortCol] = useState(null);
  const [sortDir, setSortDir] = useState('desc');
  const { error, run } = useAsync();

  useEffect(() => {
    if (!datasetId) return;
    run(async () => setRows(await get(`/api/results/rankings/${tab}?dataset_id=${datasetId}&top=${top}&split=${split}`)));
  }, [datasetId, tab, top, split]);

  const open = (row) => run(async () => setDetail(await get(`/api/results/strategies/${row.strategy_hash}?dataset_id=${datasetId}`)));
  const toBacktest = () => {
    if (!detail) return;
    saveStrategy(detail.spec);
    navigate('/backtest');
  };

  const toggleCompare = (row, e) => {
    e.stopPropagation();
    setCompareIds((prev) =>
      prev.includes(row.strategy_hash)
        ? prev.filter((h) => h !== row.strategy_hash)
        : [...prev.slice(-2), row.strategy_hash]
    );
  };

  const toggleSort = (col) => {
    if (sortCol === col) setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
    else { setSortCol(col); setSortDir('desc'); }
  };

  const filteredAndSorted = useMemo(() => {
    const q = search.trim().toLowerCase();
    let list = rows.filter((r) => {
      if (robustFilter === 'validated' && !r.validated) return false;
      if (robustFilter === 'oos' && !r.out_of_sample) return false;
      if (robustFilter === 'dsr' && !r.dsr?.significant) return false;
      if (!q) return true;
      const hay = `${r.spec?.strategy_name || ''} ${(r.spec?.indicators || []).map((i) => i.name).join(' ')} ${r.strategy_hash}`.toLowerCase();
      return hay.includes(q);
    });
    if (sortCol) {
      list = [...list].sort((a, b) => {
        const valOf = (r) => {
          if (sortCol === 'net_profit') return r.metrics?.net_profit ?? -Infinity;
          if (sortCol === 'win_rate') return r.metrics?.win_rate ?? -Infinity;
          if (sortCol === 'profit_factor') return r.metrics?.profit_factor ?? -Infinity;
          if (sortCol === 'sharpe_ratio') return r.metrics?.sharpe_ratio ?? -Infinity;
          if (sortCol === 'max_drawdown_pct') return r.metrics?.max_drawdown_pct ?? Infinity;
          if (sortCol === 'trades') return r.metrics?.trades ?? 0;
          if (sortCol === 'oos') return r.out_of_sample?.metrics?.net_profit ?? -Infinity;
          return r.rank;
        };
        const av = valOf(a);
        const bv = valOf(b);
        return sortDir === 'desc' ? (bv - av) : (av - bv);
      });
    }
    return list;
  }, [rows, search, robustFilter, sortCol, sortDir]);

  const compareRows = useMemo(
    () => rows.filter((r) => compareIds.includes(r.strategy_hash)),
    [rows, compareIds]
  );

  const renderRobustness = (r) => (
    <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
      {r.validated ? <Badge status="validated" /> : r.validation ? <Badge status="failed" /> : <span className="muted">WF: —</span>}
      {r.dsr && (
        <span className={`badge ${r.dsr.significant ? 'good' : 'warn'}`}>
          DSR {Number(r.dsr.dsr).toFixed(2)}
        </span>
      )}
    </div>
  );

  const renderOos = (r) => {
    if (!r.out_of_sample) return <span className="muted">Not run</span>;
    const om = r.out_of_sample.metrics;
    return (
      <span style={{ color: om.net_profit >= 0 ? 'var(--good)' : 'var(--bad)' }}>
        {fmtMoney(om.net_profit)} ({om.trades}t)
      </span>
    );
  };

  const best = detail && detail.experiments.find((e) => e.split === 'train');

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Results</h1>
          <div className="muted">
            Discovered indicator &amp; strategy combinations ranked from stored experiments. Click column headers to sort, select checkboxes to compare, or click a row for full details.
          </div>
        </div>
      </div>
      <ErrorBox error={error} />
      <Card>
        <div className="tabs">
          {TABS.map(([k, label]) => (
            <button key={k} className={`tab ${tab === k ? 'active' : ''}`} onClick={() => { setSortCol(null); setTab(k); }}>
              {label}
            </button>
          ))}
        </div>
        <div className="form-grid" style={{ marginBottom: 12 }}>
          <Field label="Search strategy / indicator">
            <input placeholder="Filter by name, indicator, or hash..." value={search} onChange={(e) => setSearch(e.target.value)} />
          </Field>
          <Field label="Robustness filter">
            <select value={robustFilter} onChange={(e) => setRobustFilter(e.target.value)}>
              <option value="all">All results</option>
              <option value="validated">Walk-forward validated</option>
              <option value="dsr">DSR ≥ 0.95</option>
              <option value="oos">OOS evaluated</option>
            </select>
          </Field>
          <Field label="Split ranking">
            <select value={split} onChange={(e) => setSplit(e.target.value)}>
              <option value="train">Ranked on train</option>
              <option value="validation">Ranked on validation</option>
            </select>
          </Field>
          <Field label="Limit">
            <select value={top} onChange={(e) => setTop(Number(e.target.value))}>
              <option value={10}>Top 10</option>
              <option value={50}>Top 50</option>
              <option value={200}>Top 200</option>
            </select>
          </Field>
        </div>

        {filteredAndSorted.length === 0 ? (
          <div className="empty">
            {datasetId ? 'No eligible strategies yet. Register custom indicators/strategies in the Studios and run AI Agents or Optimization.' : 'Import or generate a dataset first.'}
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th />
                  <th>#</th>
                  <th>Strategy</th>
                  <th>Indicators</th>
                  <th>Parameters</th>
                  <th className="right sortable" onClick={() => toggleSort('net_profit')}>Net Profit</th>
                  <th className="right sortable" onClick={() => toggleSort('win_rate')}>Win Rate</th>
                  <th className="right sortable" onClick={() => toggleSort('profit_factor')}>Profit Factor</th>
                  <th className="right sortable" onClick={() => toggleSort('sharpe_ratio')}>Sharpe Ratio</th>
                  <th className="right sortable" onClick={() => toggleSort('max_drawdown_pct')}>Max Drawdown</th>
                  <th className="right sortable" onClick={() => toggleSort('trades')}>Trades</th>
                  <th className="right sortable" onClick={() => toggleSort('oos')}>OOS Performance</th>
                  <th>Robustness</th>
                </tr>
              </thead>
              <tbody>
                {filteredAndSorted.map((r) => (
                  <tr
                    key={r.strategy_hash}
                    className={`clickable ${detail?.hash === r.strategy_hash ? 'selected' : ''}`}
                    onClick={() => open(r)}
                  >
                    <td onClick={(e) => toggleCompare(r, e)}>
                      <input type="checkbox" readOnly checked={compareIds.includes(r.strategy_hash)} />
                    </td>
                    <td className="right">{r.rank}</td>
                    <td>
                      <strong className="mono">{r.spec?.strategy_name || r.strategy_hash}</strong>
                      <div className="muted mono" style={{ fontSize: 11 }}>{r.strategy_hash}</div>
                    </td>
                    <td>
                      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                        {(r.spec?.indicators || []).map((i) => (
                          <span key={i.name} className="chip mono">{i.name}</span>
                        ))}
                      </div>
                    </td>
                    <td className="mono" style={{ maxWidth: 240, fontSize: 11 }}>{formatParams(r.spec)}</td>
                    <td className="right" style={{ color: r.metrics.net_profit >= 0 ? 'var(--good)' : 'var(--bad)' }}>
                      {fmtMoney(r.metrics.net_profit)}
                    </td>
                    <td className="right">{fmtPct(r.metrics.win_rate)}</td>
                    <td className="right">{fmtNum(r.metrics.profit_factor)}</td>
                    <td className="right">{fmtNum(r.metrics.sharpe_ratio ?? 0)}</td>
                    <td className="right">{fmtMoney(r.metrics.max_drawdown)} ({fmtPct(r.metrics.max_drawdown_pct)})</td>
                    <td className="right">{r.metrics.trades}</td>
                    <td className="right">{renderOos(r)}</td>
                    <td>{renderRobustness(r)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      {compareRows.length > 0 && (
        <Card title={`Strategy Comparison (${compareRows.length}/3)`} actions={<button className="ghost" onClick={() => setCompareIds([])}>Clear</button>}>
          <Table
            rowKey={(r) => r.strategy_hash}
            columns={[
              { key: 'strategy', label: 'Strategy', render: (r) => <span className="mono">{r.spec?.strategy_name || r.strategy_hash}</span> },
              { key: 'indicators', label: 'Indicators', render: (r) => (r.spec?.indicators || []).map((i) => i.name).join(', ') },
              { key: 'net_profit', label: 'Net Profit', align: 'right', render: (r) => fmtMoney(r.metrics.net_profit) },
              { key: 'sharpe_ratio', label: 'Sharpe', align: 'right', render: (r) => fmtNum(r.metrics.sharpe_ratio ?? 0) },
              { key: 'win_rate', label: 'Win Rate', align: 'right', render: (r) => fmtPct(r.metrics.win_rate) },
              { key: 'profit_factor', label: 'PF', align: 'right', render: (r) => fmtNum(r.metrics.profit_factor) },
              { key: 'max_dd', label: 'Max DD %', align: 'right', render: (r) => fmtPct(r.metrics.max_drawdown_pct) },
              { key: 'trades', label: 'Trades', align: 'right', render: (r) => r.metrics.trades },
              { key: 'oos', label: 'OOS', align: 'right', render: renderOos },
              { key: 'robustness', label: 'Robustness', render: renderRobustness },
            ]}
            rows={compareRows}
          />
        </Card>
      )}

      {detail && (
        <Card title={`Strategy detail · ${detail.spec?.strategy_name || detail.hash}`} actions={<button className="primary" onClick={toBacktest}>Open in Backtest</button>}>
          <div className="split-2">
            <div>
              <h3 style={{ marginBottom: 8 }}>Configuration</h3>
              <div className="mono" style={{ marginBottom: 10 }}>{specText(detail.spec)}</div>
              <table>
                <thead><tr><th>Indicator</th><th>Parameters</th></tr></thead>
                <tbody>
                  {(detail.spec?.indicators || []).map((i) => (
                    <tr key={i.name}>
                      <td className="mono">{i.name}</td>
                      <td className="mono">{Object.entries(i.params || {}).map(([k, v]) => `${k}=${v}`).join(', ')}</td>
                    </tr>
                  ))}
                  {(!detail.spec?.indicators || detail.spec.indicators.length === 0) && (
                    <tr><td colSpan="2" className="muted">Standalone custom strategy code</td></tr>
                  )}
                </tbody>
              </table>
            </div>
            <div>
              <h3 style={{ marginBottom: 8 }}>Performance by split</h3>
              <Table
                rowKey={(_e, i) => i}
                columns={[
                  { key: 'split', label: 'Split' },
                  { key: 'net', label: 'Net', align: 'right', render: (e) => fmtMoney(e.metrics.net_profit) },
                  { key: 'sharpe', label: 'Sharpe', align: 'right', render: (e) => fmtNum(e.metrics.sharpe_ratio ?? 0) },
                  { key: 'win', label: 'Win %', align: 'right', render: (e) => fmtPct(e.metrics.win_rate) },
                  { key: 'pf', label: 'PF', align: 'right', render: (e) => fmtNum(e.metrics.profit_factor) },
                  { key: 'dd', label: 'Max DD %', align: 'right', render: (e) => fmtPct(e.metrics.max_drawdown_pct) },
                  { key: 'tr', label: 'Trades', align: 'right', render: (e) => e.metrics.trades },
                ]}
                rows={detail.experiments}
              />
              {best && <div className="muted" style={{ marginTop: 6 }}>Recorded {fmtTime(best.created_at)} by {best.agent}{best.study ? ` · study ${best.study}` : ''}</div>}
            </div>
          </div>
          <h3 style={{ margin: '16px 0 8px' }}>Walk-forward validation</h3>
          <Table
            rowKey={(v) => v.id}
            columns={[
              { key: 'created_at', label: 'Checked', render: (v) => fmtTime(v.created_at) },
              { key: 'profitable_segments', label: 'Profitable segments', align: 'right', render: (v) => `${v.profitable_segments} / ${v.total_segments}` },
              { key: 'validation_net_profit', label: 'Validation net', align: 'right', render: (v) => fmtMoney(v.validation_net_profit) },
              { key: 'validated', label: 'Status', render: (v) => <Badge status={v.validated ? 'validated' : 'failed'} /> },
            ]}
            rows={detail.validations}
            empty="Not walk-forward validated yet."
          />
          <h3 style={{ margin: '16px 0 8px' }}>Statistical Robustness &amp; Out-of-Sample</h3>
          {detail.dsr && (
            <div className="info-box">
              Deflated Sharpe Ratio (DSR): <strong>{fmtNum(detail.dsr.dsr, 4)}</strong> · Expected Max Sharpe: {fmtNum(detail.dsr.expected_max_sharpe, 5)} · Trials: {detail.dsr.n_trials} · {detail.dsr.significant ? 'Statistically Significant (DSR ≥ 0.95)' : 'Below 0.95 significance threshold'}
            </div>
          )}
          {detail.out_of_sample
            ? <div className="info-box">OOS Net {fmtMoney(detail.out_of_sample.metrics.net_profit)} · Sharpe {fmtNum(detail.out_of_sample.metrics.sharpe_ratio ?? 0)} · {detail.out_of_sample.metrics.trades} trades · win rate {fmtPct(detail.out_of_sample.metrics.win_rate)} · recorded {fmtTime(detail.out_of_sample.created_at)}</div>
            : <div className="muted">OOS not run. Run it once from the Backtest page when the strategy is final.</div>}
        </Card>
      )}
    </>
  );
}
