import { useMemo } from 'react';
import Plot from './Plot.jsx';
import { Card, Table } from './ui.jsx';

// Optional analysis. Pareto is over stored individual metrics (return up, win rate up, drawdown down).
// Clusters come from the latest combination run's Spearman matrix. Nothing here runs a backtest.
export function ParetoCard({ analysis }) {
  const points = analysis?.points || [];
  const front = new Set((analysis?.pareto?.front || []).map((p) => p.source_key));
  const figure = useMemo(() => ({
    data: [
      { type: 'scatter', mode: 'markers', name: 'Other', x: points.filter((p) => !front.has(p.source_key)).map((p) => p.max_drawdown_pct),
        y: points.filter((p) => !front.has(p.source_key)).map((p) => p.net_return_pct), text: points.filter((p) => !front.has(p.source_key)).map((p) => p.name), marker: { color: '#64748b', size: 9 } },
      { type: 'scatter', mode: 'markers', name: 'Pareto front', x: points.filter((p) => front.has(p.source_key)).map((p) => p.max_drawdown_pct),
        y: points.filter((p) => front.has(p.source_key)).map((p) => p.net_return_pct), text: points.filter((p) => front.has(p.source_key)).map((p) => p.name), marker: { color: '#f59e0b', size: 12 } },
    ],
    layout: { xaxis: { title: 'Max drawdown %' }, yaxis: { title: 'Net return %' }, margin: { l: 60, r: 16, t: 12, b: 50 } },
  }), [points, front]);
  return (
    <Card title="Pareto optimisation (return, win rate, risk)">
      {points.length === 0 ? <div className="empty">Available after an individual evaluation.</div> : (
        <>
          <Plot data={figure.data} layout={figure.layout} height={320} />
          <Table
            rowKey={(p) => p.source_key}
            rows={(analysis.pareto?.front || [])}
            columns={[
              { key: 'name', label: 'Front indicator' },
              { key: 'net_return_pct', label: 'Net return %', align: 'num', render: (p) => Number(p.net_return_pct).toFixed(2) },
              { key: 'raw_win_rate', label: 'Win %', align: 'num', render: (p) => Number(p.raw_win_rate).toFixed(2) },
              { key: 'max_drawdown_pct', label: 'Max DD %', align: 'num', render: (p) => Number(p.max_drawdown_pct).toFixed(2) },
            ]}
          />
          <small className="muted">{front.size} of {points.length} indicators are non-dominated. No weights are applied.</small>
        </>
      )}
    </Card>
  );
}

export function ClustersCard({ analysis }) {
  const c = analysis?.clusters;
  return (
    <Card title="Indicator clusters (Spearman)">
      {!c ? <div className="empty">Available after a combination run finishes.</div> : (
        <>
          <ul className="cluster-list">
            {c.clusters.map((cl, i) => <li key={i}><b>Cluster {i + 1}</b>: {cl.names ? cl.names.join(', ') : JSON.stringify(cl)}</li>)}
          </ul>
          <small className="muted">Links at |ρ| ≥ {c.threshold}. Shown for review; nothing is removed automatically.</small>
        </>
      )}
    </Card>
  );
}
