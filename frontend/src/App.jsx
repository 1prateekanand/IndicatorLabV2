import { useCallback, useEffect, useMemo, useState } from 'react';
import { NavLink, Navigate, Route, Routes } from 'react-router-dom';
import { AppContext } from './context.js';
import { get, post } from './api.js';
import { ErrorBox } from './components/ui.jsx';
import Dashboard from './pages/Dashboard.jsx';
import IndicatorResearch from './pages/IndicatorResearch.jsx';
import Strategy from './pages/Strategy.jsx';
import MarketData from './pages/MarketData.jsx';
import Agents from './pages/Agents.jsx';
import IndicatorStudio from './pages/IndicatorStudio.jsx';
import StrategyStudio from './pages/StrategyStudio.jsx';
import Optimization from './pages/Optimization.jsx';
import Backtest from './pages/Backtest.jsx';
import Results from './pages/Results.jsx';
import Experiments from './pages/Experiments.jsx';
import Analytics from './pages/Analytics.jsx';

// The main navigation has exactly four sections. Other pages stay reachable by URL and from the
// "Other pages" list on the Dashboard.
import { OTHER_PAGES } from './navigation.js';

const NAV = [
  ['/dashboard', 'Dashboard'],
  ['/market-data', 'Market Data'],
  ['/indicator', 'Indicator'],
  ['/strategy', 'Strategy'],
];

export default function App() {
  const [datasets, setDatasets] = useState([]);
  const [datasetId, setDatasetIdState] = useState(() => Number(localStorage.getItem('il_dataset')) || null);
  const [online, setOnline] = useState(null);
  const [dsModal, setDsModal] = useState(false);
  const [dsBusy, setDsBusy] = useState(false);
  const [dsErr, setDsErr] = useState(null);
  const [synthForm, setSynthForm] = useState({ name: 'SYNTH_1m_demo', days: 60, seed: 42 });
  const [csvForm, setCsvForm] = useState({ name: '', symbol: '', timeframe: '1min', csv_text: '' });

  const reloadDatasets = useCallback(async () => {
    try {
      const list = await get('/api/datasets');
      setDatasets(list);
      setOnline(true);
      setDatasetIdState((current) => {
        if (current && list.some((d) => d.id === current)) return current;
        return list.length ? list[list.length - 1].id : null;
      });
    } catch {
      setOnline(false);
    }
  }, []);

  useEffect(() => { reloadDatasets(); }, [reloadDatasets]);

  const setDatasetId = (id) => {
    localStorage.setItem('il_dataset', String(id));
    setDatasetIdState(id);
  };

  const generateSynth = async () => {
    setDsBusy(true);
    setDsErr(null);
    try {
      const ds = await post('/api/datasets/synthetic', {
        name: synthForm.name,
        days: Number(synthForm.days),
        seed: Number(synthForm.seed),
      });
      await reloadDatasets();
      setDatasetId(ds.id);
      setDsModal(false);
    } catch (e) {
      setDsErr(e.message);
    } finally {
      setDsBusy(false);
    }
  };

  const handleFilePick = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const base = file.name.replace(/\.csv$/i, '');
      setCsvForm((f) => ({
        ...f,
        name: f.name || base,
        symbol: f.symbol || base.split('_')[0].toUpperCase(),
        csv_text: String(reader.result || ''),
      }));
    };
    reader.readAsText(file);
  };

  const uploadCsv = async () => {
    setDsBusy(true);
    setDsErr(null);
    try {
      const ds = await post('/api/datasets/upload', csvForm);
      await reloadDatasets();
      setDatasetId(ds.id);
      setDsModal(false);
    } catch (e) {
      setDsErr(e.message);
    } finally {
      setDsBusy(false);
    }
  };

  const value = useMemo(() => ({ datasets, datasetId, setDatasetId, reloadDatasets }),
    [datasets, datasetId, reloadDatasets]);

  return (
    <AppContext.Provider value={value}>
      <div className="layout">
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-mark">IL</div>
            <div>
              <div className="brand-name">Indicator Lab</div>
              <div className="brand-sub">Code-First Indicator &amp; Strategy Studio</div>
            </div>
          </div>
          <nav>
            {NAV.map(([to, label]) => (
              <NavLink key={to} to={to} className={({ isActive }) => (isActive ? 'nav active' : 'nav')}>
                {label}
              </NavLink>
            ))}

          </nav>
          <div className="sidebar-foot">
            <div className={`dot ${online === null ? '' : online ? 'on' : 'off'}`} />
            <span>{online === null ? 'connecting…' : online ? 'API connected' : 'API unreachable'}</span>
          </div>
        </aside>
        <main className="content">
          <div className="topbar" style={{ gap: 10, alignItems: 'center' }}>
            <label className="dataset-picker">
              <span>Dataset</span>
              <select value={datasetId || ''} onChange={(e) => setDatasetId(Number(e.target.value))}>
                {datasets.length === 0 && <option value="">No datasets imported</option>}
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>{d.name} ({d.symbol}, {d.rows.toLocaleString()} bars)</option>
                ))}
              </select>
            </label>
            <button onClick={() => { setDsErr(null); setDsModal(true); }}>+ Import Dataset</button>
          </div>
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/market-data" element={<MarketData />} />
            <Route path="/agents" element={<Agents />} />
            <Route path="/indicator" element={<IndicatorResearch />} />
            <Route path="/indicator-studio" element={<IndicatorStudio />} />
            <Route path="/strategy" element={<Strategy />} />
            <Route path="/strategy-studio" element={<StrategyStudio />} />
            <Route path="/indicators" element={<Navigate to="/indicator-studio" replace />} />
            <Route path="/optimization" element={<Optimization />} />
            <Route path="/backtest" element={<Backtest />} />
            <Route path="/results" element={<Results />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/analytics" element={<Analytics />} />
          </Routes>
        </main>
      </div>

      {dsModal && (
        <div className="modal-backdrop" onClick={() => setDsModal(false)}>
          <div className="modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="card-head">
              <h3>Import Historical Candles</h3>
              <button className="ghost" onClick={() => setDsModal(false)}>Close</button>
            </div>
            <ErrorBox error={dsErr} />
            <div style={{ marginBottom: 16 }}>
              <strong style={{ display: 'block', marginBottom: 8 }}>Upload CSV (OHLCV)</strong>
              <div className="form-grid">
                <label className="field"><span>CSV File</span><input type="file" accept=".csv" onChange={handleFilePick} /></label>
                <label className="field"><span>Dataset Name</span><input value={csvForm.name} onChange={(e) => setCsvForm({ ...csvForm, name: e.target.value })} /></label>
                <label className="field"><span>Symbol</span><input value={csvForm.symbol} onChange={(e) => setCsvForm({ ...csvForm, symbol: e.target.value })} /></label>
              </div>
              <button className="primary" style={{ marginTop: 10 }} disabled={dsBusy || !csvForm.csv_text} onClick={uploadCsv}>
                Upload &amp; Import CSV
              </button>
            </div>
            <hr style={{ border: 0, borderTop: '1px solid var(--line)', margin: '14px 0' }} />
            <div>
              <strong style={{ display: 'block', marginBottom: 4 }}>Generate Synthetic Test Candles</strong>
              <p className="muted" style={{ margin: '0 0 10px', fontSize: 12 }}>
                Random-walk 1-minute bars for pipeline verification only (not real market data).
              </p>
              <div className="form-grid">
                <label className="field"><span>Name</span><input value={synthForm.name} onChange={(e) => setSynthForm({ ...synthForm, name: e.target.value })} /></label>
                <label className="field"><span>Days (5–500)</span><input type="number" min="5" max="500" value={synthForm.days} onChange={(e) => setSynthForm({ ...synthForm, days: e.target.value })} /></label>
                <label className="field"><span>Seed</span><input type="number" value={synthForm.seed} onChange={(e) => setSynthForm({ ...synthForm, seed: e.target.value })} /></label>
              </div>
              <button style={{ marginTop: 10 }} disabled={dsBusy} onClick={generateSynth}>
                Generate Synthetic Dataset
              </button>
            </div>
          </div>
        </div>
      )}
    </AppContext.Provider>
  );
}
