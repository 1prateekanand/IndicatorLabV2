import { useEffect, useMemo, useState } from 'react';
import { get, post, api, fmtTime } from '../api.js';
import { useApp } from '../context.js';
import { Card, Badge, Field, ErrorBox, Stat, Table, useAsync } from '../components/ui.jsx';
import CodeEditor from '../components/CodeEditor.jsx';

// Structural skeleton only (no predefined trading logic or indicator formula)
const EMPTY_SKELETON = `def compute(df, **params):
    """Return a causal Series/list of bar signals in {-1, 0, 1} aligned to df.
    Available columns: df['open'], df['high'], df['low'], df['close'], df['volume'].
    """
    n = len(df)
    return [0] * n
`;

function blankDraft() {
  return {
    id: null,
    name: '',
    version: '1.0.0',
    category: 'Custom',
    description: '',
    params: [],
    code: EMPTY_SKELETON,
    status: 'new',
    registered: false,
  };
}

export default function IndicatorStudio() {
  const { datasetId } = useApp();
  const [items, setItems] = useState([]);
  const [query, setQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [draft, setDraft] = useState(blankDraft);
  const [savedSnapshot, setSavedSnapshot] = useState(JSON.stringify(blankDraft()));
  const [validation, setValidation] = useState(null);
  const [notice, setNotice] = useState(null);
  const { error, busy, run, setError } = useAsync();

  const loadList = async (selectId = null) => {
    const res = await get('/api/studio/indicators');
    setItems(res.items || []);
    if (selectId) {
      const found = (res.items || []).find((i) => i.id === selectId);
      if (found) selectItem(found);
    }
  };

  useEffect(() => {
    run(async () => {
      const res = await get('/api/studio/indicators');
      const list = res.items || [];
      setItems(list);
      if (list.length > 0 && !draft.id) selectItem(list[0]);
    });
  }, []);

  const selectItem = (item) => {
    const copy = {
      id: item.id,
      name: item.name,
      version: item.version || '1.0.0',
      category: item.category || 'Custom',
      description: item.description || '',
      params: (item.params || []).map((p) => ({ ...p })),
      code: item.code || EMPTY_SKELETON,
      status: item.status,
      registered: item.registered,
    };
    setDraft(copy);
    setSavedSnapshot(JSON.stringify(copy));
    setValidation(item.last_validation || null);
    setNotice(null);
    setError(null);
  };

  const newIndicator = () => {
    const fresh = blankDraft();
    setDraft(fresh);
    setSavedSnapshot(JSON.stringify(fresh));
    setValidation(null);
    setNotice(null);
    setError(null);
  };

  const dirty = JSON.stringify(draft) !== savedSnapshot;

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return items.filter((i) => {
      if (statusFilter === 'registered' && !i.registered) return false;
      if (statusFilter === 'draft' && i.registered) return false;
      if (statusFilter === 'error' && i.status !== 'error') return false;
      if (!q) return true;
      return i.name.toLowerCase().includes(q) || (i.description || '').toLowerCase().includes(q) || (i.category || '').toLowerCase().includes(q);
    });
  }, [items, query, statusFilter]);

  const addParam = () => {
    setDraft((d) => ({
      ...d,
      params: [...d.params, { name: `param_${d.params.length + 1}`, kind: 'int', default: 14, low: 2, high: 100, step: 1 }],
    }));
  };

  const updateParam = (idx, key, val) => {
    setDraft((d) => ({
      ...d,
      params: d.params.map((p, i) => (i === idx ? { ...p, [key]: val } : p)),
    }));
  };

  const removeParam = (idx) => {
    setDraft((d) => ({ ...d, params: d.params.filter((_, i) => i !== idx) }));
  };

  const persistDraft = async () => {
    const payload = {
      name: draft.name,
      version: draft.version,
      category: draft.category,
      description: draft.description,
      params: draft.params,
      code: draft.code,
    };
    if (draft.id) {
      return api(`/api/studio/indicators/${draft.id}`, { method: 'PUT', body: payload });
    }
    return post('/api/studio/indicators', payload);
  };

  const saveDraft = () => run(async () => {
    setNotice(null);
    const saved = await persistDraft();
    await loadList(saved.id);
    setNotice(`Saved draft '${saved.name}' v${saved.version}. Click Update Indicator to validate and register it in the Research Agent catalogue.`);
  });

  const validateCode = (syntaxOnly = false) => run(async () => {
    setNotice(null);
    const res = await post('/api/studio/indicators/validate', {
      code: draft.code,
      params: draft.params,
      dataset_id: datasetId,
      syntax_only: syntaxOnly,
    });
    setValidation(res);
    if (!res.ok) {
      setError(`${res.error?.type || 'ValidationError'} on line ${res.error?.line || 1}: ${res.error?.message}`);
    }
  });

  const updateIndicator = () => run(async () => {
    setNotice(null);
    const saved = await persistDraft();
    const res = await post(`/api/studio/indicators/${saved.id}/update`, {
      name: draft.name,
      version: draft.version,
      category: draft.category,
      description: draft.description,
      params: draft.params,
      code: draft.code,
      dataset_id: datasetId,
    });
    setValidation(res.validation);
    await loadList(res.indicator.id);
    setNotice(res.message);
  });

  const deleteIndicator = (item) => run(async () => {
    setNotice(null);
    await api(`/api/studio/indicators/${item.id}`, { method: 'DELETE' });
    const res = await get('/api/studio/indicators');
    const remaining = res.items || [];
    setItems(remaining);
    if (draft.id === item.id) {
      if (remaining.length) selectItem(remaining[0]);
      else newIndicator();
    }
    setNotice(`Deleted indicator '${item.name}' and removed it from the dynamic registry.`);
  });

  const registeredCount = items.filter((i) => i.registered && i.available).length;
  const errorLine = validation && !validation.ok ? (validation.error?.line || null) : null;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Indicator Studio</h1>
          <div className="muted">
            Code-first custom technical indicators. Write Python in an isolated sandbox, test against historical candles, and explicitly update the dynamic registry.
          </div>
        </div>
        <div className="card-actions">
          <Badge status="info" />
          <span className="badge info">Implemented: Python 3.11 Sandbox</span>
          <button className="primary" onClick={newIndicator}>+ New Indicator</button>
        </div>
      </div>

      <div className="grid stats" style={{ marginBottom: 14 }}>
        <Stat label="Custom Indicators" value={items.length} hint="unlimited user-owned registry" />
        <Stat label="Registered & Active" value={registeredCount} tone="good" hint="visible to Research Agent" />
        <Stat label="Drafts / Errors" value={items.length - registeredCount} tone={items.some((i) => i.status === 'error') ? 'bad' : ''} />
        <Stat label="Test Dataset" value={datasetId ? `#${datasetId}` : 'Probe (400 bars)'} hint={datasetId ? 'train split historical candles' : 'import CSV or synthetic data'} />
      </div>

      <ErrorBox error={error} />
      {notice && <div className="info-box">{notice}</div>}

      <div className="studio-shell">
        {/* Left pane: Registered & Draft custom indicators list */}
        <aside className="studio-list">
          <div className="studio-toolbar">
            <strong>Custom Indicators ({items.length})</strong>
          </div>
          <input
            placeholder="Search indicators..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <select value={statusFilter} onChange={(e) => setStatusFilter(e.target.value)}>
            <option value="all">All ({items.length})</option>
            <option value="registered">Registered ({registeredCount})</option>
            <option value="draft">Drafts ({items.filter((i) => !i.registered).length})</option>
            <option value="error">Errors ({items.filter((i) => i.status === 'error').length})</option>
          </select>

          {filtered.length === 0 && (
            <div className="empty" style={{ padding: 14 }}>
              No custom indicators yet. Click <strong>+ New Indicator</strong> to author your first Python indicator.
            </div>
          )}

          {filtered.map((item) => (
            <div
              key={item.id}
              className={`studio-item ${draft.id === item.id ? 'active' : ''}`}
              onClick={() => selectItem(item)}
            >
              <div className="studio-item-head">
                <strong className="mono">{item.name}</strong>
                <Badge status={item.registered ? 'completed' : item.status === 'error' ? 'error' : 'pending'} />
              </div>
              <div className="muted" style={{ fontSize: 12 }}>
                v{item.version} · {item.category} · {(item.params || []).length} param(s)
              </div>
              {item.description && <div className="muted" style={{ fontSize: 11 }}>{item.description}</div>}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: 4 }}>
                <small>{item.registered ? 'In Agent Catalogue' : 'Draft only'}</small>
                <button
                  className="ghost"
                  style={{ padding: '2px 7px', fontSize: 11, color: 'var(--bad)' }}
                  onClick={(e) => { e.stopPropagation(); deleteIndicator(item); }}
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
        </aside>

        {/* Center/Right workspace */}
        <div className="studio-workspace">
          <Card
            title={draft.id ? `Editing: ${draft.name}` : 'New Custom Indicator'}
            actions={
              <>
                <button disabled={busy} onClick={saveDraft}>Save Draft</button>
                <button disabled={busy} onClick={() => validateCode(true)}>Validate Syntax</button>
                <button disabled={busy} onClick={() => validateCode(false)}>Test on Historical Data</button>
                <button className="primary" disabled={busy} onClick={updateIndicator}>Update Indicator</button>
              </>
            }
          >
            <div className="form-grid" style={{ marginBottom: 12 }}>
              <Field label="Indicator Name (snake_case)" hint="Unique registry identifier">
                <input
                  className="mono"
                  placeholder="e.g. custom_momentum_alpha"
                  value={draft.name}
                  onChange={(e) => setDraft({ ...draft, name: e.target.value })}
                />
              </Field>
              <Field label="Version">
                <input
                  className="mono"
                  value={draft.version}
                  onChange={(e) => setDraft({ ...draft, version: e.target.value })}
                />
              </Field>
              <Field label="Category">
                <input
                  placeholder="Trend / Momentum / Volatility / Custom"
                  value={draft.category}
                  onChange={(e) => setDraft({ ...draft, category: e.target.value })}
                />
              </Field>
              <Field label="Description">
                <input
                  placeholder="What signal logic does this indicator compute?"
                  value={draft.description}
                  onChange={(e) => setDraft({ ...draft, description: e.target.value })}
                />
              </Field>
            </div>

            <div style={{ marginBottom: 12 }}>
              <div className="studio-toolbar" style={{ marginBottom: 6 }}>
                <span style={{ color: 'var(--muted)', fontSize: 13 }}>
                  Custom Parameters ({draft.params.length}) — exposed to Strategy Studio &amp; Research Agent search spaces
                </span>
                <button type="button" onClick={addParam}>+ Add Parameter</button>
              </div>
              {draft.params.length > 0 && (
                <div className="param-row" style={{ color: 'var(--muted)', fontSize: 11, textTransform: 'uppercase' }}>
                  <span>Name</span><span>Type</span><span>Default</span><span>Min</span><span>Max</span><span>Step</span><span />
                </div>
              )}
              {draft.params.map((p, idx) => (
                <div key={idx} className="param-row">
                  <input className="mono" value={p.name} onChange={(e) => updateParam(idx, 'name', e.target.value)} />
                  <select value={p.kind} onChange={(e) => updateParam(idx, 'kind', e.target.value)}>
                    <option value="int">int</option>
                    <option value="float">float</option>
                  </select>
                  <input type="number" step="any" value={p.default} onChange={(e) => updateParam(idx, 'default', e.target.value)} />
                  <input type="number" step="any" value={p.low} onChange={(e) => updateParam(idx, 'low', e.target.value)} />
                  <input type="number" step="any" value={p.high} onChange={(e) => updateParam(idx, 'high', e.target.value)} />
                  <input type="number" step="any" placeholder="auto" value={p.step ?? ''} onChange={(e) => updateParam(idx, 'step', e.target.value)} />
                  <button type="button" className="ghost" style={{ color: 'var(--bad)' }} onClick={() => removeParam(idx)}>Remove</button>
                </div>
              ))}
            </div>

            <CodeEditor
              value={draft.code}
              onChange={(code) => setDraft({ ...draft, code })}
              filename={`${draft.name || 'new_indicator'}.py`}
              errorLine={errorLine}
              dirty={dirty}
              height={360}
            />
          </Card>

          <Card
            title="Validation &amp; Historical Candle Execution"
            actions={validation ? <Badge status={validation.ok ? 'ok' : 'error'} /> : null}
          >
            {!validation && (
              <div className="empty">
                Click <strong>Validate Syntax</strong>, <strong>Test on Historical Data</strong>, or <strong>Update Indicator</strong> to run your Python code inside the isolated sandbox.
              </div>
            )}
            {validation && !validation.ok && (
              <div className="error-box">
                <strong>{validation.error?.type || 'Error'}</strong> (line {validation.error?.line || 1}): {validation.error?.message}
              </div>
            )}
            {validation && validation.ok && validation.stage === 'syntax' && (
              <div className="info-box">Python AST syntax and sandbox safety checks passed.</div>
            )}
            {validation && validation.ok && validation.stage === 'executed' && (
              <>
                <div className="grid stats" style={{ marginBottom: 12 }}>
                  <Stat label="Execution Time" value={`${validation.execution_ms} ms`} />
                  <Stat label="Bars Evaluated" value={validation.bars} hint={validation.data_source} />
                  <Stat label="Bullish (+1)" value={validation.signal_counts?.bullish ?? 0} tone="good" />
                  <Stat label="Bearish (-1)" value={validation.signal_counts?.bearish ?? 0} tone="bad" />
                  <Stat label="Neutral (0)" value={validation.signal_counts?.neutral ?? 0} />
                  <Stat
                    label="Causality Check"
                    value={validation.causal ? 'Causal' : 'Look-ahead'}
                    tone={validation.causal ? 'good' : 'bad'}
                    hint={validation.causality_note}
                  />
                </div>
                {validation.sample && validation.sample.length > 0 && (
                  <Table
                    rowKey={(r, i) => i}
                    columns={[
                      { key: 'time', label: 'Timestamp', render: (r) => <span className="mono">{r.time}</span> },
                      { key: 'close', label: 'Close', align: 'right', render: (r) => Number(r.close).toFixed(2) },
                      {
                        key: 'signal',
                        label: 'Signal',
                        align: 'right',
                        render: (r) => (
                          <Badge status={r.signal === 1 ? 'ok' : r.signal === -1 ? 'error' : 'muted'} />
                        ),
                      },
                    ]}
                    rows={validation.sample}
                  />
                )}
              </>
            )}
            {draft.id && items.find((i) => i.id === draft.id)?.last_validated_at && (
              <div className="muted" style={{ marginTop: 8, fontSize: 12 }}>
                Last validated: {fmtTime(items.find((i) => i.id === draft.id).last_validated_at)}
              </div>
            )}
          </Card>
        </div>
      </div>
    </>
  );
}
