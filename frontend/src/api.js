// Thin client for the FastAPI backend. Errors are surfaced to the UI, never hidden.
export async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
    body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
  });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = text; }
  if (!res.ok) {
    const detail = data && data.detail ? (typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)) : res.statusText;
    throw new Error(detail);
  }
  return data;
}

export const get = (p) => api(p);
export const post = (p, body) => api(p, { method: 'POST', body: body ?? {} });
export const put = (p, body) => api(p, { method: 'PUT', body: body ?? {} });

export function fmtMoney(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toLocaleString('en-IN', { maximumFractionDigits: 2, minimumFractionDigits: 2 });
}
export function fmtPct(v, digits = 2) {
  if (v === null || v === undefined) return '—';
  return `${Number(v).toFixed(digits)}%`;
}
export function fmtNum(v, digits = 2) {
  if (v === null || v === undefined) return '—';
  return Number(v).toFixed(digits);
}
export function fmtTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}
export function specText(spec) {
  if (!spec) return '';
  const parts = (spec.indicators || []).map((i) => {
    const ps = Object.entries(i.params || {}).map(([k, v]) => `${k}=${v}`).join(', ');
    return `${i.name}(${ps})`;
  });
  const indPart = parts.length ? parts.join(' + ') : (spec.strategy_name || 'custom_strategy');
  return `${indPart} | votes>=${spec.min_votes ?? 1} | ${spec.direction || 'both'} | SL ${spec.stop_loss_pct ?? 0}% TP ${spec.target_pct ?? 0}%`;
}
