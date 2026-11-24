// Realistic backtest rules shared by individual and combination research. Defaults come from the backend.
export const DEFAULT_RULES = {
  direction: 'both', stop_loss_pct: 0.5, target_pct: 1.0, max_hold_bars: null,
  brokerage_per_order: 20, charges_pct: 0.0005, slippage_points: 0.5,
};

export default function RulesForm({ value, onChange, disabled }) {
  const set = (k, v) => onChange({ ...value, [k]: v });
  const num = (k) => (e) => set(k, e.target.value === '' ? '' : Number(e.target.value));
  return (
    <div className="form-grid rules-grid">
      <label className="field"><span>Direction</span>
        <select disabled={disabled} value={value.direction} onChange={(e) => set('direction', e.target.value)}>
          <option value="both">Long and short</option><option value="long">Long only</option><option value="short">Short only</option>
        </select></label>
      <label className="field"><span>Stop-loss %</span>
        <input type="number" step="0.05" min="0" max="10" disabled={disabled} value={value.stop_loss_pct} onChange={num('stop_loss_pct')} /></label>
      <label className="field"><span>Target %</span>
        <input type="number" step="0.05" min="0" max="20" disabled={disabled} value={value.target_pct} onChange={num('target_pct')} /></label>
      <label className="field"><span>Max hold (bars, blank = none)</span>
        <input type="number" min="1" disabled={disabled} value={value.max_hold_bars ?? ''}
          onChange={(e) => set('max_hold_bars', e.target.value === '' ? null : Number(e.target.value))} /></label>
      <label className="field"><span>Brokerage per order</span>
        <input type="number" step="1" min="0" disabled={disabled} value={value.brokerage_per_order} onChange={num('brokerage_per_order')} /></label>
      <label className="field"><span>Charges (fraction of turnover)</span>
        <input type="number" step="0.0001" min="0" disabled={disabled} value={value.charges_pct} onChange={num('charges_pct')} /></label>
      <label className="field"><span>Slippage (points)</span>
        <input type="number" step="0.1" min="0" disabled={disabled} value={value.slippage_points} onChange={num('slippage_points')} /></label>
    </div>
  );
}

// Payload for the API. Blank numeric fields are rejected by the backend, so they are converted here.
export function rulesPayload(v) {
  return {
    direction: v.direction,
    stop_loss_pct: Number(v.stop_loss_pct),
    target_pct: Number(v.target_pct),
    max_hold_bars: v.max_hold_bars === '' || v.max_hold_bars === null ? null : Number(v.max_hold_bars),
    brokerage_per_order: Number(v.brokerage_per_order),
    charges_pct: Number(v.charges_pct),
    slippage_points: Number(v.slippage_points),
  };
}
