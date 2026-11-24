import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';

const THEME = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#c9d1e0', family: 'Inter, system-ui, sans-serif', size: 12 },
  xaxis: { gridcolor: '#232a38', zerolinecolor: '#232a38' },
  yaxis: { gridcolor: '#232a38', zerolinecolor: '#232a38' },
  margin: { l: 56, r: 20, t: 36, b: 44 },
  legend: { orientation: 'h', y: -0.2 },
};

// Renders Plotly figures. ``figure`` is {data, layout}; layout is merged with the dark theme.
export default function Plot({ data, layout = {}, height = 360, onRelayout }) {
  const ref = useRef(null);
  const relayoutRef = useRef(onRelayout);
  relayoutRef.current = onRelayout;
  useEffect(() => {
    if (!ref.current || !data) return;
    const merged = { ...THEME, ...layout, height, xaxis: { ...THEME.xaxis, ...(layout.xaxis || {}) },
      yaxis: { ...THEME.yaxis, ...(layout.yaxis || {}) } };
    Plotly.react(ref.current, data, merged, { displaylogo: false, responsive: true });
    // Optional: report user zoom/pan (or autoscale). Plotly adds .on() to the element only after react().
    const el = ref.current;
    if (el.removeAllListeners) el.removeAllListeners('plotly_relayout');
    if (el.on) el.on('plotly_relayout', (ev) => { if (relayoutRef.current) relayoutRef.current(ev); });
    return () => {};
  }, [data, layout, height]);
  useEffect(() => () => { if (ref.current) Plotly.purge(ref.current); }, []);
  return <div ref={ref} style={{ width: '100%', height }} />;
}
