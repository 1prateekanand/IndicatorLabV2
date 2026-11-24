import { useEffect, useState } from 'react';
import { get } from '../api.js';

// Polls one research run while it is running or paused. Returns the latest status from the backend.
export function useRun(runId, intervalMs = 2000) {
  const [run, setRun] = useState(null);
  useEffect(() => {
    if (!runId) { setRun(null); return undefined; }
    let alive = true;
    let timer;
    const tick = async () => {
      try {
        const r = await get(`/api/research/runs/${runId}`);
        if (!alive) return;
        setRun(r);
        if (r.status === 'running' || r.status === 'paused') timer = setTimeout(tick, intervalMs);
      } catch {
        if (alive) timer = setTimeout(tick, intervalMs * 3);
      }
    };
    tick();
    return () => { alive = false; clearTimeout(timer); };
  }, [runId, intervalMs]);
  return run;
}
