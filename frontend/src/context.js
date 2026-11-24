import { createContext, useContext } from 'react';

// Shared state: the selected dataset and the strategy currently being edited.
export const AppContext = createContext({ datasets: [], datasetId: null, setDatasetId: () => {}, reloadDatasets: () => {} });
export const useApp = () => useContext(AppContext);

export const STRATEGY_KEY = 'il_strategy';
export function loadStrategy() {
  try {
    const raw = localStorage.getItem(STRATEGY_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}
export function saveStrategy(spec) {
  localStorage.setItem(STRATEGY_KEY, JSON.stringify(spec));
}
