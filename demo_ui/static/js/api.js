// Talking to the demo server (demo_ui/server.py).
const cache = new Map();

export async function get(path, { memo = false } = {}) {
  if (memo && cache.has(path)) return cache.get(path);
  const r = await fetch(path);
  const d = await r.json();
  if (!r.ok) throw new Error(d.error || r.statusText);
  if (memo) cache.set(path, d);
  return d;
}

export async function post(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });
  const d = await r.json();
  if (!r.ok) throw new Error(d.error || r.statusText);
  return d;
}

export function stream(runId, onEvent, onDone, from = 0) {
  const es = new EventSource(`/api/live/stream?id=${encodeURIComponent(runId)}&from=${from}`);
  es.onmessage = e => { try { onEvent(JSON.parse(e.data)); } catch (err) { console.error(err); } };
  es.addEventListener('done', e => { es.close(); onDone && onDone(JSON.parse(e.data)); });
  es.onerror = () => { /* the browser retries; the server resumes from the last index we pass on reconnect */ };
  return es;
}

export const state = { boot: null, results: null };
