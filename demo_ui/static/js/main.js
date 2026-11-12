// App shell: navigation, theme, keyboard shortcuts, status pills; views are built lazily on first visit.
import { get, state } from './api.js';
import { h, $, toast } from './ui.js';
import { icon } from './icons.js';
import { initOverview } from './overview.js';
import { initMission, loadRecording, loadReplay, missionKeys, missionTheme, missionReady, openPicker, startLive } from './mission.js';
import { initH2H, loadH2H, h2hKeys, h2hTheme, H2H_DEFAULT } from './h2h.js';
import { initSafety } from './safety.js';
import { initLab } from './lab.js';
import { initResults } from './results.js';

const VIEWS = [
  { id: 'overview', label: 'Overview', icon: 'dropFill' },
  { id: 'mission', label: 'Mission Control', icon: 'radar' },
  { id: 'h2h', label: 'Crew vs ReAct', icon: 'swords' },
  { id: 'safety', label: 'Safety Lab', icon: 'shieldCheck' },
  { id: 'lab', label: 'Reader & RAG', icon: 'chip' },
  { id: 'results', label: 'Results', icon: 'chart' },
];
const built = new Set();
let boot = null, current = null;

async function go(id, arg) {
  if (!VIEWS.some(v => v.id === id)) id = 'overview';
  current = id;
  document.querySelectorAll('.view').forEach(v => v.classList.toggle('active', v.id === `view-${id}`));
  document.querySelectorAll('#nav button').forEach(b => b.classList.toggle('active', b.dataset.view === id));
  if (location.hash !== `#${id}`) history.replaceState(null, '', `#${id}`);
  const root = $(`#view-${id}`);
  try {
    if (!built.has(id)) {
      built.add(id);
      if (id === 'overview') await initOverview(root, boot, go);
      if (id === 'mission') {
        initMission(root, boot);
        const first = (boot.recordings || [])[0];
        const P = new URLSearchParams(location.search), seek = P.has('seek') ? parseInt(P.get('seek'), 10) : null;
        if (!arg && P.get('live')) await startLive(boot, P.get('live'), P.get('online') === '0' ? false : null);
        else if (!arg && P.get('file')) await loadRecording(P.get('file'), { autoplay: false, seek });
        else if (!arg && P.get('case')) await loadReplay(P.get('method') || 'agentix_crew', P.get('case'), { autoplay: false, seek });
        else if (!arg) { if (first) await loadRecording(first.file, { autoplay: false, seek }); else await loadReplay('agentix_crew', 'test-brown-002', { autoplay: false, seek }); }
        if (P.has('picker')) openPicker(boot);
      }
      if (id === 'h2h') { initH2H(root, boot); if (!arg) await loadH2H(new URLSearchParams(location.search).get('case') || H2H_DEFAULT); }
      if (id === 'safety') await initSafety(root, boot, go);
      if (id === 'lab') await initLab(root, boot);
      if (id === 'results') await initResults(root);
    }
    if (id === 'mission' && arg) { if (arg.file) await loadRecording(arg.file); else if (arg.case) await loadReplay(arg.method || 'agentix_crew', arg.case); }
    if (id === 'h2h' && arg) await loadH2H(arg);
  } catch (e) {
    console.error(e);
    toast(`Could not open ${id}: ${e.message}`, 'bad');
  }
}

function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem('aquamind-theme', t); } catch (e) { /* private mode */ }
  $('#btn-theme').innerHTML = icon(t === 'light' ? 'moon' : 'sun', 17);
  missionTheme(); h2hTheme();
  // re-render charts that read CSS variables at draw time is automatic (SVG uses var()); canvases need a redraw
}

async function refreshStatus() {
  try {
    const s = await get('/api/status');
    const rd = $('#st-reader'), lv = $('#st-live');
    const r = s.reader || {};
    rd.className = `status-dot ${r.server || r.mlx_loaded ? 'ok' : r.mlx_allowed ? 'warn' : ''}`;
    rd.title = r.server ? `Fine-tuned reader server at ${r.server}` : r.mlx_loaded ? 'Fine-tuned reader loaded in-process (MLX)' : r.mlx_allowed ? 'Fine-tuned reader: cached reads now; loads the local MLX model on the first new message' : 'Fine-tuned reader: cached reads only (rule reader fallback for new text)';
    rd.querySelector('span').textContent = r.server ? 'Reader · server' : r.mlx_loaded ? 'Reader · MLX' : 'Reader · cache';
    lv.className = `status-dot ${s.api_keys ? 'ok' : ''}`;
    lv.title = s.api_keys ? `${s.api_keys} NVIDIA API key(s): live runs possible` : 'No API key: replays only';
    lv.querySelector('span').textContent = s.live && s.live.length ? 'Live · running' : s.api_keys ? `Live · ${s.api_keys} keys` : 'Live · off';
  } catch (e) { /* server restarting */ }
}

async function start() {
  if (new URLSearchParams(location.search).has('noanim')) document.documentElement.dataset.noanim = '1';
  let t = new URLSearchParams(location.search).get('theme');
  try { t = t || localStorage.getItem('aquamind-theme'); } catch (e) { /* private mode */ }
  setTheme(t || 'dark');
  $('#logo').innerHTML = icon('dropFill', 17);
  $('#btn-full').innerHTML = icon('expand', 16);
  $('#btn-full').addEventListener('click', () => document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen());
  $('#btn-theme').addEventListener('click', () => setTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light'));
  const nav = $('#nav');
  VIEWS.forEach((v, k) => nav.append(h('button', { 'data-view': v.id, onclick: () => go(v.id), title: `${v.label} (${k + 1})` },
    h('span', { html: icon(v.icon, 15) }), h('span', { class: 'lbl' }, v.label), h('span', { class: 'kbd' }, `${k + 1}`))));
  document.querySelector('.brand').addEventListener('click', () => go('overview'));

  boot = await get('/api/bootstrap');
  state.boot = boot;
  $('#loading').remove();
  refreshStatus();
  setInterval(refreshStatus, 15000);

  document.addEventListener('keydown', e => {
    if (e.target.closest('input, textarea, select') || e.metaKey || e.ctrlKey || e.altKey) return;
    const n = parseInt(e.key, 10);
    if (n >= 1 && n <= VIEWS.length) { go(VIEWS[n - 1].id); return; }
    if (e.key === 't' || e.key === 'T') { setTheme(document.documentElement.dataset.theme === 'light' ? 'dark' : 'light'); return; }
    if (e.key === 'f' || e.key === 'F') { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen(); return; }
    if (current === 'mission' && missionKeys(e)) return;
    if (current === 'h2h' && h2hKeys(e)) return;
  });
  const hash = (location.hash || '#overview').slice(1).split('?')[0];
  go(hash);
}

start().catch(e => {
  console.error(e);
  const l = $('#loading');
  if (l) l.innerHTML = `<div style="text-align:center;color:var(--bad)">Could not reach the demo server: ${e.message}<br><span style="color:var(--muted)">Start it with: .venv/bin/python demo_ui</span></div>`;
});
