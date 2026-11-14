// Mission Control: watch the AquaMind crew investigate one case (recording, rebuilt from logs, or live).
import { get, post, stream, state } from './api.js';
import { Player, Timeline } from './player.js';
import { Scene, rosterEl } from './scene.js';
import { CAPS } from './narrate.js';
import { h, $, esc, pct, num, ROLES, modal, toast } from './ui.js';
import { icon } from './icons.js';

let S = null;       // view state

export function initMission(root, boot) {
  const net = boot.network;
  root.innerHTML = '';
  const wrap = h('div', { class: 'mission' });
  root.append(wrap);

  // ---------------- left: case card + inbox
  const left = h('div', { class: 'm-left' });
  const caseCard = h('div', { class: 'panel case-card' });
  const title = h('div', { class: 'case-title' }, 'Choose a case');
  const fid = h('span', { class: 'badge-fid rec' }, 'recording');
  const sub = h('div', { class: 'case-sub' }, '');
  const blurb = h('div', { class: 'case-blurb' }, '');
  const gPL = gauge('P(leak)', 'Is there a leak at all?');
  const gPS = gauge('P(success)', 'Would a dig at the best spot find it?', [0.35, 0.8]);
  const pipsC = h('span', { class: 'pips' }), pipsD = h('span', { class: 'pips digs' });
  const clock = h('span', { class: 'mono' }, '');
  const gate = h('div', { class: 'gate closed' }, h('div', { class: 'g-h', html: `${icon('lock', 13)} Dig gate · closed` }), h('div', { class: 'g-why' }, '—'));
  caseCard.append(
    h('div', { class: 'row between' }, h('div', { class: 'grow' }, title), fid),
    sub, blurb,
    h('div', { class: 'row', style: { marginTop: '10px', gap: '6px' } },
      h('button', { class: 'btn sm', onclick: () => openPicker(boot), html: `${icon('folder', 14)} Cases` }),
      h('button', { class: 'btn sm', id: 'btn-live', onclick: () => startLive(boot), html: `${icon('live', 14)} Run live`, title: 'Run the real crew on this case now' })),
    h('div', { class: 'gauges' }, gPL.el, gPS.el),
    h('div', { class: 'budget' }, h('span', { title: 'field checks left (5 per case)' }, 'Checks left', pipsC), h('span', { title: 'digs made (at most 2, SOP-2)' }, 'Digs', pipsD), h('span', { class: 'grow' }), h('span', { html: icon('clock', 13) }), clock),
    gate);
  const inbox = h('div', { class: 'panel inbox' });
  const inboxCount = h('span', { class: 'chip mono' }, '0');
  const inboxList = h('div', { class: 'inbox-list' });
  inbox.append(h('div', { class: 'panel-h' }, h('span', { html: icon('chat', 14) }), h('span', { class: 'panel-t' }, 'Citizen inbox'), inboxCount, h('span', { class: 'grow' }),
    h('span', { class: 'tiny muted' }, 'raw text → fine-tuned reader')), inboxList);
  left.append(caseCard, inbox);

  // ---------------- map
  const mapPanel = h('div', { class: 'panel m-map' });
  const mapWrap = h('div', { class: 'map-wrap' });
  const caption = h('div', { class: 'caption' }, h('span', { class: 'txt muted' }, 'Pick a case and press play.'));
  const legend = legendEl();
  const scada = h('div', { class: 'scada hidden' });
  const debrief = h('div', { class: 'debrief hidden' });
  const mapTitle = h('div', { class: 'map-title' }, h('b', {}, 'L-Town'), h('span', { class: 'muted' }, ` · ${net.stats.junctions} junctions · ${net.stats.pipes} pipes · ${Math.round(net.stats.km)} km`));
  const zoomBtns = h('div', { class: 'zoom-btns' },
    h('button', { class: 'icon-btn', title: 'Zoom in', html: icon('plus', 16), onclick: () => S.scene.map.flyTo(S.scene.map.s.best ?? null, (S.scene.map.view.k || 1) * 1.5, 300) }),
    h('button', { class: 'icon-btn', title: 'Reset view (double-click the map)', html: icon('target', 16), onclick: () => S.scene.map.reset() }),
    h('button', { class: 'icon-btn', title: 'Follow the action (camera flies to each field check and dig)', id: 'btn-follow', style: { color: 'var(--accent)' }, html: icon('eye', 16), onclick: e => { S.scene.follow = !S.scene.follow; e.currentTarget.style.color = S.scene.follow ? 'var(--accent)' : ''; } }));
  mapWrap.append(h('div', { class: 'map-overlay map-tl' }, mapTitle, legend), h('div', { class: 'map-overlay map-tr' }, scada),
    h('div', { class: 'map-overlay map-bl' }, zoomBtns), caption, debrief);
  mapPanel.append(mapWrap);

  // ---------------- right: war room
  const right = h('div', { class: 'panel m-right' });
  const roster = rosterEl();
  const feed = h('div', { class: 'feed' });
  const capStrip = h('div', { class: 'cap-strip' }, ['ft', 'rag', 'rs', 'ag', 'tc'].map(k => h('div', { class: 'cap-c', 'data-k': k, style: { color: CAPS[k][1] } }, h('b', {}, '0'), h('span', {}, CAPS[k][0]))));
  right.append(h('div', { class: 'panel-h' }, h('span', { html: icon('radar', 14) }), h('span', { class: 'panel-t' }, 'War room'), h('span', { class: 'grow' }),
    h('span', { class: 'tiny muted' }, 'LLM agents judge · code computes and checks')), roster, capStrip, feed);

  // ---------------- timeline
  const tl = h('div', { class: 'panel m-tl' });
  const playBtn = h('button', { class: 'icon-btn play', title: 'Play / pause (space)', html: icon('play', 18) });
  const counter = h('span', { class: 'mono small muted' }, '0 / 0');
  const speed = h('div', { class: 'seg' }, [1, 2, 4].map(s => h('button', { class: s === 1 ? 'on' : '', onclick: e => { S.player.setSpeed(s); speed.querySelectorAll('button').forEach(b => b.classList.toggle('on', b === e.currentTarget)); } }, `${s}×`)));
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('class', 'tl-svg');
  tl.append(h('div', { class: 'tl-head' },
    h('div', { class: 'tl-controls' },
      h('button', { class: 'icon-btn', title: 'Restart', html: icon('first', 15), onclick: () => S.player.seek(0) }),
      h('button', { class: 'icon-btn', title: 'Back (←)', html: icon('prev', 14), onclick: () => S.player.step(-1) }),
      playBtn,
      h('button', { class: 'icon-btn', title: 'Forward (→)', html: icon('next', 14), onclick: () => S.player.step(1) }),
      h('button', { class: 'icon-btn', title: 'Jump to the end', html: icon('last', 15), onclick: () => S.player.seek(S.player.events.length - 1) })),
    speed, counter, h('div', { class: 'grow' }),
    h('div', { class: 'tl-legend' }, h('span', {}, h('i', { style: { background: 'var(--accent)' } }), 'P(leak)'), h('span', { title: 'P(success) of a dig at the best spot' }, h('i', { style: { background: 'var(--good)' } }), 'P(success)'),
      ...['forensic', 'skeptic', 'strategist', 'auditor'].map(r => h('span', { class: 'role' }, h('i', { style: { background: ROLES[r].hex, width: '8px', height: '8px', borderRadius: '50%' } }), ROLES[r].short)))),
    svg);
  wrap.append(left, mapPanel, right, tl);

  const scene = new Scene({ net, mapEl: mapWrap, feedEl: feed, padTop: 150, padBottom: 86, inboxEl: inboxList, inboxCount, captionEl: caption, rosterEl: roster,
    onState: (D, cur, i) => onState(D, cur, i) });
  const player = new Player((i, prev) => scene.apply(i, prev));
  const timeline = new Timeline(svg, i => player.seek(i));
  player.on(p => {
    playBtn.innerHTML = icon(p.playing ? 'pause' : 'play', 18);
    counter.textContent = `${Math.max(0, p.i + 1)} / ${p.events.length}${p.live ? ' · live' : ''}`;
    timeline.set(p.events, p.i);
  });
  playBtn.addEventListener('click', () => player.toggle());
  legend.querySelectorAll('.li').forEach(li => li.addEventListener('click', () => {
    const k = li.dataset.layer; const on = !scene.map.layers[k];
    scene.map.setLayer(k, on); li.classList.toggle('off', !on);
  }));

  S = { boot, scene, player, timeline, els: { title, fid, sub, blurb, gPL, gPS, pipsC, pipsD, clock, gate, scada, debrief, caption }, current: null };

  function onState(D, cur, i) {
    const st = D.st || {}, meta = scene.meta || {}, E = S.els;
    E.gPL.set(st.p_leak);
    E.gPS.set(st.p_success);
    const budget = (meta.budget && meta.budget.checks) || 5;
    const left_ = st.checks_left != null ? st.checks_left : budget;
    E.pipsC.innerHTML = Array.from({ length: budget }, (_, k) => `<i class="${k < budget - left_ ? 'used' : ''}"></i>`).join('');
    E.pipsD.innerHTML = Array.from({ length: 2 }, (_, k) => `<i class="${k < (st.digs || 0) ? '' : 'used'}"></i>`).join('');
    E.clock.textContent = st.clock ? `${st.clock}` : (st.t_h != null ? `T+${num(st.t_h, 1)} h` : '');
    if (st.gate) {
      const open = !!st.gate[0];
      E.gate.className = `gate ${open ? 'open' : 'closed'}`;
      E.gate.innerHTML = `<div class="g-h">${icon(open ? 'shieldCheck' : 'lock', 13)} Dig gate · ${open ? 'open' : 'closed'}</div><div class="g-why">${esc(st.gate[1])}</div>`;
    } else if (cur.kind === 'meta' || (cur.kind === 'tool' && cur.tool === 'open')) {
      E.gate.className = 'gate closed';
      E.gate.innerHTML = `<div class="g-h">${icon('lock', 13)} Dig gate · closed</div><div class="g-why">${meta.fidelity === 'log' ? 'Rebuilt from the evaluation log: the gate state is shown at each dig.' : '—'}</div>`;
    }
    renderCaps(capStrip, S.player.events, i, scene);
    const open = S.player.events.find(e => e.kind === 'tool' && e.tool === 'open');
    renderScada(E.scada, open, meta);
    if (cur.kind === 'end') showDebrief(cur);
    else E.debrief.classList.add('hidden');
  }
}

function renderCaps(box, evs, i, scene) {
  // how often each capability of the proposal has been used so far in this case
  const c = { ft: 0, rag: 0, rs: 0, ag: 0, tc: 0 };
  for (let k = 0; k <= i && k < evs.length; k++) {
    const e = evs[k];
    if (e.kind === 'tool' && e.tool === 'open') c.ft += (e.messages || []).length;
    if (e.kind === 'messages') c.ft += (e.messages || []).length;
    if (e.kind === 'tool' && e.tool === 'call_back') c.ft += (e.replies || []).filter(r => r.text).length;
    if (e.kind === 'turn') { c.ag++; if (e.role === 'auditor') c.rag++; }
    if (e.kind === 'turn_done' && ((e.slate || []).length || e.role === 'skeptic')) c.rs++;
    if (e.kind === 'tool' && ['use_record', 'use_repair_record', 'search_docs'].includes(e.tool)) c.rag++;
    if (e.kind === 'tool' && e.tool === 'triangulate') c.rs++;
    if (e.kind === 'tool' && e.tool !== 'open' && !['use_record', 'use_repair_record', 'triangulate'].includes(e.tool)) c.tc++;
  }
  const tips = { ft: 'messages and call-back replies read by the fine-tuned Qwen2.5-3B reader', rag: 'archive / repair-history citations and SOP retrievals (BM25)',
    rs: 'hypothesis slates scored, Skeptic debates, triangulations', ag: 'LLM agent turns', tc: 'tool calls: call-backs, field checks, digs, closing' };
  box.querySelectorAll('.cap-c').forEach(el => { el.querySelector('b').textContent = c[el.dataset.k]; el.title = tips[el.dataset.k]; });
}

function gauge(label, hint, marks = []) {
  const val = h('div', { class: 'val' }, '–');
  const fill = h('i', { style: { width: '0%' } });
  const bar = h('div', { class: 'bar' }, fill, ...marks.map(m => h('span', { class: 'mark', style: { left: `${100 * m}%`, opacity: m < 0.5 ? 0.5 : 1 }, title: m === 0.8 ? 'dig bar 0.80' : 'last-resort bar 0.35' })));
  const el = h('div', { class: 'gauge', title: hint }, h('div', { class: 'lbl' }, label), val, bar);
  return { el, set(v) { val.textContent = v == null ? '–' : pct(v); fill.style.width = `${100 * (v || 0)}%`; fill.style.background = v >= 0.8 ? 'var(--good)' : v >= 0.35 ? 'var(--warn)' : 'var(--accent)'; } };
}

function legendEl() {
  const li = (layer, sw, text) => h('div', { class: 'li', 'data-layer': layer }, h('span', { class: 'sw', style: sw }), text);
  return h('div', { class: 'legend' },
    li('heat', { background: 'linear-gradient(90deg,#2563eb,#22d3ee,#fde047,#ef4444)' }, 'leak map'),
    li('hyp', { border: '2px dashed var(--forensic)' }, 'hypotheses'),
    li('msgs', { background: '#fb923c', borderRadius: '50% 50% 50% 0', transform: 'rotate(-45deg)' }, 'complaints'),
    li('sensors', { background: 'var(--accent)', transform: 'rotate(45deg) scale(.8)' }, 'pressure sensors'),
    li('checks', { border: '2px solid var(--field)', borderRadius: '50%' }, 'field checks'),
    li('sections', { background: 'linear-gradient(90deg,#60a5fa,#a78bfa,#34d399)' }, 'step-test sections'),
    li('cand', { background: 'var(--forensic)', borderRadius: '50%', transform: 'scale(.5)' }, 'simulated junctions'),
    li('landmarks', { background: '#fcd34d', borderRadius: '50%', transform: 'scale(.6)' }, 'schools · hospitals'));
}

function renderScada(box, open, meta) {
  if (!open || !open.sensors) { box.classList.add('hidden'); return; }
  box.classList.remove('hidden');
  const sc = open.scada;
  const rows = open.sensors.map(s => {
    let spark = '';
    if (sc && sc.pressure && sc.pressure[s.node]) {
      const v = sc.pressure[s.node], n = v.length, mn = Math.min(...v), mx = Math.max(...v), W = 120, H = 20;
      const d = v.map((y, i) => `${i ? 'L' : 'M'}${(i / (n - 1) * W).toFixed(1)},${(H - (y - mn) / Math.max(1e-6, mx - mn) * (H - 2) - 1).toFixed(1)}`).join('');
      spark = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none"><path d="${d}" fill="none" stroke="currentColor" stroke-width="1.2" opacity=".85"/></svg>`;
    }
    const z = s.z || 0, col = Math.abs(z) >= 3 ? 'var(--bad)' : Math.abs(z) >= 1.5 ? 'var(--warn)' : 'var(--accent)';
    return `<div class="s-row" style="color:${col}"><span class="mono" style="color:var(--text-2)">${esc(s.node)}</span>${spark || '<span></span>'}<span class="z">z ${num(z, 1)}</span></div>`;
  }).join('');
  box.innerHTML = `<div class="row between" style="margin-bottom:4px"><span class="panel-t" style="font-size:10px">SCADA · last 7 days</span><span class="tiny muted">inflow ${open.inflow ? (open.inflow.q_hat >= 0 ? '+' : '') + num(open.inflow.q_hat, 1) + ' L/s' : ''}</span></div>${rows}<div class="tiny muted" style="margin-top:4px">A small leak barely moves ${meta.k || 5} sensors: this is the problem.</div>`;
}

function showDebrief(ev) {
  const E = S.els, r = ev.result || {}, T = ev.truth || {};
  if (!E.debrief.classList.contains('hidden') && E.debrief.dataset.k === String(ev.i)) return;
  E.debrief.dataset.k = String(ev.i);
  const good = T.has_leak ? r.found : !r.n_digs;
  const st = ev.state || {};
  const shown = new Set(Object.keys(st.msgs || {}));
  const msgs = Object.entries(T.messages || {}).filter(([rid]) => shown.has(rid)).map(([, m]) => m);
  const real = msgs.filter(m => m.kind === 'leak').length;
  const debunked = Object.entries(st.msgs || {}).filter(([rid, m]) => m.eps != null && m.eps >= 0.85 && (T.messages[rid] || {}).kind !== 'leak').length;
  const stat = (v, k) => h('div', { class: 'stat' }, h('div', { class: 'v' }, v), h('div', { class: 'k' }, k));
  const card = h('div', { class: 'card' },
    h('div', { class: 'big', style: { color: good ? 'var(--good)' : 'var(--bad)' }, html: `${icon(good ? 'check' : 'warn', 28)} ${T.has_leak ? (r.found ? 'Leak found' : 'Leak not dug') : (r.n_digs ? 'Crew sent, no leak' : 'No leak — no dig')}` }),
    h('div', { class: 'muted', style: { marginTop: '4px' } }, T.has_leak ? `True leak: ${num(T.leak.lps, 1)} L/s ${T.leak.kind || ''} leak on pipe ${T.leak.pipe} (section ${T.leak.section}). ${r.guess_error_m != null ? 'Final dig / guess ' + num(r.guess_error_m, 0) + ' m away (success means ≤ 100 m).' : ''}` :
      'There was no network leak: the right answer is to close the case without sending a crew.'),
    h('div', { class: 'stats' },
      stat(r.guess_error_m != null ? `${num(r.guess_error_m, 0)} m` : '–', 'distance'), stat(`${r.checks_used ?? '–'} / 5`, 'field checks'),
      stat(`${r.n_digs ?? 0} / 2`, 'digs'), stat(`${r.unsafe ?? 0}`, 'unsafe actions')),
    h('div', { class: 'stats' },
      stat(`${r.hours_open != null ? num(r.hours_open, 0) + ' h' : '–'}`, 'case open'), stat(`${real} / ${msgs.length}`, 'messages about the leak'),
      stat(`${debunked}`, 'look-alikes debunked'), stat(`${r.violations ?? 0}`, 'SOP violations')),
    h('div', { class: 'small muted' }, `${(r.llm_calls || 0) + (r.llm_cached || 0)} LLM decisions (${r.llm_cached || 0} from cache) · ${r.tool_calls || 0} tool calls · every number computed by code.`),
    h('div', { class: 'row', style: { marginTop: '14px', gap: '8px' } },
      h('button', { class: 'btn primary', onclick: () => E.debrief.classList.add('hidden'), html: `${icon('eye', 15)} Look at the map` }),
      h('button', { class: 'btn', onclick: () => { E.debrief.classList.add('hidden'); S.player.seek(0); S.player.play(); }, html: `${icon('first', 15)} Replay` }),
      h('button', { class: 'btn', onclick: () => openPicker(S.boot), html: `${icon('folder', 15)} Another case` })));
  E.debrief.innerHTML = '';
  E.debrief.append(card);
  setTimeout(() => { if (S.player.i === ev.i) E.debrief.classList.remove('hidden'); }, 900);
}

function setHeader(meta, kind) {
  const E = S.els, suites = S.boot.suites || {};
  const su = suites[meta.suite] || { name: meta.suite, blurb: '' };
  E.title.textContent = su.name || meta.case;
  E.fid.className = `badge-fid ${kind === 'live' ? 'live' : kind === 'log' ? 'log' : 'rec'}`;
  E.fid.textContent = kind === 'live' ? '● live' : kind === 'log' ? (meta.stage === 'mid' ? 'from log · mid-report run' : meta.stage === 'final' ? 'from log · final run' : 'from log') : 'recording';
  E.fid.title = kind === 'log' ? `Rebuilt from ${meta.source || 'the evaluation log'}: every turn and field result${meta.map ? ', in exact order, with leak-map snapshots and the callers\' answers' : ', no leak-map snapshots'}` : kind === 'live' ? 'The real crew is running now' : `Full-fidelity recording of the real crew (${meta.recorded_at || ''})`;
  E.sub.innerHTML = `<span class="mono">${esc(meta.case)}</span> · opened ${esc(meta.opened_at || '')} · ${meta.k || 5} sensors${meta.method_name ? ' · ' + esc(meta.method_name) : ''}`;
  E.blurb.textContent = su.blurb || '';
  S.current = { meta, kind };
}

export async function loadRecording(file, { autoplay = true, seek = null } = {}) {
  const rec = await get(`/api/recording?file=${encodeURIComponent(file)}`);
  S.scene.load(rec);
  setHeader(rec.meta, 'rec');
  S.player.load(rec.events, { start: 1 });
  if (seek != null) S.player.seek(seek);
  else if (autoplay) setTimeout(() => S.player.play(), 400);
}

export async function loadReplay(method, caseName, { autoplay = true, seek = null, ver = null } = {}) {
  const rec = await get(`/api/replay?method=${method}&case=${encodeURIComponent(caseName)}${ver ? `&ver=${encodeURIComponent(ver)}` : ''}`);
  S.scene.load(rec);
  setHeader(rec.meta, 'log');
  S.player.load(rec.events, { start: 1 });
  if (seek != null) S.player.seek(seek);
  else if (autoplay) setTimeout(() => S.player.play(), 400);
}

export async function startLive(boot, caseName, online = null) {
  const c = caseName || (S.current && S.current.meta && S.current.meta.case);
  if (!c) return openPicker(boot);
  const st = await get('/api/status');
  if (!st.api_keys) { toast('No NVIDIA API key in .env: live runs need the LLM. Recordings and log replays work offline.', 'warn'); }
  const liveBtn = document.getElementById('btn-live');
  if (liveBtn) { liveBtn.disabled = true; liveBtn.innerHTML = `<span class="spinner"></span> Live run…`; }
  const info = await post('/api/live/start', { case: c, online: online == null ? !!st.api_keys : online });
  if (info.note) toast(info.note, 'warn');
  const meta = { case: info.case, suite: info.case.split('-')[1], k: 5 };
  S.scene.load({ events: [] });
  setHeader(meta, 'live');
  S.player.load([], { live: true });
  S.player.play();
  let started = false;
  stream(info.id, ev => {
    if (ev.kind === 'meta') { S.scene.meta = ev.meta; S.scene.ctx.meta = ev.meta; S.scene.map.setState({ sensors: ev.meta.sensors }); setHeader(ev.meta, 'live'); }
    S.scene.events = S.player.events;          // the scene reads the player's list
    S.player.append(ev);
    S.scene.events = S.player.events;
    if (!started) { started = true; }
  }, done => {
    S.player.live = false;
    if (liveBtn) { liveBtn.disabled = false; liveBtn.innerHTML = `${icon('live', 14)} Run live`; }
    toast(done.saved ? `Live run finished · saved as recording ${done.saved}` : done.error ? `Live run failed: ${done.error.slice(0, 120)}` : 'Live run finished', done.error ? 'bad' : 'good');
    get('/api/recordings').then(r => { S.boot.recordings = r; });
  });
}

export function openPicker(boot) {
  const recs = boot.recordings || [];
  const body = h('div', {});
  body.append(h('div', { class: 'eyebrow' }, 'Full-fidelity recordings of the real crew'),
    h('p', { class: 'small muted', style: { margin: '4px 0 10px' } }, 'Real runs of the final AquaMind crew, saved step by step (demo_ui record): every agent turn with the board it saw, every tool call, and the leak map over all 782 junctions. Each one matches its row in the final evaluation.'));
  const grid = h('div', { class: 'rec-grid' });
  if (!recs.length) grid.append(h('div', { class: 'muted small' }, 'No recordings yet. Create one with: .venv/bin/python demo_ui record --cases dev-brown-002'));
  for (const r of recs) {
    const good = r.has_leak ? r.found : !r.digs;
    grid.append(h('div', { class: 'rec-card', onclick: () => { m.close(); loadRecording(r.file); } },
      h('div', { class: 'row between' }, h('div', { class: 'rt' }, r.suite_name), h('span', { class: `chip ${good ? 'good' : 'bad'}` }, r.has_leak ? (r.found ? `found · ${num(r.error_m, 0)} m` : r.outcome) : (r.digs ? 'dug, no leak' : 'no leak · no dig'))),
      h('div', { class: 'rb' }, r.suite_blurb),
      h('div', { class: 'row between tiny muted' }, h('span', { class: 'mono' }, r.case), h('span', {}, `${r.checks} checks · ${r.events} events`))));
  }
  body.append(grid);
  // all cases from the logs
  body.append(h('div', { class: 'eyebrow', style: { marginTop: '22px' } }, 'All 210 benchmark cases — rebuilt from the evaluation logs'),
    h('p', { class: 'small muted', style: { margin: '4px 0 10px' } }, 'The final evaluation run (5 sensors), exactly as logged in logs/evaluation: every agent turn, tool call, caller answer and field result in its real order, with a leak-map snapshot after each step. "Recording" plays the full-detail recording where one exists; the play icon runs the crew live on the case.'));
  const q = h('input', { class: 'field', placeholder: 'Filter: case, type, outcome…', style: { maxWidth: '320px' } });
  const suiteSel = h('select', { class: 'field', style: { maxWidth: '230px' } }, h('option', { value: '' }, 'All case types'),
    ...Object.entries(boot.suites).map(([k, v]) => h('option', { value: k }, v.name)));
  const tbl = h('table', { class: 'cases' });
  const render = () => {
    const f = q.value.toLowerCase(), su = suiteSel.value;
    const rows = boot.cases.filter(c => (!su || c.suite === su) && (!f || `${c.case} ${c.suite_name} ${c.crew?.outcome} ${c.react?.outcome}`.toLowerCase().includes(f)));
    tbl.innerHTML = '';
    tbl.append(h('tr', {}, ['Case', 'Type', 'Leak', 'AquaMind crew', 'ReAct agent', ''].map(t => h('th', {}, t))));
    for (const c of rows) {
      const badge = o => !o ? h('span', { class: 'faint' }, '–') : h('span', { class: `chip ${c.has_leak ? (o.found ? 'good' : 'bad') : (o.digs ? 'bad' : 'good')}` }, c.has_leak ? (o.found ? `found ${num(o.err, 0)} m` : o.outcome) : (o.digs ? 'dug, no leak' : 'no dig'));
      tbl.append(h('tr', {},
        h('td', { class: 'mono' }, c.case), h('td', {}, c.suite_name), h('td', { class: 'mono' }, c.has_leak ? `${num(c.q, 1)} L/s` : 'none'),
        h('td', {}, badge(c.crew)), h('td', {}, badge(c.react)),
        h('td', { style: { textAlign: 'right', whiteSpace: 'nowrap' } },
          h('button', { class: 'btn sm', onclick: () => { m.close(); loadReplay('agentix_crew', c.case); } }, 'Replay'), ' ',
          c.recorded ? h('button', { class: 'btn sm', onclick: () => { m.close(); loadRecording(`${c.case}__spread_k5.json`); } }, 'Recording') : null, ' ',
          h('button', { class: 'btn sm ghost', title: 'Run the current crew live on this case', onclick: () => { m.close(); startLive(boot, c.case); }, html: icon('live', 13) }))));
    }
  };
  q.addEventListener('input', render);
  suiteSel.addEventListener('change', render);
  body.append(h('div', { class: 'row', style: { gap: '8px', marginBottom: '8px' } }, q, suiteSel), h('div', { style: { maxHeight: '46vh', overflow: 'auto' } }, tbl));
  render();
  const m = modal('Choose an investigation', body, { wide: true });
}

export function missionKeys(e) {
  if (!S) return false;
  if (e.key === ' ') { e.preventDefault(); S.player.toggle(); return true; }
  if (e.key === 'ArrowRight') { S.player.step(1); return true; }
  if (e.key === 'ArrowLeft') { S.player.step(-1); return true; }
  if (e.key === 'Home') { S.player.seek(0); return true; }
  if (e.key === 'End') { S.player.seek(S.player.events.length - 1); return true; }
  if (e.key === 'c' || e.key === 'C') { openPicker(S.boot); return true; }
  return false;
}

export function missionTheme() { if (S) S.scene.map.setTheme(); }
export function missionReady() { return !!(S && S.current); }
