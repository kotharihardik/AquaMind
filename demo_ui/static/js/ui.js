// DOM helpers, formatting and the shared vocabulary of the UI (agent roles, symptom words).
import { icon } from './icons.js';

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

export function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

export const pct = (x, d = 0) => (x == null || isNaN(x)) ? '–' : `${(100 * x).toFixed(d)}%`;
export const num = (x, d = 0) => (x == null || isNaN(x)) ? '–' : Number(x).toFixed(d);
export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

export const ROLES = {
  forensic: { name: 'Forensic Advocate', short: 'Forensic', color: 'var(--forensic)', hex: '#a78bfa', icon: 'search',
    job: 'Writes its own questions to callers, cites the town archive, reads callers as one story, names the hypotheses.' },
  skeptic: { name: 'Critical Skeptic', short: 'Skeptic', color: 'var(--skeptic)', hex: '#fb7185', icon: 'bolt',
    job: 'Attacks the leading hypothesis with sensor, elevation and archive facts.' },
  strategist: { name: 'Hydraulic Strategist', short: 'Strategist', color: 'var(--strategist)', hex: '#22d3ee', icon: 'compass',
    job: 'Picks the next field check from a menu with code-computed chances.' },
  auditor: { name: 'Safety & SOP Auditor', short: 'Auditor', color: 'var(--auditor)', hex: '#fbbf24', icon: 'shieldCheck',
    job: 'Tries to refute the dig, reads the SOP (RAG) and lists permits and notices.' },
  react: { name: 'ReAct agent', short: 'ReAct', color: 'var(--react)', hex: '#f472b6', icon: 'sparkle', job: 'One LLM that picks every tool call itself.' },
  code: { name: 'Code', short: 'Code', color: 'var(--code)', hex: '#94a3b8', icon: 'cpu', job: 'Physics, probabilities, gates.' },
  reader: { name: 'Fine-tuned reader', short: 'Reader', color: 'var(--reader)', hex: '#34d399', icon: 'chip', job: 'Qwen2.5-3B + QLoRA, local.' },
  operator: { name: 'Duty operator', short: 'Operator', color: 'var(--operator)', hex: '#60a5fa', icon: 'key', job: 'Approves the form of a request; issues one-time tokens.' },
  field: { name: 'Field crew', short: 'Field', color: 'var(--field)', hex: '#fb923c', icon: 'shovel', job: 'Runs step tests, listening surveys and digs.' },
  citizen: { name: 'Citizen', short: 'Citizen', color: 'var(--citizen)', hex: '#cbd5e1', icon: 'user', job: '' },
};

export const SYM = {
  water_coming_out: 'water coming out', burst: 'burst', pooling: 'water collected', wet_ground: 'wet ground', sound_underground: 'sound under road',
  road_damage: 'road damage', low_pressure: 'low pressure', no_water: 'no water', dirty_water: 'dirty water',
};
export const CAUSE = { private_pipe: 'private pipe', garden_or_hose: 'garden / hose', sewer_drain: 'sewer / drain', rain: 'rain', washing: 'washing',
  other: 'other cause', works: "utility's own works", air: 'air in pipes' };

export function avatar(role, size = 28) {
  const r = ROLES[role] || ROLES.code;
  return h('div', { class: 'av', style: { background: r.hex, width: `${size}px`, height: `${size}px` }, html: icon(r.icon, Math.round(size * 0.55)) });
}

export function clockLabel(t_h) {
  if (t_h == null) return '';
  const h_ = Math.floor(t_h), m = Math.round((t_h - h_) * 60);
  return `T+${h_}h${m ? String(m).padStart(2, '0') : ''}`;
}

export function jsonHTML(obj) {
  const s = JSON.stringify(obj, null, 2) ?? '';
  return esc(s).replace(/(&quot;[^&]*?&quot;)(\s*:)?|\b(true|false|null)\b|(-?\d+\.?\d*)/g, (m, str, colon, bool, n) => {
    if (str) return colon ? `<span class="k">${str}</span>${colon}` : `<span class="s">${str}</span>`;
    if (bool) return `<span class="b">${bool}</span>`;
    if (n) return `<span class="n">${n}</span>`;
    return m;
  });
}

export function modal(title, body, { wide = false } = {}) {
  const back = h('div', { class: 'modal-back' });
  const close = () => back.remove();
  const m = h('div', { class: 'modal', style: wide ? { width: 'min(1200px, 96vw)' } : {} },
    h('div', { class: 'modal-h' }, h('h3', {}, title), h('div', { class: 'grow' }), h('button', { class: 'icon-btn', onclick: close, html: icon('x', 16) })),
    h('div', { class: 'modal-b' }, body));
  back.append(m);
  back.addEventListener('click', e => { if (e.target === back) close(); });
  document.addEventListener('keydown', function onk(e) { if (e.key === 'Escape') { close(); document.removeEventListener('keydown', onk); } });
  document.body.append(back);
  return { close, el: m };
}

export function themeColors() {
  const cs = getComputedStyle(document.documentElement);
  const g = n => cs.getPropertyValue(n).trim();
  return { text: g('--text'), muted: g('--muted'), faint: g('--faint'), line: g('--line'), line2: g('--line-2'), accent: g('--accent'),
    good: g('--good'), bad: g('--bad'), warn: g('--warn'), leak: g('--leak'), forensic: g('--forensic'), skeptic: g('--skeptic'),
    strategist: g('--strategist'), auditor: g('--auditor'), field: g('--field'), operator: g('--operator'), panel: g('--panel-solid'),
    bg: g('--bg'), react: g('--react'), light: document.documentElement.dataset.theme === 'light' };
}

export function toast(text, kind = '') {
  const t = h('div', { class: `chip ${kind}`, style: { position: 'fixed', bottom: '22px', left: '50%', transform: 'translateX(-50%)', zIndex: 120,
    padding: '9px 14px', fontSize: '13px', boxShadow: 'var(--shadow)', background: 'var(--panel-solid)' } }, text);
  document.body.append(t);
  setTimeout(() => t.remove(), 3200);
}
