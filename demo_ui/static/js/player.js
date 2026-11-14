// Playback of an investigation (recorded, rebuilt from logs, or streaming live) and its P(leak) / P(success) timeline.
import { duration } from './narrate.js';
import { ROLES, clockLabel, esc } from './ui.js';

export class Player {
  constructor(onIndex) {
    this.onIndex = onIndex;
    this.events = [];
    this.i = -1;
    this.playing = false;
    this.speed = 1;
    this.live = false;
    this.listeners = new Set();
  }
  load(events, { live = false, start = 0 } = {}) {
    this.stop();
    this.events = events.slice();
    this.live = live;
    this.i = -1;
    this.seek(Math.min(start, this.events.length - 1));
  }
  append(ev) {
    if (this.events.some(e => e.i === ev.i && e.kind === ev.kind)) return;
    this.events.push(ev);
    this._emit();
    if (this.playing && this._waiting) { this._waiting = false; this._schedule(0); }
  }
  on(fn) { this.listeners.add(fn); }
  _emit() { for (const fn of this.listeners) fn(this); }
  seek(i) {
    if (!this.events.length) { this.i = -1; this._emit(); return; }
    i = Math.max(0, Math.min(this.events.length - 1, i));
    const prev = this.i;
    this.i = i;
    this.onIndex(i, prev);
    this._emit();
  }
  step(d = 1) { this.pause(); this.seek(this.i + d); }
  play() {
    if (!this.events.length && !this.live) return;
    if (this.i >= this.events.length - 1 && !this.live) this.seek(0);
    this.playing = true;
    this._emit();
    if (!this.events.length || this.i >= this.events.length - 1) { this._waiting = true; if (!this.live) this._schedule(150); return; }
    this._schedule(150);
  }
  pause() { this.playing = false; clearTimeout(this._t); this._waiting = false; this._emit(); }
  stop() { this.pause(); }
  toggle() { this.playing ? this.pause() : this.play(); }
  setSpeed(s) { this.speed = s; this._emit(); }
  _schedule(ms) {
    clearTimeout(this._t);
    this._t = setTimeout(() => {
      if (!this.playing) return;
      if (this.i >= this.events.length - 1) {
        if (this.live) { this._waiting = true; return; }
        this.pause(); return;
      }
      this.seek(this.i + 1);
      const ev = this.events[this.i];
      // skip near-instant events quickly when far behind a live stream
      const behind = this.live ? this.events.length - 1 - this.i : 0;
      const d = duration(ev) / this.speed / (behind > 6 ? 3 : 1);
      this._schedule(d);
    }, ms);
  }
}

const COLOR = ev => {
  if (ev.kind === 'turn' || ev.kind === 'thinking' || ev.kind === 'turn_done') return (ROLES[ev.role] || ROLES.code).hex;
  if (ev.kind === 'messages') return ROLES.reader.hex;
  if (ev.kind === 'end') return '#ff4d6d';
  if (ev.kind === 'tool') {
    if (ev.tool === 'field_check' || ev.tool === 'step_test' || ev.tool === 'listen') return ROLES.field.hex;
    if (ev.tool === 'propose_dig' || ev.tool === 'dig') return (ev.result || {}).leak_found ? '#34d399' : (ev.result || {}).ok ? '#fb7185' : '#94a3b8';
    if (ev.tool === 'call_back' || ev.tool === 'call') return '#cbd5e1';
  }
  return '#94a3b8';
};

export class Timeline {
  constructor(svg, onSeek) {
    this.svg = svg; this.onSeek = onSeek;
    this.events = []; this.i = -1;
    svg.addEventListener('click', e => { const i = this._idx(e); if (i != null) onSeek(i); });
    svg.addEventListener('mousemove', e => this._hover(e));
    svg.addEventListener('mouseleave', () => { this.hi = null; this.render(); });
    new ResizeObserver(() => this.render()).observe(svg);
  }
  set(events, i) { this.events = events; this.i = i; this.render(); }
  _geom() {
    const r = this.svg.getBoundingClientRect();
    return { W: Math.max(100, r.width), H: Math.max(60, r.height), L: 34, R: 12, T: 10, B: 22 };
  }
  _idx(e) {
    const n = this.events.length; if (!n) return null;
    const { W, L, R } = this._geom(), r = this.svg.getBoundingClientRect();
    const x = e.clientX - r.left;
    return Math.max(0, Math.min(n - 1, Math.round((x - L) / Math.max(1, W - L - R) * (n - 1))));
  }
  _hover(e) { this.hi = this._idx(e); this.render(); }
  render() {
    const ev = this.events, n = ev.length, { W, H, L, R, T, B } = this._geom();
    this.svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    if (!n) { this.svg.innerHTML = ''; return; }
    const X = i => L + (n === 1 ? 0 : i / (n - 1)) * (W - L - R);
    const Y = p => T + (1 - p) * (H - T - B);
    let pl = null, ps = null, th = null;
    const pts = ev.map(e => { if (e.state) { if (e.state.p_leak != null) pl = e.state.p_leak; if (e.state.p_success != null) ps = e.state.p_success; if (e.state.t_h != null) th = e.state.t_h; } return [pl, ps, th]; });
    const upto = this.i;
    const path = (k, until) => { let d = '', started = false; for (let i = 0; i <= until; i++) { const v = pts[i][k]; if (v == null) continue; d += `${started ? 'L' : 'M'}${X(i).toFixed(1)},${Y(v).toFixed(1)}`; started = true; } return d; };
    let s = '';
    for (const p of [0, 0.5, 1]) s += `<line x1="${L}" x2="${W - R}" y1="${Y(p)}" y2="${Y(p)}" stroke="var(--line)" />`;
    s += `<line x1="${L}" x2="${W - R}" y1="${Y(0.8)}" y2="${Y(0.8)}" stroke="var(--warn)" stroke-dasharray="4 4" opacity=".8"/>`;
    s += `<text x="${W - R - 2}" y="${Y(0.8) + 12}" text-anchor="end" font-size="10" fill="var(--warn)">dig bar 0.80</text>`;
    for (const p of [0, 0.5, 1]) s += `<text x="${L - 6}" y="${Y(p) + 3}" text-anchor="end" font-size="10" fill="var(--faint)">${Math.round(p * 100)}%</text>`;
    // future (faint) then past (bold)
    s += `<path d="${path(1, n - 1)}" fill="none" stroke="var(--good)" stroke-width="1.2" opacity=".14"/>`;
    s += `<path d="${path(0, n - 1)}" fill="none" stroke="var(--accent)" stroke-width="1.2" opacity=".14"/>`;
    const area = path(1, upto);
    if (area) s += `<path d="${area}L${X(upto)},${Y(0)}L${X(0)},${Y(0)}Z" fill="var(--good)" opacity=".08"/>`;
    s += `<path d="${path(0, upto)}" fill="none" stroke="var(--accent)" stroke-width="2.4" stroke-linejoin="round"/>`;
    s += `<path d="${path(1, upto)}" fill="none" stroke="var(--good)" stroke-width="2.4" stroke-linejoin="round"/>`;
    // event ticks
    for (let i = 0; i < n; i++) {
      const e = ev[i];
      if (e.kind === 'thinking' || e.kind === 'turn_done' || e.kind === 'meta') continue;
      const x = X(i), y = H - B + 9, c = COLOR(e), past = i <= upto;
      const big = e.kind === 'turn' || (e.kind === 'tool' && ['field_check', 'propose_dig', 'dig', 'step_test', 'listen', 'close'].includes(e.tool)) || e.kind === 'end';
      s += big ? `<circle cx="${x}" cy="${y}" r="${e.kind === 'turn' ? 3.6 : 4.2}" fill="${c}" opacity="${past ? 1 : 0.28}"/>` : `<rect x="${x - 1}" y="${y - 3}" width="2" height="6" fill="${c}" opacity="${past ? 0.8 : 0.2}"/>`;
      if (e.kind === 'tool' && (e.tool === 'propose_dig' || e.tool === 'dig') && (e.result || {}).ok)
        s += `<text x="${x}" y="${y - 8}" text-anchor="middle" font-size="12" opacity="${past ? 1 : 0.3}">${(e.result || {}).leak_found ? '✓' : '✕'}</text>`;
    }
    // cursor
    if (upto >= 0) {
      const x = X(upto);
      s += `<line x1="${x}" x2="${x}" y1="${T - 2}" y2="${H - B + 2}" stroke="var(--text)" stroke-width="1.4" opacity=".8"/>`;
      const tv = pts[upto][2];
      if (tv != null) s += `<text x="${Math.min(W - R - 30, Math.max(L + 30, x))}" y="${T + 9}" text-anchor="middle" font-size="10.5" font-weight="700" fill="var(--text-2)">${clockLabel(tv)}</text>`;
    }
    if (this.hi != null && this.hi !== upto) {
      const x = X(this.hi), e = ev[this.hi];
      s += `<line x1="${x}" x2="${x}" y1="${T}" y2="${H - B}" stroke="var(--accent)" stroke-dasharray="2 3"/>`;
      const label = e.kind === 'tool' ? e.tool : e.kind === 'turn' ? (ROLES[e.role] || {}).short : e.kind;
      s += `<text x="${Math.min(W - R - 40, Math.max(L + 40, x))}" y="${H - B - 4}" text-anchor="middle" font-size="10.5" fill="var(--accent)">${esc(label)}</text>`;
    }
    this.svg.innerHTML = s;
  }
}
