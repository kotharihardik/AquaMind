// One investigation on screen: applies event i to the map, the feed, the inbox, the gauges and the caption.
import { NetworkMap } from './map.js';
import { caption, makeCard } from './narrate.js';
import { h, esc, pct, num, ROLES, SYM, CAUSE, clockLabel } from './ui.js';
import { icon, CHANNEL_ICON } from './icons.js';

export class Scene {
  constructor(opts) {
    this.o = opts;
    this.net = opts.net;
    this.map = new NetworkMap(opts.mapEl, opts.net, { compact: !!opts.compact, labels: !opts.compact, onClick: opts.onMapClick, padTop: opts.padTop, padBottom: opts.padBottom });
    this.events = [];
    this.meta = {};
    this.msgs = new Map();
    this.follow = true;
    this.sel = null;
    // a junction near the middle of each step-test area, for ripples
    this.areaCenter = {};
    for (const s of this.net.sections) {
      const pick = js => { let best = js[0], bd = 1e18; const [cx, cy] = s.c; for (const j of js) { const d = (this.net.nodes[j][0] - cx) ** 2 + (this.net.nodes[j][1] - cy) ** 2; if (d < bd) { bd = d; best = j; } } return best; };
      this.areaCenter[s.id] = pick(s.j);
      for (const p of s.parts || []) this.areaCenter[p.id] = p.j[Math.floor(p.j.length / 2)];
    }
  }

  load(rec) {
    this.rec = rec;
    this.events = rec.events || [];
    const m = this.events.find(e => e.kind === 'meta');
    this.meta = (m && m.meta) || rec.meta || {};
    this.ctx = { net: this.net, meta: this.meta, openTurn: null };
    this.msgs = new Map();
    this.sel = null;
    if (this.o.feedEl) this.o.feedEl.innerHTML = '';
    this.map.setState({ heat: null, cand: null, slate: null, options: null, msgs: {}, msgMeta: {}, checks: [], best: null, p_leak: null, p_success: null,
      sensors: this.meta.sensors || [], sensorZ: {}, stepVals: {}, digs: [], truth: null, finalJ: null, errM: null, selMsg: null });
    this.map.reset();
    this.lastCaption = null;
    this.shownDebrief = false;
  }

  appendEvent(ev) { this.events.push(ev); if (ev.kind === 'meta') { this.meta = ev.meta; this.ctx.meta = ev.meta; } }

  _feedAdd(ev, animate) {
    if (ev.kind === 'tool' && ev.tool === 'open') for (const m of ev.messages || []) this.msgs.set(m.rid, { ...m, order: this.msgs.size, fresh: false });
    if (ev.kind === 'messages') for (const m of ev.messages || []) this.msgs.set(m.rid, { ...m, order: this.msgs.size, fresh: animate });
    if (!this.o.feedEl) return null;
    if (ev.kind === 'thinking' || ev.kind === 'meta') return null;
    const el = makeCard(ev, this.ctx);
    if (el) this.o.feedEl.append(el);
    return el || (ev.kind === 'turn_done' || ev.kind === 'tool' ? this.ctx.openTurn && this.ctx.openTurn.el : null);
  }

  apply(i, prev) {
    const evs = this.events;
    if (i < 0 || !evs.length) return;
    let el = null;
    const full = prev == null || prev < 0 || i < prev || i - prev > 1;
    if (full) {
      if (this.o.feedEl) { this.o.feedEl.innerHTML = ''; this.o.feedEl.classList.add('instant'); setTimeout(() => this.o.feedEl.classList.remove('instant'), 60); }
      this.ctx.openTurn = null;
      this.msgs = new Map();
      for (let k = 0; k <= i; k++) el = this._feedAdd(evs[k], false) || el;
    } else {
      for (let k = prev + 1; k <= i; k++) el = this._feedAdd(evs[k], true) || el;
    }
    if (this.o.feedEl) {
      this.o.feedEl.querySelectorAll('.item.cur').forEach(x => x.classList.remove('cur'));
      const cur = el || this.o.feedEl.lastElementChild;
      if (cur) cur.classList.add('cur');
      if (full) { this.o.feedEl.style.scrollBehavior = 'auto'; this.o.feedEl.scrollTop = this.o.feedEl.scrollHeight; this.o.feedEl.style.scrollBehavior = ''; }
      else this.o.feedEl.scrollTop = this.o.feedEl.scrollHeight;
    }
    const cur = evs[i];
    const D = this.derive(i);
    this.map.setState(D.map);
    if (!full) this._effects(cur, D);
    if (this.o.onState) this.o.onState(D, cur, i);
    const c = caption(cur, this.ctx);
    if (c) this.lastCaption = c;
    if (this.o.captionEl && this.lastCaption) {
      const r = ROLES[this.lastCaption.who] || ROLES.code;
      this.o.captionEl.innerHTML = `<span class="who" style="color:${r.color}">${esc(r.short)}</span><span class="txt">${this.lastCaption.text}</span>`;
    }
    if (this.o.inboxEl) this.renderInbox(D);
    if (this.o.rosterEl) this.renderRoster(i);
    return D;
  }

  derive(i) {
    const evs = this.events;
    let st = {}, heat = null, cand = null, slate = null, stepVals = {}, digs = [], sensorZ = {}, opts = null;
    for (let k = 0; k <= i; k++) {
      const e = evs[k];
      if (e.state) { st = e.state; if (e.state.heat) heat = e.state.heat; if (e.state.cand) cand = e.state.cand; }
      if (e.kind === 'turn_done' && e.slate && e.slate.length) slate = e.slate;
      if (e.kind === 'tool') {
        const r = e.result || {};
        if (e.tool === 'open') for (const s of e.sensors || []) sensorZ[s.j] = s.z;
        if (e.tool === 'field_check' && r.ok && r.excess_flow_lps) Object.assign(stepVals, r.excess_flow_lps);
        if ((e.tool === 'propose_dig' || e.tool === 'dig') && r.ok) digs.push({ j: e.dig_j != null ? e.dig_j : (e.state || {}).best, found: !!r.leak_found });
      }
    }
    const cur = evs[i];
    if (cur.kind === 'turn' && cur.role === 'strategist' && cur.options) {
      const pick = String((cur.decision || {}).choose || '').toUpperCase();
      opts = cur.options.map(o => ({ ...o, pick: o.id === pick }));
    } else if (cur.kind === 'turn_done' && cur.role === 'strategist') {
      const t = [...evs.slice(0, i)].reverse().find(e => e.kind === 'turn' && e.role === 'strategist');
      if (t && t.options) { const pick = String(cur.record.override ? cur.record.override.best : cur.record.choose || '').toUpperCase(); opts = t.options.map(o => ({ ...o, pick: o.id === pick })); }
    }
    // a rebuilt-from-logs replay has no leak map: show where the agents' hypotheses put the mass instead
    let pseudo = false;
    if (!heat && slate) {
      const J = this.net.n_junctions, arr = new Array(J).fill(0);
      for (const x of slate) if (x.region && x.region.length && x.mass) for (const j of x.region) arr[j] += x.mass / x.region.length;
      const mx = Math.max(...arr);
      if (mx > 0) { heat = arr.map(v => Math.round(1000 * v / mx)); pseudo = true; }
    }
    const end = cur.kind === 'end' ? cur : null;
    const msgMeta = {};
    for (const [rid, m] of this.msgs) msgMeta[rid] = { read: m.read, trigger: m.trigger };
    const msgsState = { ...(st.msgs || {}) };
    for (const [rid, m] of this.msgs) if (!msgsState[rid] && m.j != null) msgsState[rid] = { eps: null, usable: false, j: m.j, notes: [] };
    return {
      st, cur, end, pseudo, stepVals, digs,
      map: { heat, cand, slate, options: opts, msgs: msgsState, msgMeta, checks: st.checks || [], best: st.best, p_leak: st.p_leak, p_success: st.p_success,
        sensors: this.meta.sensors || [], sensorZ, stepVals, digs, selMsg: this.sel, hideBest: !!st.done,
        truth: end ? end.truth : null, finalJ: end ? (end.result.final_j != null ? end.result.final_j : (digs.length ? digs[digs.length - 1].j : null)) : null,
        errM: end ? end.result.guess_error_m : null },
    };
  }

  _effects(ev, D) {
    const m = this.map, C = { field: '#fb923c', forensic: '#a78bfa', reader: '#34d399', strat: '#22d3ee', leak: '#ff4d6d' };
    if (ev.kind === 'messages') for (const x of ev.messages || []) if (x.j != null) m.ripple(x.j, C.reader, 160);
    if (ev.kind === 'tool') {
      const r = ev.result || {};
      if (ev.tool === 'open') { m.reset(); for (const s of ev.sensors || []) if (Math.abs(s.z || 0) >= 2) m.ripple(s.j, '#fb7185', 300, 1800); }
      if (ev.tool === 'call_back') for (const rp of ev.replies || []) { const mm = this.msgs.get(rp.rid); if (mm && mm.j != null) m.ripple(mm.j, C.forensic, 130); }
      if (ev.tool === 'field_check' && r.ok) {
        if (r.kind === 'listening' && ev.option && ev.option.center != null) {
          m.ripple(ev.option.center, r.heard ? '#34d399' : C.strat, 250, 1600);
          if (this.follow) m.flyTo(ev.option.center, 1.7, 900);
        }
        if (r.kind === 'step test') {
          const hot = (r.clearly_lit || []).concat(r.suggestive || []);
          for (const sid of hot) if (this.areaCenter[sid] != null) m.ripple(this.areaCenter[sid], C.field, 420, 1800);
          if (this.follow && hot.length && this.areaCenter[hot[0]] != null) m.flyTo(this.areaCenter[hot[0]], 1.5, 900);
          else if (this.follow) m.reset();
        }
      }
      if ((ev.tool === 'propose_dig' || ev.tool === 'dig') && r.ok && ev.dig_j != null) {
        m.ripple(ev.dig_j, r.leak_found ? '#34d399' : '#fb7185', 220, 1800);
        if (this.follow) m.flyTo(ev.dig_j, 2.1, 900);
      }
    }
    if (ev.kind === 'end' && ev.truth && ev.truth.leak && this.follow) {
      m.flyTo(ev.truth.leak.j, 2.0, 1100);
      setTimeout(() => m.ripple(ev.truth.leak.j, C.leak, 260, 2200), 600);
    }
  }

  select(rid) {
    this.sel = this.sel === rid ? null : rid;
    this.map.setState({ selMsg: this.sel });
    const mm = this.msgs.get(rid);
    if (this.sel && mm && mm.j != null) this.map.flyTo(mm.j, 2.4, 650);
    if (this.o.inboxEl && this._lastD) this.renderInbox(this._lastD);
  }

  renderInbox(D) {
    this._lastD = D;
    const box = this.o.inboxEl, st = D.st || {}, T = D.end && D.end.truth;
    const items = [...this.msgs.values()].sort((a, b) => b.order - a.order);
    if (this.o.inboxCount) this.o.inboxCount.textContent = `${items.length}`;
    box.innerHTML = '';
    for (const m of items) {
      const s = (st.msgs || {})[m.rid] || {};
      const rd = m.read || {};
      const quarantine = rd.contains_instructions;
      const debunked = s.eps != null && s.eps >= 0.85;
      const el = h('div', { class: `msg${m.fresh ? ' new' : ''}${debunked ? ' debunked' : ''}${quarantine ? ' quarantine' : ''}${this.sel === m.rid ? ' sel open' : ''}` });
      m.fresh = false;
      el.append(h('div', { class: 'm-h' }, h('span', { html: icon(CHANNEL_ICON[m.ch] || 'chat', 13) }), h('b', {}, m.rid), h('span', {}, (m.at || '').replace('T', ' ').slice(5)),
        m.trigger ? h('span', { class: 'chip accent', style: { fontSize: '10px', padding: '0 6px' } }, 'opened the case') : null,
        quarantine ? h('span', { class: 'chip bad', style: { fontSize: '10px', padding: '0 6px' } }, 'quarantined') : null));
      el.append(h('div', { class: 'm-t' }, m.text || ''));
      const chips = [];
      if (rd.category) chips.push(h('span', { class: `chip ${rd.category === 'leak' ? 'accent' : ''}` }, rd.category));
      for (const sy of rd.symptoms || []) chips.push(h('span', { class: 'chip' }, SYM[sy] || sy));
      for (const p of rd.places || []) chips.push(h('span', { class: 'chip mono' }, `📍 ${p.text}`));
      if (rd.source && rd.source !== 'unknown') chips.push(h('span', { class: 'chip' }, `source: ${rd.source}`));
      if (chips.length) el.append(h('div', { class: 'm-r' }, chips));
      if (s.eps != null) {
        const cred = 1 - s.eps;
        el.append(h('div', { class: 'cred' }, h('span', {}, 'credible'), h('div', { class: 'bar' }, h('i', { style: { width: `${100 * cred}%`, background: cred > 0.5 ? 'var(--field)' : cred > 0.2 ? 'var(--warn)' : 'var(--faint)' } })), h('span', { class: 'mono' }, pct(cred))));
      } else if (!quarantine) el.append(h('div', { class: 'cred' }, h('span', {}, rd.category && rd.category !== 'leak' && rd.category !== 'quality' ? `${rd.category}: not leak evidence` : 'no usable place / not leak evidence')));
      const note = (s.notes || []).slice(-1)[0];
      if (note) el.append(h('div', { class: 'note' }, note));
      if (this.sel === m.rid && m.agent_view) el.append(h('div', { class: 'small mono', style: { marginTop: '6px', color: 'var(--text-2)' } }, h('span', { class: 'muted' }, 'agents read: '), m.agent_view));
      if (T && T.messages && T.messages[m.rid]) {
        const t = T.messages[m.rid];
        const real = t.kind === 'leak';
        el.append(h('div', { class: 'truth', style: { color: real ? 'var(--good)' : 'var(--faint)' } }, real ? '● truth: a real sign of the leak' : `○ truth: ${t.kind === 'false' ? 'false alarm — ' + (CAUSE[t.cause] || t.cause || 'other') : t.kind || 'not about a leak'}`));
      }
      el.addEventListener('click', () => this.select(m.rid));
      box.append(el);
    }
  }

  renderRoster(i) {
    const box = this.o.rosterEl, evs = this.events, cur = evs[i] || {};
    const counts = {};
    for (let k = 0; k <= i; k++) if (evs[k].kind === 'turn') counts[evs[k].role] = (counts[evs[k].role] || 0) + 1;
    const active = cur.role || (cur.kind === 'tool' && this.ctx.openTurn ? this.ctx.openTurn.role : null);
    for (const a of box.querySelectorAll('.agent')) {
      const role = a.dataset.role;
      a.classList.toggle('active', role === active);
      a.classList.toggle('thinking', cur.kind === 'thinking' && cur.role === role);
      const c = a.querySelector('.cnt'); if (c) c.textContent = `${counts[role] || 0} turn${(counts[role] || 0) === 1 ? '' : 's'}`;
    }
  }
}

export function rosterEl(roles = ['forensic', 'skeptic', 'strategist', 'auditor']) {
  return h('div', { class: 'roster' }, roles.map(r => {
    const R = ROLES[r];
    return h('div', { class: 'agent', 'data-role': r, title: R.job, style: { color: R.hex } },
      h('div', { class: 'av', style: { background: R.hex }, html: icon(R.icon, 18) }), h('div', { class: 'nm' }, R.short), h('div', { class: 'cnt' }, '0 turns'));
  }));
}

export { clockLabel, num };
