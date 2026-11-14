// The L-Town network on a canvas: pipes, step-test sections, the leak map (heat), hypotheses, complaints, sensors, field checks, digs, truth.
import { h, esc, themeColors, pct } from './ui.js';

const RAMP = [[0, [37, 99, 235]], [0.3, [14, 165, 233]], [0.5, [34, 211, 238]], [0.7, [253, 224, 71]], [0.85, [251, 146, 60]], [1, [239, 68, 68]]];
function ramp(t) {
  t = Math.max(0, Math.min(1, t));
  for (let i = 1; i < RAMP.length; i++) {
    if (t <= RAMP[i][0]) {
      const [t0, c0] = RAMP[i - 1], [t1, c1] = RAMP[i];
      const u = (t - t0) / (t1 - t0);
      return c0.map((v, k) => Math.round(v + u * (c1[k] - v)));
    }
  }
  return RAMP[RAMP.length - 1][1];
}
const SPRITES = [];
function sprite(level) {          // radial gradient sprites, 12 colour levels
  if (!SPRITES.length) {
    for (let i = 0; i < 12; i++) {
      const c = document.createElement('canvas');
      c.width = c.height = 96;
      const g = c.getContext('2d');
      const [r, gg, b] = ramp(i / 11);
      const gr = g.createRadialGradient(48, 48, 0, 48, 48, 48);
      gr.addColorStop(0, `rgba(${r},${gg},${b},0.95)`);
      gr.addColorStop(0.35, `rgba(${r},${gg},${b},0.55)`);
      gr.addColorStop(1, `rgba(${r},${gg},${b},0)`);
      g.fillStyle = gr;
      g.fillRect(0, 0, 96, 96);
      SPRITES.push(c);
    }
  }
  return SPRITES[Math.max(0, Math.min(11, Math.round(level * 11)))];
}

function hull(pts) {               // Andrew's monotone chain, pts [[x,y]]
  if (pts.length < 3) return pts.slice();
  const p = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  up.pop(); lo.pop();
  return lo.concat(up);
}

const SEC_HUES = [205, 265, 160, 30, 330, 190, 95, 285, 50, 175, 230, 0, 135, 305, 70];

export class NetworkMap {
  constructor(container, net, opts = {}) {
    this.net = net;
    this.opts = { interactive: true, labels: true, compact: false, sensorsAll: true, ...opts };
    this.box = container;
    this.canvas = h('canvas', { class: 'netmap' });
    this.tip = h('div', { class: 'tooltip' });
    container.append(this.canvas, this.tip);
    this.ctx = this.canvas.getContext('2d');
    this.layers = { pipes: true, sections: true, heat: true, hyp: true, cand: true, msgs: true, sensors: true, checks: true, landmarks: true, truth: true };
    this.s = {};
    this.view = { k: 1, tx: 0, ty: 0 };
    this.anims = [];
    this.dirty = true;
    this.hover = -1;
    this.t0 = performance.now();
    this._prep();
    this.colors = themeColors();
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(container);
    if (this.opts.interactive) this._bind();
    this.alive = true;
    this._loop = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  // ------------------------------------------------------------------ geometry
  _prep() {
    const N = this.net.nodes.length, J = this.net.n_junctions;
    this.X = new Float32Array(N); this.Y = new Float32Array(N);
    this.net.nodes.forEach((n, i) => { this.X[i] = n[0]; this.Y[i] = n[1]; });
    this.SX = new Float32Array(N); this.SY = new Float32Array(N);
    const [x0, y0, x1, y1] = this.net.bounds;
    this.bx = [x0, y0, x1, y1];
    this.secOf = this.net.section_of;
    this.areaLinks = {};                     // section / part id -> link indices inside it
    const memberOf = {};
    for (const s of this.net.sections) {
      memberOf[s.id] = new Set(s.j);
      for (const p of s.parts || []) memberOf[p.id] = new Set(p.j);
    }
    this.areaMembers = memberOf;
    for (const [aid, set] of Object.entries(memberOf)) {
      const ls = [];
      this.net.links.forEach((l, i) => { if (l[0] < J && l[1] < J && set.has(l[0]) && set.has(l[1])) ls.push(i); });
      this.areaLinks[aid] = ls;
    }
    this.secCentroid = {};
    for (const s of this.net.sections) this.secCentroid[s.id] = s.c;
    for (const s of this.net.sections) for (const p of s.parts || []) {
      let cx = 0, cy = 0; p.j.forEach(j => { cx += this.X[j]; cy += this.Y[j]; });
      this.secCentroid[p.id] = [cx / p.j.length, cy / p.j.length];
    }
    this.sensorSet = new Set(this.net.sensor_candidates);
  }

  resize() {
    const r = this.box.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.W = Math.max(50, r.width); this.H = Math.max(50, r.height);
    this.canvas.width = Math.round(this.W * dpr); this.canvas.height = Math.round(this.H * dpr);
    this.dpr = dpr;
    const [x0, y0, x1, y1] = this.bx;
    const pad = this.opts.compact ? 14 : 34;
    const pt = this.opts.padTop ?? pad, pb = this.opts.padBottom ?? pad;
    this.base = Math.min((this.W - 2 * pad) / (x1 - x0), (this.H - pt - pb) / (y1 - y0));
    this.ox = (this.W - (x1 - x0) * this.base) / 2;
    this.oy = pb + (this.H - pt - pb - (y1 - y0) * this.base) / 2;
    this.dirty = true;
  }

  _project() {
    const { k, tx, ty } = this.view, b = this.base * k, [x0, y0] = this.bx;
    const cx = this.W / 2, cy = this.H / 2;
    for (let i = 0; i < this.X.length; i++) {
      const sx = this.ox + (this.X[i] - x0) * this.base;
      const sy = this.H - (this.oy + (this.Y[i] - y0) * this.base);
      this.SX[i] = (sx - cx) * k + cx + tx;
      this.SY[i] = (sy - cy) * k + cy + ty;
    }
    this.mpp = 1 / b;                          // metres per pixel
  }
  w2s(x, y) {
    const { k, tx, ty } = this.view, [x0, y0] = this.bx, cx = this.W / 2, cy = this.H / 2;
    const sx = this.ox + (x - x0) * this.base, sy = this.H - (this.oy + (y - y0) * this.base);
    return [(sx - cx) * k + cx + tx, (sy - cy) * k + cy + ty];
  }
  px(m) { return m / this.mpp; }

  // ------------------------------------------------------------------ interaction
  _bind() {
    const c = this.canvas;
    c.addEventListener('wheel', e => {
      e.preventDefault();
      const r = c.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      const k0 = this.view.k, k1 = Math.max(0.7, Math.min(9, k0 * Math.exp(-e.deltaY * 0.0016)));
      const cx = this.W / 2, cy = this.H / 2;
      // keep the point under the cursor fixed
      this.view.tx = mx - (mx - cx - this.view.tx) * (k1 / k0) - cx;
      this.view.ty = my - (my - cy - this.view.ty) * (k1 / k0) - cy;
      this.view.k = k1;
      this.dirty = true;
    }, { passive: false });
    let drag = null;
    c.addEventListener('mousedown', e => { drag = { x: e.clientX, y: e.clientY, tx: this.view.tx, ty: this.view.ty, moved: false }; c.classList.add('dragging'); });
    window.addEventListener('mouseup', e => {
      if (drag && !drag.moved && this.opts.onClick && this.hover >= 0) this.opts.onClick(this.hover, e);
      drag = null; c.classList.remove('dragging');
    });
    window.addEventListener('mousemove', e => {
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
        this.view.tx = drag.tx + dx; this.view.ty = drag.ty + dy; this.dirty = true;
      }
    });
    c.addEventListener('mousemove', e => this._hover(e));
    c.addEventListener('mouseleave', () => { this.hover = -1; this.tip.style.display = 'none'; this.dirty = true; });
    c.addEventListener('dblclick', () => this.reset());
  }

  _hover(e) {
    const r = this.canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    let best = -1, bd = 14 * 14;
    for (let j = 0; j < this.net.n_junctions; j++) {
      const dx = this.SX[j] - mx, dy = this.SY[j] - my, d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = j; }
    }
    if (best !== this.hover) { this.hover = best; this.dirty = true; }
    if (best < 0) { this.tip.style.display = 'none'; return; }
    const s = this.s, n = this.net;
    const heat = s.heat ? s.heat[best] / 10 : null;
    const sec = n.sections[n.section_of[best]];
    const hyps = (s.slate || []).filter(x => x.region && x.region.includes(best)).map(x => x.id);
    const msgs = Object.entries(s.msgs || {}).filter(([, m]) => m.j === best).map(([rid]) => rid);
    const rows = [['junction', n.junction_ids[best]], ['elevation', `${n.nodes[best][3].toFixed(1)} m`], ['zone · section', `${n.zone[best]} · ${sec ? sec.id : '-'}`]];
    if (heat != null) rows.push(['leak map (relative)', `${heat.toFixed(1)}%`]);
    if (hyps.length) rows.push(['in hypotheses', hyps.join(', ')]);
    if (msgs.length) rows.push(['complaints here', msgs.join(', ')]);
    if (this.sensorSet.has(best)) rows.push(['sensor place', (s.sensors || []).includes(best) ? 'active sensor' : 'not used in this layout']);
    this.tip.innerHTML = `<div class="tt-h">${esc(n.describe[best])}</div>` + rows.map(([a, b]) => `<div class="tt-r"><span>${esc(a)}</span><span>${esc(b)}</span></div>`).join('');
    this.tip.style.display = 'block';
    const tw = this.tip.offsetWidth, th = this.tip.offsetHeight;
    this.tip.style.left = `${Math.min(mx + 16, this.W - tw - 8)}px`;
    this.tip.style.top = `${Math.min(my + 16, this.H - th - 8)}px`;
  }

  reset() { this.flyTo(null); }

  flyTo(j, k = 2.6, ms = 700) {
    const from = { ...this.view };
    let to = { k: 1, tx: 0, ty: 0 };
    if (j != null && j >= 0) {
      const [x0, y0] = this.bx, cx = this.W / 2, cy = this.H / 2;
      const sx = this.ox + (this.X[j] - x0) * this.base, sy = this.H - (this.oy + (this.Y[j] - y0) * this.base);
      to = { k, tx: -(sx - cx) * k, ty: -(sy - cy) * k };
    }
    const t0 = performance.now();
    this._fly = t => {
      const u = Math.min(1, (t - t0) / ms), e = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2;
      this.view = { k: from.k + (to.k - from.k) * e, tx: from.tx + (to.tx - from.tx) * e, ty: from.ty + (to.ty - from.ty) * e };
      this.dirty = true;
      if (u >= 1) this._fly = null;
    };
  }

  ripple(j, color, maxM = 260, ms = 1400) {
    if (j == null || j < 0) return;
    this.anims.push({ j, color, maxM, t0: performance.now(), ms });
  }

  setState(s) { this.s = { ...this.s, ...s }; this._hullKey = null; this.dirty = true; }
  setTheme() { this.colors = themeColors(); this.dirty = true; }
  setLayer(name, on) { this.layers[name] = on; this.dirty = true; }
  destroy() { this.alive = false; this._ro.disconnect(); }

  // ------------------------------------------------------------------ drawing
  _loop(t) {
    if (!this.alive) return;
    requestAnimationFrame(this._loop);
    if (!this.box.offsetParent) return;      // hidden view
    if (this._fly) this._fly(t);
    const animating = this.anims.length > 0 || this._pulsing;
    if (!this.dirty && !animating) return;
    if (animating && !this.dirty && t - (this._last || 0) < 33) return;
    this._last = t;
    this.dirty = false;
    this.draw(t);
  }

  draw(t = performance.now()) {
    const g = this.ctx, C = this.colors, L = this.layers, s = this.s, n = this.net, J = n.n_junctions;
    this._ff = this._ff || getComputedStyle(document.body).fontFamily;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.clearRect(0, 0, this.W, this.H);
    this._project();
    const light = C.light;
    const zoom = this.view.k;
    this._pulsing = false;
    const pulse = 0.5 + 0.5 * Math.sin((t - this.t0) / 380);

    // background
    g.fillStyle = light ? '#f8fafc' : '#040a15';
    g.fillRect(0, 0, this.W, this.H);
    if (!this.opts.compact) {
      g.strokeStyle = light ? 'rgba(15,23,42,0.045)' : 'rgba(120,160,220,0.05)';
      g.lineWidth = 1;
      const step = 40;
      g.beginPath();
      for (let x = (this.view.tx % step + step) % step; x < this.W; x += step) { g.moveTo(x, 0); g.lineTo(x, this.H); }
      for (let y = (this.view.ty % step + step) % step; y < this.H; y += step) { g.moveTo(0, y); g.lineTo(this.W, y); }
      g.stroke();
    }

    // which areas were step-tested
    const lit = new Set(), cleared = new Set();
    for (const c of s.checks || []) if (c.kind === 'step' && c.sid) (c.lit ? lit : cleared).add(c.sid);

    // pipes
    if (L.pipes) {
      const lw = d => Math.max(0.7, Math.min(3.2, 0.5 + Math.log2(1 + d / 60))) * (zoom > 2 ? 1.25 : 1);
      g.lineCap = 'round';
      n.links.forEach((l, i) => {
        const [a, b, kind, d] = l;
        let col;
        if (L.sections && a < J) {
          const k = this.secOf[a];
          const sid = n.sections[k] && n.sections[k].id;
          const hue = SEC_HUES[k % SEC_HUES.length];
          col = light ? `hsla(${hue},45%,38%,0.42)` : `hsla(${hue},55%,66%,0.40)`;
          if (cleared.has(sid)) col = light ? 'rgba(100,116,139,0.28)' : 'rgba(100,116,139,0.26)';
        } else col = light ? 'rgba(30,41,59,0.35)' : 'rgba(125,160,210,0.38)';
        if (kind === 'U') col = C.accent;
        g.strokeStyle = col;
        g.lineWidth = lw(d);
        g.beginPath(); g.moveTo(this.SX[a], this.SY[a]); g.lineTo(this.SX[b], this.SY[b]); g.stroke();
      });
    }

    // lit / cleared step-test areas (glow on their pipes)
    if (L.checks) {
      const drawArea = (aid, color, width, alpha, dash) => {
        const ls = this.areaLinks[aid] || [];
        g.save(); g.strokeStyle = color; g.globalAlpha = alpha; g.lineWidth = width; if (dash) g.setLineDash(dash);
        if (!light) { g.shadowColor = color; g.shadowBlur = 12; }
        g.beginPath();
        for (const i of ls) { const [a, b] = n.links[i]; g.moveTo(this.SX[a], this.SY[a]); g.lineTo(this.SX[b], this.SY[b]); }
        g.stroke(); g.restore();
      };
      for (const sid of cleared) if (!lit.has(sid)) drawArea(sid, light ? '#94a3b8' : '#64748b', 2.2, 0.55, [3, 4]);
      for (const sid of lit) drawArea(sid, C.field, 4.2, 0.85 + 0.15 * pulse);
      if (lit.size) this._pulsing = true;
      // options preview (strategist's menu)
      for (const o of s.options || []) {
        if (o.kind === 'step') for (const sid of o.sections || []) drawArea(sid, C.strategist, o.pick ? 3.6 : 2.2, o.pick ? 0.9 : 0.45, [6, 5]);
      }
    }

    // heat (the leak map)
    if (L.heat && s.heat) {
      const pl = s.p_leak == null ? 0.6 : s.p_leak;
      const rad = Math.max(9, this.px(75));
      g.save();
      g.globalCompositeOperation = light ? 'multiply' : 'lighter';
      for (let j = 0; j < J; j++) {
        const v = s.heat[j] / 1000;
        if (v < 0.015) continue;
        const I = Math.sqrt(v);
        const r = rad * (0.55 + 0.75 * I);
        g.globalAlpha = Math.min(1, (0.25 + 0.75 * I) * (0.4 + 0.6 * pl)) * (light ? 0.75 : 0.9);
        g.drawImage(sprite(I), this.SX[j] - r, this.SY[j] - r, 2 * r, 2 * r);
      }
      g.restore();
    }

    // hypothesised junctions (simulated on demand) and hypothesis regions
    if (L.cand && s.cand && s.cand.length) {
      g.fillStyle = light ? 'rgba(124,58,237,0.55)' : 'rgba(196,181,253,0.65)';
      const r = zoom > 2 ? 2.2 : 1.5;
      for (const j of s.cand) { g.beginPath(); g.arc(this.SX[j], this.SY[j], r, 0, 6.283); g.fill(); }
    }
    if (L.hyp && s.slate) {
      for (const hyp of s.slate) {
        if (!hyp.region || !hyp.region.length || hyp.kind !== 'network_leak') continue;
        const pts = hull(hyp.region.map(j => [this.SX[j], this.SY[j]]));
        if (pts.length < 2) continue;
        g.save();
        g.beginPath();
        pts.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
        g.closePath();
        g.fillStyle = light ? 'rgba(124,58,237,0.06)' : 'rgba(167,139,250,0.07)';
        g.fill();
        g.setLineDash([7, 5]); g.lineWidth = 1.6; g.strokeStyle = C.forensic; g.globalAlpha = 0.9; g.stroke();
        g.restore();
        const top = pts.reduce((a, b) => (b[1] < a[1] ? b : a));
        this._label(top[0], top[1] - 8, `${hyp.id}${hyp.mass != null ? ' · ' + pct(hyp.mass) : ''}`, C.forensic, true);
      }
    }

    // listening surveys and options
    if (L.checks) {
      for (const o of s.options || []) {
        if (o.kind !== 'listen' || o.center == null) continue;
        const r = this.px(250);
        g.save(); g.setLineDash([5, 5]); g.lineWidth = o.pick ? 2.6 : 1.4; g.strokeStyle = C.strategist; g.globalAlpha = o.pick ? 0.95 : 0.55;
        g.beginPath(); g.arc(this.SX[o.center], this.SY[o.center], r, 0, 6.283); g.stroke(); g.restore();
        this._label(this.SX[o.center], this.SY[o.center] - r - 6, `${o.id}${o.chance != null ? ' · ' + pct(o.chance) : ''}`, C.strategist, true);
      }
      for (const c of s.checks || []) {
        if ((c.kind === 'listen_quiet' || c.kind === 'listen_hit') && c.center != null) {
          const r = this.px(250), hit = c.kind === 'listen_hit';
          g.save();
          g.lineWidth = hit ? 2.4 : 1.5;
          g.strokeStyle = hit ? C.good : (light ? '#64748b' : '#94a3b8');
          g.fillStyle = hit ? (light ? 'rgba(5,150,105,0.07)' : 'rgba(52,211,153,0.07)') : (light ? 'rgba(100,116,139,0.05)' : 'rgba(148,163,184,0.05)');
          if (!hit) g.setLineDash([3, 4]);
          g.beginPath(); g.arc(this.SX[c.center], this.SY[c.center], r, 0, 6.283); g.fill(); g.stroke(); g.restore();
          if (!this.opts.compact) this._label(this.SX[c.center], this.SY[c.center] + r + 12, hit ? 'leak noise heard' : 'listening: quiet', hit ? C.good : C.muted, false);
          if (hit && c.at != null) {
            this._pulsing = true;
            const x = this.SX[c.at], y = this.SY[c.at];
            g.save(); g.strokeStyle = C.good; g.lineWidth = 2;
            g.beginPath(); g.arc(x, y, 7 + 5 * pulse, 0, 6.283); g.stroke();
            g.beginPath(); g.moveTo(x - 12, y); g.lineTo(x + 12, y); g.moveTo(x, y - 12); g.lineTo(x, y + 12); g.stroke(); g.restore();
          }
        }
      }
      if (!this.opts.compact) {
        const litParts = [...lit].filter(x => x.includes('.'));
        for (const sid of lit) {
          if (!sid.includes('.') && litParts.some(p => p.startsWith(sid + '.'))) continue;      // the lit part says more than its section
          const c = this.secCentroid[sid]; if (c) { const [x, y] = this.w2s(c[0], c[1]);
          this._label(x, y, `${sid} lit${s.stepVals && s.stepVals[sid] != null ? ' · +' + s.stepVals[sid].toFixed(1) + ' L/s' : ''}`, C.field, true); } }
        for (const sid of cleared) if (!lit.has(sid) && !sid.includes('.')) { const c = this.secCentroid[sid]; if (c) { const [x, y] = this.w2s(c[0], c[1]); this._label(x, y, `${sid} cleared`, C.muted, false); } }
      }
    }

    // section labels
    if (L.sections && this.opts.labels && !this.opts.compact) {
      g.font = `600 10.5px ${this._ff}`;
      g.textAlign = 'center';
      for (const sec of n.sections) {
        if (lit.has(sec.id) || cleared.has(sec.id)) continue;
        const [x, y] = this.w2s(sec.c[0], sec.c[1]);
        g.fillStyle = light ? 'rgba(51,65,85,0.55)' : 'rgba(180,196,221,0.42)';
        g.fillText(sec.id, x, y);
      }
    }

    // landmarks
    if (L.landmarks) {
      for (const lm of n.landmarks) {
        const x = this.SX[lm.j], y = this.SY[lm.j];
        const special = lm.sub === 'school' || lm.sub === 'hospital';
        if (!special && zoom < 1.7) continue;
        g.fillStyle = lm.sub === 'hospital' ? (light ? '#dc2626' : '#f87171') : lm.sub === 'school' ? (light ? '#b45309' : '#fcd34d') : (light ? '#64748b' : '#94a3b8');
        g.globalAlpha = special ? 0.9 : 0.6;
        if (lm.sub === 'hospital') { g.fillRect(x - 1.5, y - 5, 3, 10); g.fillRect(x - 5, y - 1.5, 10, 3); }
        else { g.beginPath(); g.arc(x, y, special ? 3.2 : 2.3, 0, 6.283); g.fill(); }
        g.globalAlpha = 1;
        if (zoom >= 2.2 && this.opts.labels) this._label(x, y - 9, lm.name, C.muted, false, 10);
      }
    }

    // triangulation: water runs down from a caller's place to another
    if (L.msgs && s.msgs) {
      for (const [rid, m] of Object.entries(s.msgs)) {
        if (m.up == null || m.j == null || m.up === m.j) continue;
        this._arrow(this.SX[m.up], this.SY[m.up], this.SX[m.j], this.SY[m.j], C.forensic);
      }
    }

    // sensors
    if (L.sensors) {
      const active = new Set(s.sensors || []);
      if (this.opts.sensorsAll) for (const j of n.sensor_candidates) {
        if (active.has(j)) continue;
        this._diamond(this.SX[j], this.SY[j], 3.5, null, light ? 'rgba(100,116,139,0.55)' : 'rgba(148,163,184,0.45)');
      }
      for (const j of active) {
        const z = (s.sensorZ || {})[j];
        const col = z == null ? C.accent : Math.abs(z) >= 3 ? C.bad : Math.abs(z) >= 1.5 ? C.warn : C.accent;
        if (z != null && Math.abs(z) >= 2) {
          this._pulsing = true;
          g.save(); g.globalAlpha = 0.35 * (1 - pulse * 0.6); g.fillStyle = col;
          g.beginPath(); g.arc(this.SX[j], this.SY[j], 9 + 7 * pulse, 0, 6.283); g.fill(); g.restore();
        }
        this._diamond(this.SX[j], this.SY[j], 6.5, col, light ? '#fff' : '#0b1426');
        if (this.opts.labels && !this.opts.compact) this._label(this.SX[j] + 10, this.SY[j] + 4, `${n.junction_ids[j]}${z != null ? ' · z ' + z.toFixed(1) : ''}`, col, false, 10, 'left');
      }
    }

    // complaints
    if (L.msgs && s.msgs) {
      const meta = s.msgMeta || {};
      for (const [rid, m] of Object.entries(s.msgs)) {
        if (m.j == null) continue;
        const mm = meta[rid] || {};
        const quarantine = mm.read && mm.read.contains_instructions;
        const debunked = m.eps != null && m.eps >= 0.85;
        let fill, ring = null, size = 6.5;
        if (quarantine) fill = C.bad;
        else if (!m.usable) { fill = null; size = 4; }
        else if (debunked) fill = light ? '#94a3b8' : '#64748b';
        else if (m.eps != null && m.eps < 0.6) fill = light ? '#ea580c' : '#fb923c';
        else fill = light ? '#d97706' : '#fcd34d';
        if (mm.trigger) ring = C.text;
        const x = this.SX[m.j], y = this.SY[m.j];
        const sel = s.selMsg === rid;
        this._pin(x, y, size * (sel ? 1.4 : 1), fill, ring, debunked, quarantine);
        if (sel || (this.opts.labels && zoom >= 1.9)) this._label(x, y - size * 2.4 - 6, rid, fill || C.muted, sel, 10);
      }
    }

    // best dig spot
    if (L.checks && s.best != null && (s.p_success || 0) >= 0.05 && !s.hideBest) {
      const x = this.SX[s.best], y = this.SY[s.best];
      g.save(); g.strokeStyle = C.text; g.lineWidth = 1.6; g.globalAlpha = 0.9;
      g.beginPath(); g.arc(x, y, 10, 0, 6.283); g.stroke();
      g.beginPath(); g.moveTo(x - 15, y); g.lineTo(x - 6, y); g.moveTo(x + 6, y); g.lineTo(x + 15, y); g.moveTo(x, y - 15); g.lineTo(x, y - 6); g.moveTo(x, y + 6); g.lineTo(x, y + 15); g.stroke();
      g.restore();
      if (!this.opts.compact) this._label(x, y + 26, `best dig spot · P(success) ${pct(s.p_success)}`, C.text, true);
    }

    // digs
    for (const d of s.digs || []) {
      if (d.j == null) continue;
      const x = this.SX[d.j], y = this.SY[d.j];
      if (d.found) {
        this._pulsing = true;
        g.save(); g.fillStyle = C.good; g.globalAlpha = 0.25 + 0.2 * pulse;
        g.beginPath(); g.arc(x, y, 16 + 6 * pulse, 0, 6.283); g.fill(); g.globalAlpha = 1;
        g.beginPath(); g.arc(x, y, 9, 0, 6.283); g.fill();
        g.strokeStyle = '#fff'; g.lineWidth = 2.4; g.beginPath(); g.moveTo(x - 4.5, y); g.lineTo(x - 1, y + 3.5); g.lineTo(x + 5, y - 3.5); g.stroke(); g.restore();
        this._label(x, y - 22, 'DUG · LEAK FOUND', C.good, true);
      } else {
        g.save(); g.fillStyle = light ? '#334155' : '#1e293b'; g.strokeStyle = light ? '#94a3b8' : '#94a3b8'; g.lineWidth = 2;
        g.beginPath(); g.arc(x, y, 9, 0, 6.283); g.fill(); g.stroke();
        g.strokeStyle = C.bad; g.beginPath(); g.moveTo(x - 4, y - 4); g.lineTo(x + 4, y + 4); g.moveTo(x + 4, y - 4); g.lineTo(x - 4, y + 4); g.stroke(); g.restore();
        this._label(x, y - 18, 'dry hole', C.muted, true);
      }
    }

    // truth
    if (L.truth && s.truth) {
      const T = s.truth;
      if (T.trap && T.trap.j != null) {
        const x = this.SX[T.trap.j], y = this.SY[T.trap.j];
        g.save(); g.strokeStyle = C.bad; g.lineWidth = 2.4; g.setLineDash([3, 3]);
        g.beginPath(); g.arc(x, y, 13, 0, 6.283); g.stroke(); g.setLineDash([]);
        g.beginPath(); g.moveTo(x - 7, y - 7); g.lineTo(x + 7, y + 7); g.moveTo(x + 7, y - 7); g.lineTo(x - 7, y + 7); g.stroke(); g.restore();
        this._label(x, y + 26, "the fake message's spot", C.bad, true);
      }
      if (T.leak) {
        let x = this.SX[T.leak.j], y = this.SY[T.leak.j];
        if (T.leak.a != null && T.leak.b != null && T.leak.frac != null) {
          x = this.SX[T.leak.a] + (this.SX[T.leak.b] - this.SX[T.leak.a]) * T.leak.frac;
          y = this.SY[T.leak.a] + (this.SY[T.leak.b] - this.SY[T.leak.a]) * T.leak.frac;
        }
        if (s.finalJ != null) {
          const fx = this.SX[s.finalJ], fy = this.SY[s.finalJ];
          g.save(); g.strokeStyle = C.leak; g.setLineDash([4, 4]); g.lineWidth = 1.6; g.beginPath(); g.moveTo(fx, fy); g.lineTo(x, y); g.stroke(); g.restore();
          if (s.errM != null) this._label((fx + x) / 2 + 10, (fy + y) / 2, `${Math.round(s.errM)} m`, C.leak, true, 11, 'left');
        }
        this._pulsing = true;
        g.save(); g.fillStyle = C.leak; g.globalAlpha = 0.18 + 0.22 * pulse;
        g.beginPath(); g.arc(x, y, 18 + 10 * pulse, 0, 6.283); g.fill(); g.restore();
        this._star(x, y, 9, C.leak);
        this._label(x, y + 24, `TRUE LEAK · ${T.leak.lps} L/s`, C.leak, true);
      }
    }

    // animations
    const now = performance.now();
    this.anims = this.anims.filter(a => now - a.t0 < a.ms);
    for (const a of this.anims) {
      const u = (now - a.t0) / a.ms;
      g.save(); g.strokeStyle = a.color; g.lineWidth = 2.5 * (1 - u); g.globalAlpha = 1 - u;
      g.beginPath(); g.arc(this.SX[a.j], this.SY[a.j], this.px(a.maxM) * (0.15 + 0.85 * u), 0, 6.283); g.stroke(); g.restore();
    }

    // hover
    if (this.hover >= 0) {
      g.save(); g.strokeStyle = C.accent; g.lineWidth = 2;
      g.beginPath(); g.arc(this.SX[this.hover], this.SY[this.hover], 6, 0, 6.283); g.stroke(); g.restore();
    }
  }

  // ------------------------------------------------------------------ marks
  _label(x, y, text, color, strong = false, size = 11, align = 'center') {
    const g = this.ctx, C = this.colors;
    g.save();
    g.font = `${strong ? 700 : 600} ${size}px ${this._ff || 'sans-serif'}`;
    g.textAlign = align; g.textBaseline = 'middle';
    const w = g.measureText(text).width;
    const bx = align === 'center' ? x - w / 2 - 5 : x - 4;
    g.fillStyle = C.light ? 'rgba(255,255,255,0.86)' : 'rgba(6,12,24,0.78)';
    g.beginPath(); g.roundRect ? g.roundRect(bx, y - size / 2 - 4, w + 10, size + 8, 6) : g.rect(bx, y - size / 2 - 4, w + 10, size + 8); g.fill();
    g.fillStyle = color; g.fillText(text, x, y + 0.5);
    g.restore();
  }
  _diamond(x, y, r, fill, stroke) {
    const g = this.ctx;
    g.beginPath(); g.moveTo(x, y - r); g.lineTo(x + r, y); g.lineTo(x, y + r); g.lineTo(x - r, y); g.closePath();
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = fill ? 1.6 : 1.2; g.stroke(); }
  }
  _pin(x, y, r, fill, ring, debunked, quarantine) {
    const g = this.ctx, C = this.colors;
    g.save();
    const top = y - r * 2.2;
    g.beginPath();
    g.moveTo(x, y);
    g.bezierCurveTo(x - r * 1.3, top + r * 0.9, x - r, top - r * 0.2, x, top - r * 0.2);
    g.bezierCurveTo(x + r, top - r * 0.2, x + r * 1.3, top + r * 0.9, x, y);
    if (fill) { g.fillStyle = fill; g.globalAlpha = debunked ? 0.7 : 1; g.fill(); g.globalAlpha = 1; g.strokeStyle = C.light ? '#fff' : '#0b1426'; g.lineWidth = 1.4; g.stroke(); }
    else { g.strokeStyle = C.light ? '#94a3b8' : '#64748b'; g.lineWidth = 1.3; g.stroke(); }
    if (ring) { g.strokeStyle = ring; g.lineWidth = 1.6; g.beginPath(); g.arc(x, top + r * 0.55, r * 1.45, 0, 6.283); g.stroke(); }
    if (debunked || quarantine) {
      g.strokeStyle = '#fff'; g.lineWidth = 1.6; const cy = top + r * 0.55, d = r * 0.42;
      g.beginPath();
      if (quarantine) { g.moveTo(x, cy - d * 1.2); g.lineTo(x, cy + d * 0.3); g.moveTo(x, cy + d * 1); g.lineTo(x, cy + d * 1.1); }
      else { g.moveTo(x - d, cy - d); g.lineTo(x + d, cy + d); g.moveTo(x + d, cy - d); g.lineTo(x - d, cy + d); }
      g.stroke();
    }
    g.restore();
  }
  _star(x, y, r, color) {
    const g = this.ctx;
    g.save(); g.beginPath();
    for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r; g.lineTo(x + rr * Math.cos(a), y + rr * Math.sin(a)); }
    g.closePath(); g.fillStyle = color; g.fill(); g.strokeStyle = '#fff'; g.lineWidth = 1.3; g.stroke(); g.restore();
  }
  _arrow(x0, y0, x1, y1, color) {
    const g = this.ctx, a = Math.atan2(y1 - y0, x1 - x0), L = Math.hypot(x1 - x0, y1 - y0);
    if (L < 6) return;
    g.save(); g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 1.8; g.setLineDash([5, 4]); g.globalAlpha = 0.9;
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1 - 9 * Math.cos(a), y1 - 9 * Math.sin(a)); g.stroke(); g.setLineDash([]);
    g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - 10 * Math.cos(a - 0.4), y1 - 10 * Math.sin(a - 0.4)); g.lineTo(x1 - 10 * Math.cos(a + 0.4), y1 - 10 * Math.sin(a + 0.4)); g.closePath(); g.fill();
    g.restore();
  }
}
