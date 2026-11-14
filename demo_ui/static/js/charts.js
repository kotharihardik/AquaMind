// Small SVG charts (no library): horizontal bars with intervals, grouped bars, diverging bars, line chart.
import { esc } from './ui.js';

const svg = (w, hgt, body) => `<svg class="chart" viewBox="0 0 ${w} ${hgt}" preserveAspectRatio="xMidYMid meet">${body}</svg>`;

export function hbars(rows, { max = 100, unit = '%', labelW = 210, W = 640, bar = 22, gap = 12, fmt = v => `${v}${unit}` } = {}) {
  // rows: [{label, sub, value, lo, hi, color, bold}]
  const H = rows.length * (bar + gap) + 26;
  const x0 = labelW, x1 = W - 70, X = v => x0 + (Math.max(0, v) / max) * (x1 - x0);
  let s = '';
  const div = max % 4 === 0 ? 4 : 5;
  for (const t of Array.from({ length: div + 1 }, (_, k) => k / div)) {
    const x = x0 + t * (x1 - x0);
    s += `<line class="grid" x1="${x}" x2="${x}" y1="0" y2="${H - 20}"/><text x="${x}" y="${H - 6}" text-anchor="middle">${Math.round(t * max)}${unit}</text>`;
  }
  rows.forEach((r, i) => {
    const y = i * (bar + gap) + 4;
    s += `<text class="lab" x="${labelW - 10}" y="${y + bar / 2 + (r.sub ? -2 : 4)}" text-anchor="end" font-weight="${r.bold ? 800 : 600}">${esc(r.label)}</text>`;
    if (r.sub) s += `<text x="${labelW - 10}" y="${y + bar / 2 + 12}" text-anchor="end" font-size="10.5">${esc(r.sub)}</text>`;
    s += `<rect x="${x0}" y="${y}" width="${Math.max(2, X(r.value) - x0)}" height="${bar}" rx="6" fill="${r.color || 'var(--accent)'}" opacity="${r.bold ? 1 : 0.8}"/>`;
    if (r.lo != null && r.hi != null) {
      s += `<line x1="${X(r.lo)}" x2="${X(r.hi)}" y1="${y + bar / 2}" y2="${y + bar / 2}" stroke="var(--text)" stroke-width="1.5" opacity=".7"/>`;
      s += `<line x1="${X(r.lo)}" x2="${X(r.lo)}" y1="${y + 6}" y2="${y + bar - 6}" stroke="var(--text)" stroke-width="1.5" opacity=".7"/><line x1="${X(r.hi)}" x2="${X(r.hi)}" y1="${y + 6}" y2="${y + bar - 6}" stroke="var(--text)" stroke-width="1.5" opacity=".7"/>`;
    }
    s += `<text class="val" x="${Math.max(X(r.value), r.hi != null ? X(r.hi) : 0) + 8}" y="${y + bar / 2 + 4}">${esc(fmt(r.value))}</text>`;
  });
  return svg(W, H, s);
}

export function grouped(groups, series, { max = 100, unit = '%', W = 640, H = 260, barMax = 34, fmt = v => `${Math.round(v)}${unit}` } = {}) {
  // groups: ['3 sensors', ...]; series: [{name, color, values: [..]}]
  const L = 40, R = 10, T = 16, B = 46;
  const gw = (W - L - R) / groups.length, bw = Math.min(barMax, (gw - 18) / series.length);
  const Y = v => T + (1 - Math.max(0, v) / max) * (H - T - B);
  let s = '';
  for (const t of [0, 0.25, 0.5, 0.75, 1]) s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(t * max)}" y2="${Y(t * max)}"/><text x="${L - 6}" y="${Y(t * max) + 4}" text-anchor="end">${Math.round(t * max)}${unit}</text>`;
  groups.forEach((g, gi) => {
    const gx = L + gi * gw + (gw - bw * series.length) / 2;
    series.forEach((se, si) => {
      const v = se.values[gi]; if (v == null) return;
      const x = gx + si * bw;
      s += `<rect x="${x + 2}" y="${Y(v)}" width="${bw - 4}" height="${H - B - Y(v)}" rx="4" fill="${se.color}" opacity="${se.bold ? 1 : 0.78}"/>`;
      s += `<text class="val" x="${x + bw / 2}" y="${Y(v) - 5}" text-anchor="middle" font-size="10.5">${esc(fmt(v))}</text>`;
    });
    s += `<text class="lab" x="${L + gi * gw + gw / 2}" y="${H - B + 18}" text-anchor="middle">${esc(g)}</text>`;
  });
  // legend (wraps onto a second row)
  let lx = L, ly = H - 16;
  const H2 = H + 18;
  series.forEach(se => {
    const w = 22 + se.name.length * 6.2;
    if (lx + w > W - R) { lx = L; ly += 18; }
    s += `<rect x="${lx}" y="${ly}" width="10" height="10" rx="2" fill="${se.color}"/><text x="${lx + 14}" y="${ly + 9}">${esc(se.name)}</text>`; lx += w;
  });
  return svg(W, Math.max(H, ly + 14 > H ? H2 : H), s);
}

export function diverging(rows, { W = 640, labelW = 220, bar = 20, gap = 10, span = 70, unit = ' pts' } = {}) {
  // rows: [{label, sub, delta}] delta in percentage points (negative = worse)
  const H = rows.length * (bar + gap) + 24, mid = labelW + (W - labelW - 60) * 0.85, X = d => mid + (d / span) * (W - labelW - 60) * 0.85;
  let s = `<line class="axis" x1="${mid}" x2="${mid}" y1="0" y2="${H - 18}"/>`;
  const stepT = span > 40 ? 20 : span > 20 ? 10 : 5;
  for (let t = 0; t >= -span * 0.95; t -= stepT) s += `<text x="${X(t)}" y="${H - 4}" text-anchor="middle">${t}</text>`;
  rows.forEach((r, i) => {
    const y = i * (bar + gap) + 2, d = r.delta;
    s += `<text class="lab" x="${labelW - 10}" y="${y + bar / 2 + (r.sub ? -2 : 4)}" text-anchor="end">${esc(r.label)}</text>`;
    if (r.sub) s += `<text x="${labelW - 10}" y="${y + bar / 2 + 11}" text-anchor="end" font-size="10.5">${esc(r.sub)}</text>`;
    const x = Math.min(X(d), mid), w = Math.abs(X(d) - mid);
    s += `<rect x="${x}" y="${y}" width="${Math.max(2, w)}" height="${bar}" rx="5" fill="${d < -15 ? 'var(--bad)' : d < -3 ? 'var(--warn)' : d > 3 ? 'var(--good)' : 'var(--faint)'}"/>`;
    s += `<text class="val" x="${Math.max(mid, X(d)) + 8}" y="${y + bar / 2 + 4}" text-anchor="start">${d > 0 ? '+' : ''}${d.toFixed(1)}${unit}</text>`;
  });
  return svg(W, H, s);
}

export function line(series, { W = 640, H = 220, xLabel = '', yLabel = '' } = {}) {
  // series: [{name, color, pts: [[x,y]], dash}]
  const all = series.flatMap(s => s.pts);
  if (!all.length) return svg(W, H, '');
  const xs = all.map(p => p[0]), ys = all.map(p => p[1]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = 0, y1 = Math.max(...ys) * 1.08;
  const L = 44, R = 12, T = 12, B = 34;
  const X = x => L + (x - x0) / Math.max(1e-9, x1 - x0) * (W - L - R), Y = y => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  let s = '';
  for (const t of [0, 0.5, 1]) { const v = y0 + t * (y1 - y0); s += `<line class="grid" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text x="${L - 6}" y="${Y(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`; }
  for (const se of series) {
    const d = se.pts.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="2.4" ${se.dash ? 'stroke-dasharray="5 4"' : ''}/>`;
    for (const p of se.pts) if (se.dots) s += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="3.5" fill="${se.color}"/>`;
  }
  s += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">${esc(xLabel)}</text>`;
  let lx = L + 10;
  for (const se of series) { s += `<rect x="${lx}" y="${T}" width="12" height="3" fill="${se.color}"/><text x="${lx + 16}" y="${T + 5}">${esc(se.name)}</text>`; lx += 30 + se.name.length * 6.5; }
  return svg(W, H, s);
}
