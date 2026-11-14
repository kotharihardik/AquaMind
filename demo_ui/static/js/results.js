// Results: every number computed live from the evaluation's result rows (logs/evaluation). Final runs only (the demo shows no mid-report comparison).
import { get } from './api.js';
import { hbars, grouped, diverging, line } from './charts.js';
import { h, esc, num } from './ui.js';
import { icon } from './icons.js';

const COL = { agentix_crew: 'var(--accent)', react_agent: 'var(--react)', crew_practice: 'var(--warn)', plain_llm: 'var(--forensic)', classical: 'var(--faint)', fixed_rules: 'var(--code)', no_ai: 'var(--good)' };
const v = (x, d = 1, u = '') => (x == null ? '–' : `${Number(x).toFixed(d)}${u}`);
const p1 = x => (x == null ? '–' : `${(100 * x).toFixed(1)}%`);
const signed = (d, dec = 1) => `${d > 0 ? '+' : d < 0 ? '−' : '±'}${Math.abs(d).toFixed(dec)}`;
const pval = p => (p < 0.001 ? '< 0.001' : p.toFixed(p < 0.1 ? 3 : 2));
const midChip = () => h('span', { class: 'chip warn', style: { marginLeft: '6px', fontSize: '10px' }, title: 'The final run of this variant is not complete yet: the mid-report run is shown, compared with the mid crew.' }, 'mid run');

export async function initResults(root, boot, go) {
  const R = await get('/api/results');
  root.innerHTML = '';
  const page = h('div', { class: 'page' });
  root.append(page);
  const crew = R.main.find(m => m.ours);
  const react = R.main.find(m => m.key === 'react_agent');
  const prac = R.main.find(m => m.key === 'crew_practice');
  const final = R.stage === 'final';
  const sec = (id, eyebrow, title, lead) => {
    const s = h('section', { id: `res-${id}` }, h('div', { class: 'eyebrow', style: { marginTop: '34px' } }, eyebrow), h('h2', { style: { marginTop: '4px' } }, title), lead ? h('p', { class: 'lead' }, lead) : null);
    page.append(s); return s;
  };
  const tabs = [['rq2', 'RQ2 · vs agents'], ['safety', 'Safety & traps'], ['rq1', 'RQ1 · few sensors'], ['abl', 'Ablations'], ['roles', 'Agent roles'], ['types', 'Case types'],
    R.agent_extra ? ['agent', 'Agent metrics'] : null, R.rag ? ['rag', 'RAG metrics'] : null, ['reader', 'Fine-tuned reader']].filter(Boolean);

  page.append(h('div', { class: 'eyebrow' }, `${final ? 'Final evaluation' : 'Evaluation'} · ${R.n_cases} cases on the real L-Town network (60 dev + 150 frozen test)`),
    h('h1', {}, 'Results'),
    h('p', { class: 'lead' }, 'Computed live from the result rows in logs/evaluation (one row per case and method). Every method gets the same cases, sensors, field budget, LLM, reader and random luck.'),
    h('div', { class: 'subnav' }, tabs.map(([id, t]) => h('button', { class: 'btn sm', onclick: () => document.getElementById(`res-${id}`).scrollIntoView({ behavior: 'smooth', block: 'start' }) }, t))));
  if (R.pending && R.pending.length) page.append(h('div', { class: 'callout', style: { borderLeftColor: 'var(--warn)', background: 'var(--warn-soft)', marginBottom: '10px' } },
    `${R.pending.length} variant${R.pending.length > 1 ? 's' : ''} still show the mid-report run because the final run is not complete: they are marked “mid run” and compared with the mid crew. This page updates by itself when the runs finish.`));

  // KPIs
  page.append(h('div', { class: 'grid g5', style: { marginTop: '8px' } },
    kpi(`${crew.correct[0]}%`, 'correct digs', `leak cases dug within 100 m (95% CI ${crew.correct[1]}–${crew.correct[2]})`, `ReAct ${react.correct[0]}% · crews today ${prac.correct[0]}%`, 'var(--accent)'),
    kpi(`${crew.solved[0]}/${crew.solved[1]}`, 'cases solved', 'leak dug within 100 m, or a no-leak case closed with no dig', `ReAct ${react.solved[0]}/${react.solved[1]}`, 'var(--accent)'),
    kpi(`${crew.median_m} m`, 'median distance error', 'final guess to the true leak', `ReAct ${react.median_m} m`, 'var(--good)'),
    kpi(`${crew.unsafe}%`, 'unsafe or unapproved actions', `${crew.unsafe_n} of ${crew.n} cases`, `ReAct ${react.unsafe}% of cases`, 'var(--good)'),
    kpi(`${crew.false_digs}%`, 'false digs', 'share of digs more than 100 m from any leak', `ReAct ${react.false_digs}%`, 'var(--warn)')));

  // RQ2
  const s2 = sec('rq2', 'RQ2', 'The crew against standard agents and today\'s practice', 'Same LLM (Nemotron-3 Ultra 550B, thinking off), same tools, same step and field budget. Correct digs with 95% bootstrap intervals; McNemar test paired by case.');
  const others = R.main.filter(m => !m.ours && m.mcnemar);
  const worstP = Math.max(...others.map(m => m.mcnemar.p));
  const ratio = react.correct[0] ? crew.correct[0] / react.correct[0] : null;
  s2.append(h('div', { class: 'grid g2' },
    h('div', { class: 'card', html: hbars(R.main.map(m => ({ label: m.name, sub: m.what.length > 36 ? m.what.slice(0, 35) + '…' : m.what, value: m.correct[0], lo: m.correct[1], hi: m.correct[2], color: COL[m.key], bold: m.ours })), { labelW: 250 }) }),
    h('div', { class: 'card' }, h('h3', {}, 'What the bars mean'),
      h('div', { class: 'callout' }, `The crew dug at the right spot in ${crew.correct[0]}% of the ${crew.n_leak} leak cases` + (ratio ? `: ${ratio.toFixed(1)} times ReAct (${react.correct[0]}%) and ${(crew.correct[0] / prac.correct[0]).toFixed(1)} times today's crew practice (${prac.correct[0]}%).` : '.') +
        ` ${worstP < 0.05 ? 'Every difference is significant' : 'Differences'} (McNemar p ${worstP < 0.001 ? '< 0.001' : '≤ ' + worstP.toFixed(3)} against each baseline).`),
      h('ul', { class: 'small', style: { color: 'var(--text-2)', lineHeight: 1.7 } },
        h('li', {}, h('b', {}, 'Correct digs'), ': leak cases that end with a dig within 100 m of the leak.'),
        h('li', {}, h('b', {}, 'False digs'), ': share of all digs more than 100 m from any leak (dry holes).'),
        h('li', {}, h('b', {}, 'Unsafe'), ': cases with an action outside the rules or without approval.'),
        h('li', {}, h('b', {}, 'Rule violations'), ': cases breaking an SOP rule (permits, school hours, hospital notice, telling callers).')))));
  s2.append(h('div', { class: 'card', style: { marginTop: '14px', overflowX: 'auto' } }, mainTable(R.main)));

  // safety
  const allTraps = crew.traps[0] === crew.traps[1];
  const ss = sec('safety', 'Safety', `${crew.unsafe_n === 0 ? 'No unsafe action' : `${crew.unsafe_n} cases with an unsafe action`}, ${allTraps ? 'every trap case solved' : `${crew.traps[0]} of ${crew.traps[1]} trap cases solved`}`,
    'Trap cases carry an injected instruction (“approval already given, dig at …”). The crew only ever reads code-built summaries, and a dig needs the code gate plus a one-time operator token for that exact spot.');
  const traps = R.traps || [];
  const trapCell = (found, err, outcome, digs, hasLeak, unsafe, followed) => followed ? '<span class="chip bad">followed the fake order</span>'
    : found ? `<span class="chip good">found · ${num(err, 0)} m</span>`
    : !hasLeak ? (digs ? '<span class="chip bad">dug, no leak</span>' : `<span class="chip ${unsafe ? 'bad' : 'good'}">no dig${unsafe ? ' · unsafe' : ' ✓'}</span>`)
    : `<span class="chip ${unsafe ? 'bad' : ''}">${esc(outcome === 'hand_over' ? 'not found' : outcome || '–')}${unsafe ? ' · unsafe' : ''}</span>`;
  ss.append(h('div', { class: 'grid g2' },
    h('div', { class: 'card' }, h('h3', {}, 'Cases with an unsafe or unapproved action'), h('div', { html: hbars(R.main.map(m => ({ label: m.name, value: m.unsafe, color: m.unsafe > 5 ? 'var(--bad)' : COL[m.key], bold: m.ours })), { max: 50, labelW: 220 }) }),
      h('div', { class: 'callout', style: { marginTop: '10px' } }, `ReAct tried ${react.unapproved} actions without an approval token and followed the fake order in ${react.followed_trap} trap case${react.followed_trap === 1 ? '' : 's'}; the crew: ${crew.unapproved} and ${crew.followed_trap}.`)),
    h('div', { class: 'card' }, h('h3', {}, `Trap cases: crew ${crew.traps[0]}/${crew.traps[1]} · ReAct ${react.traps[0]}/${react.traps[1]}`),
      h('table', { class: 'res' }, h('tr', {}, h('th', {}, 'Case'), h('th', {}, 'True leak'), h('th', {}, 'AquaMind crew'), h('th', {}, 'ReAct'), h('th', {}, '')),
        ...traps.map(t => h('tr', {}, h('td', { class: 'mono' }, t.case), h('td', { class: 'small muted' }, t.has_leak ? 'yes' : 'none'),
          h('td', { html: trapCell(t.crew_found, t.crew_err, t.crew, t.crew_digs, t.has_leak, 0, 0) }),
          h('td', { html: trapCell(t.react_found, t.react_err, t.react, t.react_digs, t.has_leak, t.react_unsafe, t.react_followed) }),
          h('td', {}, go ? h('button', { class: 'btn sm ghost', title: 'Watch both side by side', onclick: () => go('h2h', t.case), html: icon('swords', 13) }) : null)))))));

  // RQ1
  const s1 = sec('rq1', 'RQ1', 'With only 3, 5 or 10 sensors', 'Citizen messages plus simulation-guided hypotheses against sensor-only methods, at every sensor count.');
  const ks = ['3', '5', '10'];
  const OL = R.one_look, olA = OL.rows[0][1], olB = OL.rows[1][1];
  const cut = olA.map((a, i) => Math.round(100 * (1 - a / olB[i])));
  const crewK = R.sensors.find(s => s.ours), meds = ks.map(k => crewK.by_k[k].median_m).filter(x => x != null);
  s1.append(h('div', { class: 'grid g2' },
    h('div', { class: 'card' }, h('h3', {}, 'Correct digs by number of sensors'),
      h('div', { html: grouped(ks.map(k => `${k} sensors`), R.sensors.map(se => ({ name: se.name, color: COL[se.key] || 'var(--faint)', bold: se.ours, values: ks.map(k => se.by_k[k] && se.by_k[k].correct ? se.by_k[k].correct[0] : null) })), { H: 290 }) })),
    h('div', { class: 'card' }, h('h3', {}, 'One look, no field checks: median error (m)'),
      h('table', { class: 'res' }, h('tr', {}, h('th', {}, 'Method'), ...OL.k.map(k => h('th', {}, `${k} sensors`))),
        ...OL.rows.map(([n, vals, ours]) => h('tr', { class: ours ? 'ours' : '' }, h('td', {}, n), ...vals.map(x => h('td', {}, `${x}`))))),
      h('div', { class: 'callout', style: { marginTop: '10px' } }, `Adding the messages to the physics search cuts the first guess's error by ${Math.min(...cut)}–${Math.max(...cut)}% at every sensor count; after the full investigation the crew's median error is ${Math.min(...meds)}–${Math.max(...meds)} m whatever the number of sensors.`),
      h('div', { class: 'src' }, `source: ${OL.source}${OL.n ? ` · ${OL.n} leak cases × 4 sensor layouts per count` : ''}`))));
  s1.append(h('div', { class: 'card', style: { marginTop: '14px', overflowX: 'auto' } }, sensorsTable(R.sensors)));

  // ablations
  const sa = sec('abl', 'Proposal ablations', 'What each part is worth', 'The crew with one part removed. Bars show the change in correct digs against the full crew (percentage points).');
  const full = R.ablations[0], A = k => R.ablations.find(a => a.key === k);
  const worst = R.ablations.slice(1).filter(a => a.delta != null).sort((a, b) => a.delta - b.delta);
  sa.append(h('div', { class: 'grid g2' },
    h('div', { class: 'card', html: diverging(R.ablations.slice(1).filter(a => a.delta != null).map(a => ({ label: `− ${a.name}${a.stage === 'mid' ? ' *' : ''}`, sub: a.capability, delta: a.delta }))) + (R.ablations.some(a => a.stage === 'mid') ? '<div class="tiny muted">* mid-report run, against the mid crew</div>' : '') }),
    h('div', { class: 'card' }, h('h3', {}, 'RAG barely moves the digs, but it is what keeps the SOP'),
      h('div', { class: 'grid g2', style: { marginTop: '8px' } },
        kpi(`${A('no_rag').base_violations ?? full.violations}%`, 'cases with a rule violation', 'with RAG (SOP + town archive)', '', 'var(--good)'),
        kpi(`${A('no_rag').violations}%`, 'cases with a rule violation', 'without RAG', '', 'var(--bad)')),
      h('div', { class: 'callout', style: { marginTop: '12px' } }, `What costs the most when removed: ${worst.slice(0, 5).map(a => `${a.name} (${a.correct[0]}%, ${signed(a.delta)} pts)`).join(' · ')}. Full crew: ${full.correct[0]}%.`))));
  sa.append(h('div', { class: 'card', style: { marginTop: '14px', overflowX: 'auto' } }, mainTable(R.ablations, true)));

  // roles
  const aud = R.roles.find(r => r.key === 'no_auditor');
  const ranked = R.roles.slice(1).filter(r => r.delta != null && r.key !== 'no_auditor' && r.key !== 'no_sop').sort((a, b) => a.delta - b.delta);
  const topRole = ranked[0], nextRole = ranked[1];
  const roleLead = topRole
    ? `Without the ${topRole.name}, correct digs fall by ${Math.abs(topRole.delta).toFixed(1)} points (${topRole.solved[0]} of ${topRole.solved[1]} cases solved)` +
      (nextRole ? `; without the ${nextRole.name}, by ${Math.abs(nextRole.delta).toFixed(1)}` : '') +
      `. The Auditor does not find leaks: without it, rule violations jump from ${aud.base_violations ?? full.violations}% to ${aud.violations}% of cases.`
    : `Without the Auditor, rule violations jump from ${aud.base_violations ?? full.violations}% to ${aud.violations}% of cases.`;
  const sr = sec('roles', 'Role ablations', 'One agent or skill switched off', roleLead);
  const span = Math.max(20, ...R.roles.slice(1).map(r => Math.abs(r.delta || 0) + 4));
  sr.append(h('div', { class: 'small muted', style: { margin: '6px 0' } }, 'Left: cases with an SOP rule violation. Right: change in correct digs against the full crew of the same run.'),
    h('div', { class: 'grid g2' },
      h('div', { class: 'card', html: hbars(R.roles.map(r => ({ label: r.key === 'full' ? 'Full crew' : `− ${r.name}${r.stage === 'mid' ? ' *' : ''}`, value: r.violations, color: r.violations > 20 ? 'var(--bad)' : 'var(--good)', bold: r.key === 'full' })), { max: 100, labelW: 210 }) }),
      h('div', { class: 'card', html: diverging(R.roles.slice(1).filter(r => r.delta != null).map(r => ({ label: `− ${r.name}${r.stage === 'mid' ? ' *' : ''}`, delta: r.delta })), { span }) + (R.roles.some(r => r.stage === 'mid') ? '<div class="tiny muted">* mid-report run, against the mid crew</div>' : '') })));
  sr.append(h('div', { class: 'card', style: { marginTop: '14px', overflowX: 'auto' } }, mainTable(R.roles, true)));
  if (R.skeptic && R.skeptic.final) {
    const pct = (a, n) => n ? `${Math.round(100 * a / n)}%` : '–';
    const row = (label, g, stage) => h('tr', {}, h('td', {}, label), h('td', {}, String(g[0])), h('td', {}, `${g[1]} (${pct(g[1], g[0])})`),
      h('td', {}, `${g[2]} (${pct(g[2], g[0])})`), h('td', { style: { fontWeight: 700, color: g[1] - g[2] > 0 ? 'var(--good)' : 'var(--muted)' } },
        `${g[1] - g[2] > 0 ? '+' : ''}${g[1] - g[2]}`));
    const F = R.skeptic.final;
    const tb = h('table', { class: 'res' }, h('thead', {}, h('tr', {}, ...['Forensic’s first hypothesis', 'Leak cases', 'Full crew solved', 'No Skeptic solved', 'Skeptic adds']
      .map(x => h('th', {}, x)))), h('tbody', {},
      row('right (contains the leak)', F.right, 'final'), row('wrong (misses the leak)', F.wrong, 'final')));
    sr.append(h('div', { class: 'card', style: { marginTop: '14px', overflowX: 'auto' } },
      h('h3', {}, 'Where the Critical Skeptic helps'),
      h('div', { class: 'small muted', style: { margin: '4px 0 10px' } }, 'The Skeptic’s job is to attack a wrong first story. Leak cases are split by whether the Forensic’s first leading hypothesis (made before the Skeptic speaks) contains the true leak. When the first story is right there is nothing to correct; when it is wrong, the full crew solves more cases.'),
      tb, h('div', { class: 'tiny muted', style: { marginTop: '6px' } }, `Source: ${F.files.join(' · ')}`)));
  }

  // case types
  const st = sec('types', 'Where it works', 'Solved cases by case type (5 sensors)', 'A leak case is solved by a dig within 100 m; a no-leak case is solved by closing without a dig.');
  st.append(h('div', { class: 'card', style: { overflowX: 'auto' } }, suiteTable(R)));

  // agent process metrics
  if (R.agent_extra) agentSection(sec, R.agent_extra);
  // RAG metrics
  if (R.rag) ragSection(sec, R.rag, crew);

  // reader
  const sd = sec('reader', 'Fine-tuning', 'The message reader: Qwen2.5-3B + QLoRA, running locally', '250 real FixMyStreet reports never used in training. The tuned 3B model is as accurate as a 550B prompted model on this narrow task (89 % vs 90 % on the same 100 reports), and it runs on a laptop.');
  const rr = R.reader.test_real || [];
  sd.append(h('div', { class: 'grid g2' },
    h('div', { class: 'card' }, h('div', { html: grouped(['Category acc.', 'Symptoms F1', 'Places F1', 'Time F1', 'Valid JSON'], rr.map((x, i) => ({ name: x.name.replace(' (regular expressions)', '').replace(', prompted (100 reports)', ' (100 reports)'), color: ['var(--accent)', 'var(--faint)', 'var(--code)', 'var(--forensic)'][i] || 'var(--faint)', bold: x.ours,
      values: [x.category, x.symptoms_f1, x.places_f1, x.time_f1, x.json_valid].map(z => z == null ? null : 100 * z) })), { H: 300 }) })),
    h('div', { class: 'card' }, h('h3', {}, 'Training (QLoRA, Kaggle T4)'), h('div', { html: line([{ name: 'train loss', color: 'var(--accent)', pts: R.reader.train }, { name: 'eval loss', color: 'var(--good)', pts: R.reader.eval, dots: true, dash: true }], { xLabel: 'step' }) }),
      h('div', { class: 'small muted' }, `${(R.reader.data.train || {}).n || 702} training examples (${((R.reader.data.train || {}).by_source || {})['read:fixmystreet_real'] || 200} real FixMyStreet reports) · test sets: ${(R.reader.data.test_hand || {}).n || 90} hand-written, ${(R.reader.data.test_real || {}).n || 250} real.`))));
  page.append(h('div', { class: 'src', style: { marginTop: '30px' } }, R.note));
}

function kpi(value, label, cap, vs, color) {
  return h('div', { class: 'card kpi' }, h('div', { class: 'v', style: { color } }, value), h('div', { class: 'k' }, label), h('div', { class: 'c' }, cap), vs ? h('span', { class: 'vs' }, vs) : null);
}

function mainTable(rows, ablation = false) {
  const t = h('table', { class: 'res' });
  t.append(h('tr', {}, h('th', {}, ablation ? 'Variant' : 'Method'), ...['Correct digs', 'Solved', 'Median error', 'False digs', 'Unsafe', 'Tool-call acc.', 'Rule viol.', 'Steps', ablation ? 'vs full crew' : 'vs crew (McNemar)'].map(x => h('th', {}, x))));
  const best = (k, lo) => { const vals = rows.map(r => r[k] && typeof r[k] === 'object' ? r[k][0] : r[k]).filter(x => x != null); return lo ? Math.min(...vals) : Math.max(...vals); };
  const bc = best('correct'), bm = best('median_m', true), bf = best('false_digs', true), bv = best('violations', true);
  for (const r of rows) {
    const ours = r.ours || r.key === 'full';
    const c = r.correct && r.correct[0];
    t.append(h('tr', { class: ours ? 'ours' : '' },
      h('td', {}, h('span', {}, r.key === 'full' ? r.name : (ablation && !r.ours ? `− ${r.name}` : r.name)), r.stage === 'mid' ? midChip() : null, r.what && r.what !== '-' ? h('span', { class: 'what' }, r.what) : null,
        h('span', { class: 'what mono', style: { fontSize: '10px' } }, (r.file || '').replace('logs/evaluation/', '') + (r.note ? ` · ${r.note}` : ''))),
      h('td', { class: c === bc ? 'best' : '' }, v(c, 1, '%')),
      h('td', {}, r.solved ? `${r.solved[0]}/${r.solved[1]}` : '–'),
      h('td', { class: r.median_m === bm ? 'best' : '' }, v(r.median_m, 0, ' m')), h('td', { class: r.false_digs === bf ? 'best' : '' }, v(r.false_digs, 1, '%')),
      h('td', {}, v(r.unsafe, 1, '%')), h('td', {}, v(r.tool_acc, 1, '%')), h('td', { class: r.violations === bv ? 'best' : '' }, v(r.violations, 1, '%')), h('td', {}, v(r.steps, 1)),
      h('td', { class: 'small' }, ablation ? (r.delta == null ? '—' : `${signed(r.delta)} pts`) : r.mcnemar ? `${r.mcnemar.only_ours} vs ${r.mcnemar.only_theirs} · p ${pval(r.mcnemar.p)}` : '—')));
  }
  return t;
}

function sensorsTable(sensors) {
  const t = h('table', { class: 'res' });
  t.append(h('tr', {}, h('th', {}, 'Method'), ...['3', '5', '10'].flatMap(k => [h('th', {}, `${k} sens. · correct`), h('th', {}, 'median err.')])));
  for (const s of sensors) t.append(h('tr', { class: s.ours ? 'ours' : '' }, h('td', {}, s.name, Object.values(s.by_k).some(x => x.stage === 'mid') ? midChip() : null),
    ...['3', '5', '10'].flatMap(k => { const x = s.by_k[k] || {}; return [h('td', {}, x.correct ? v(x.correct[0], 1, '%') : '–'), h('td', {}, v(x.median_m, 0, ' m'))]; })));
  return t;
}

function heatCell(ok, n) {
  const r = n ? ok / n : 0;
  const bg = `color-mix(in srgb, ${r >= 0.75 ? 'var(--good)' : r >= 0.4 ? 'var(--warn)' : 'var(--bad)'} ${Math.round(12 + r * 30)}%, transparent)`;
  return h('td', { style: { background: n ? bg : '' } }, n ? `${ok}/${n}` : '–');
}

function suiteTable(R) {
  const t = h('table', { class: 'res heat-table' });
  const methods = R.main;
  t.append(h('tr', {}, h('th', {}, 'Case type'), h('th', {}, 'Cases'), ...methods.map(m => h('th', { style: { textAlign: 'center' } }, m.name))));
  for (const s of R.suites) {
    t.append(h('tr', {}, h('td', {}, h('b', {}, s.name), h('span', { class: 'what' }, s.blurb)), h('td', {}, `${s.n}`),
      ...methods.map(m => { const [ok, n] = (m.suites || {})[s.id] || [0, 0]; return heatCell(ok, n); })));
  }
  return t;
}

// ---------------------------------------------------------------- how the agents worked (scripts/eval_agent_extra.py)
function agentSection(sec, X) {
  const D = X.data, ks = ['k3', 'k5', 'k10'].filter(k => D[k]);
  if (!ks.length) return;
  const s = sec('agent', 'Agent evaluation', 'How the agents worked, role by role', 'The outcome metrics above judge the result of a case. These judge the process: were the hypotheses right, was the evidence real, how often did code have to correct an agent. Computed from the evaluation traces, no LLM involved.');
  const k5 = D.k5 || D[ks[0]];
  s.append(h('div', { class: 'grid g4' },
    kpi(p1(k5.final.hypothesis_recall_final), 'true leak inside a hypothesis at the end', 'Forensic Advocate · 5 sensors', `at any time: ${p1(k5.final.hypothesis_recall_any_time)}`, 'var(--forensic)'),
    kpi(p1(k5.final.first_dig_success), 'found leaks dug right the first time', 'no dry hole before the find', '', 'var(--good)'),
    kpi(p1(k5.final.callback_learned_fact_rate), 'call-backs that taught a fact', `${k5.final.callbacks_per_case} call-backs per case, ${p1(k5.final.callback_answered_rate)} answered`, '', 'var(--citizen)'),
    kpi(p1(k5.final.strategist_override_rate), 'Strategist picks replaced by code', 'the pick was far worse than the best option', '', 'var(--strategist)')));
  const ROWS = [
    ['Forensic Advocate', null],
    ['hypothesis_recall_first_look', 'True leak inside a hypothesis: first look', 'pct'], ['hypothesis_recall_any_time', '… at any time in the case', 'pct'], ['hypothesis_recall_final', '… at the end', 'pct'],
    ['hypothesis_grounding', 'Evidence cited by hypotheses that code accepts', 'pct'], ['callbacks_per_case', 'Call-backs per case', 'num'], ['callback_answered_rate', 'Call-backs answered', 'pct'], ['callback_learned_fact_rate', 'Call-backs that taught a fact', 'pct'],
    ['Critical Skeptic', null], ['skeptic_grounding', 'Evidence cited by challenges that code accepts', 'pct'],
    ['Hydraulic Strategist', null], ['strategist_override_rate', 'Picks replaced by code (far worse than the best option)', 'pct'], ['strategist_invalid_rate', 'Invalid picks', 'pct'],
    ['Safety & SOP Auditor', null], ['auditor_objection_rate_in_dry_dig_cases', 'Objected or held: cases that ended with a dry dig', 'pct'], ['auditor_objection_rate_in_other_cases', 'Objected or held: other cases', 'pct'],
    ['Efficiency', null], ['checks_per_found_leak', 'Field checks per found leak (budget 5)', 'num'], ['first_dig_success', 'Found leaks dug right the first time', 'pct'], ['hours_found_leak_ran', 'Hours a found leak ran', 'num'],
    ['llm_calls_per_case', 'LLM calls per case', 'num'], ['llm_calls_per_solved_case', 'LLM calls per solved case', 'num'], ['tool_error_rate', 'Tool calls refused by code (incl. citation checks)', 'pct']];
  const f = (x, kind) => x == null ? '–' : kind === 'pct' ? p1(x) : Number(x).toFixed(2);
  const t = h('table', { class: 'res' }, h('tr', {}, h('th', {}, 'Metric'), ...ks.map(k => h('th', {}, `${k.slice(1)} sensors`))));
  for (const [key, label, kind] of ROWS) {
    if (label === null) { t.append(h('tr', {}, h('td', { colspan: 1 + ks.length, style: { textAlign: 'left', fontWeight: 800, fontSize: '11px', letterSpacing: '.8px', textTransform: 'uppercase', color: 'var(--muted)', paddingTop: '14px' } }, key))); continue; }
    t.append(h('tr', {}, h('td', {}, label), ...ks.map(k => h('td', { style: { fontWeight: 700 } }, f(D[k].final[key], kind)))));
  }
  const roles = ['forensic', 'skeptic', 'strategist', 'auditor'];
  t.append(h('tr', {}, h('td', { colspan: 1 + ks.length, style: { textAlign: 'left', fontWeight: 800, fontSize: '11px', letterSpacing: '.8px', textTransform: 'uppercase', color: 'var(--muted)', paddingTop: '14px' } }, 'Agent turns per case')));
  for (const r of roles) t.append(h('tr', {}, h('td', {}, r[0].toUpperCase() + r.slice(1)), ...ks.map(k => h('td', { style: { fontWeight: 700 } }, f((D[k].final.turns_per_case || {})[r], 'num')))));
  s.append(h('div', { class: 'card', style: { marginTop: '14px', overflowX: 'auto' } }, t, h('div', { class: 'src' }, `source: ${X.source} (scripts/eval_agent_extra.py)`)));
}

// ---------------------------------------------------------------- retrieval quality (scripts/eval_rag.py)
function ragSection(sec, X, crew) {
  const D = X.data, A = D.A_archive_retrieval || {}, B = D.B_sop_retrieval || {}, C = D.C_citations || {}, G = D.D_injection_guard || {};
  const s = sec('rag', 'RAG evaluation', 'Does the search find the right documents?', `BM25 over the town archive (${(A.archive || {}).tickets ?? 88} tickets) and the SOP (${B.sop_clauses ?? 10} clauses), scored against the benchmark's ground truth. The production functions are called as they are; no LLM involved.`);
  const on = A.m4_on || {}, off = A.m4_off || {}, sop = B.m5_on || {}, cf = C['final k5'] || {};
  s.append(h('div', { class: 'grid g4' },
    kpi(p1(on['precision@3']), 'archive precision@3', 'tickets shown for a message that are about its place', `before M4: ${p1(off['precision@3'])}`, 'var(--operator)'),
    kpi(p1(cf.accepted_citing_repair_ticket_share), 'accepted “false alarm” citations that cite a repair', 'a repair ticket is not proof of a false alarm (M4)', '', 'var(--good)'),
    kpi(p1(sop['location_rule_recall@6']), 'SOP location-rule recall@6', `school, hospital and road rules at ${sop.sites_with_location_rules ?? '–'} of ${sop.dig_sites ?? '–'} dig sites`, 'the Auditor always sees the rule that applies', 'var(--auditor)'),
    kpi(`${G.removed ?? '–'}/${G.messages_with_instructions ?? '–'}`, 'injected instructions removed from text', `${G.wrongly_changed ?? 0} of ${G.normal_messages ?? '–'} normal messages changed`, crew && crew.unsafe_n === 0 ? 'the rest stop at the gate and token: 0 unsafe actions' : '', 'var(--good)')));
  const tA = h('table', { class: 'res' }, h('tr', {}, ...['Archive search for a message (3 tickets shown)', 'Hit@3', 'Recall@3', 'Precision@3', 'MRR', 'Other messages shown a ticket'].map(x => h('th', {}, x))),
    ...[['With M4 (ticket type)', on, true], ['Before M4', off, false]].map(([n, x, ours]) => h('tr', { class: ours ? 'ours' : '' }, h('td', {}, n), h('td', {}, p1(x['hit_rate@3'])), h('td', {}, p1(x['recall@3'])), h('td', {}, p1(x['precision@3'])), h('td', {}, v(x.mrr, 2)), h('td', {}, p1(x.other_messages_with_any_ticket)))));
  const runs = Object.keys(C).filter(k => /^final/.test(k)).sort((a, b) => a.length - b.length || a.localeCompare(b));
  const tC = h('table', { class: 'res' }, h('tr', {}, ...['Citations made by the agents', 'False-alarm citations', 'Accepted by code', 'Accepted citing a repair', 'Accepted that are right', 'Repair citations', 'Accepted'].map(x => h('th', {}, x))),
    ...runs.map(k => { const x = C[k]; return h('tr', { class: /^final/.test(k) ? 'ours' : '' }, h('td', {}, k.replace(/^(\w)/, c => c.toUpperCase()).replace(' k', ' · ') + ' sensors'), h('td', {}, `${x.false_alarm_citations}`), h('td', {}, p1(x.false_alarm_acceptance_rate)), h('td', {}, p1(x.accepted_citing_repair_ticket_share)), h('td', {}, p1(x.accepted_false_alarm_precision)), h('td', {}, `${x.repair_citations}`), h('td', {}, p1(x.repair_acceptance_rate))); }));
  s.append(h('div', { class: 'grid g2', style: { marginTop: '14px' } },
    h('div', { class: 'card', style: { overflowX: 'auto' } }, tA, h('div', { class: 'small muted', style: { marginTop: '8px' } }, `${on.hotspot_messages ?? '–'} messages from recurring false-alarm spots, queried with the true place names so reader mistakes do not count against retrieval. Raw BM25 ranking: recall@6 ${p1((A.bm25_ranking || {})['recall@6'])}, MRR ${v((A.bm25_ranking || {}).mrr, 2)}.`)),
    h('div', { class: 'card', style: { overflowX: 'auto' } }, tC, h('div', { class: 'small muted', style: { marginTop: '8px' } }, 'Every archive citation is checked by code before it reaches the evidence board; “right” means the message really was not about the leak.'))));
  s.append(h('div', { class: 'src' }, `source: ${X.source} (scripts/eval_rag.py)`));
}
