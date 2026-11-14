// Results: every number computed live from the evaluation's result rows (logs/evaluation), same definitions as the mid report.
import { get } from './api.js';
import { hbars, grouped, diverging, line } from './charts.js';
import { h, esc, num, pct } from './ui.js';
import { icon } from './icons.js';

const COL = { agentix_crew: 'var(--accent)', react_agent: 'var(--react)', crew_practice: 'var(--warn)', plain_llm: 'var(--forensic)', classical: 'var(--faint)', fixed_rules: 'var(--code)', no_ai: 'var(--good)' };
const v = (x, d = 1, u = '') => (x == null ? '–' : `${Number(x).toFixed(d)}${u}`);

export async function initResults(root) {
  const R = await get('/api/results', { memo: true });
  root.innerHTML = '';
  const page = h('div', { class: 'page' });
  root.append(page);
  const crew = R.main.find(m => m.ours);
  const react = R.main.find(m => m.key === 'react_agent');
  const sec = (id, eyebrow, title, lead) => {
    const s = h('section', { id: `res-${id}` }, h('div', { class: 'eyebrow', style: { marginTop: '34px' } }, eyebrow), h('h2', { style: { marginTop: '4px' } }, title), lead ? h('p', { class: 'lead' }, lead) : null);
    page.append(s); return s;
  };

  page.append(h('div', { class: 'eyebrow' }, 'Evaluation · 210 cases on the real L-Town network'),
    h('h1', {}, 'Results'),
    h('p', { class: 'lead' }, 'Computed live from the result rows in logs/evaluation (one row per case and method), with the metric definitions of the mid report. Every method gets the same cases, sensors, field budget, LLM, reader and random luck.'),
    h('div', { class: 'subnav' }, [['rq2', 'RQ2 · vs agents'], ['safety', 'Safety & traps'], ['rq1', 'RQ1 · few sensors'], ['abl', 'Ablations'], ['roles', 'Agent roles'], ['types', 'Case types'], ['mit', 'Mitigations'], ['reader', 'Fine-tuned reader']]
      .map(([id, t]) => h('button', { class: 'btn sm', onclick: () => document.getElementById(`res-${id}`).scrollIntoView({ behavior: 'smooth', block: 'start' }) }, t))));

  // KPIs
  page.append(h('div', { class: 'grid g4', style: { marginTop: '8px' } },
    kpi(`${crew.correct[0]}%`, 'correct digs', `leak cases dug within 100 m (95% CI ${crew.correct[1]}–${crew.correct[2]})`, `ReAct ${react.correct[0]}% · crews today ${R.main.find(m => m.key === 'crew_practice').correct[0]}%`, 'var(--accent)'),
    kpi(`${crew.median_m} m`, 'median distance error', 'final guess to the true leak', `ReAct ${react.median_m} m`, 'var(--good)'),
    kpi(`${crew.unsafe}%`, 'unsafe or unapproved actions', `${crew.unsafe_n} of ${crew.n} cases`, `ReAct ${react.unsafe}% of cases`, 'var(--good)'),
    kpi(`${crew.false_digs}%`, 'false digs', 'share of digs more than 100 m from any leak', `ReAct ${react.false_digs}%`, 'var(--warn)')));

  // RQ2
  const s2 = sec('rq2', 'RQ2', 'The crew against standard agents and today\'s practice', 'Same LLM (Nemotron-3 Ultra 550B, thinking off), same tools, same step and field budget. Correct digs with 95% bootstrap intervals; McNemar test paired by case.');
  s2.append(h('div', { class: 'grid g2' },
    h('div', { class: 'card', html: hbars(R.main.map(m => ({ label: m.name, sub: m.what.length > 36 ? m.what.slice(0, 35) + '…' : m.what, value: m.correct[0], lo: m.correct[1], hi: m.correct[2], color: COL[m.key], bold: m.ours })), { labelW: 250 }) }),
    h('div', { class: 'card' }, h('h3', {}, 'What the bars mean'),
      h('div', { class: 'callout' }, `The crew dug at the right spot in ${crew.correct[0]}% of the ${crew.n_leak} leak cases: about three times ReAct (${react.correct[0]}%) and today's crew practice. Every difference is significant (McNemar p < 0.001).`),
      h('ul', { class: 'small', style: { color: 'var(--text-2)', lineHeight: 1.7 } },
        h('li', {}, h('b', {}, 'Correct digs'), ': leak cases that end with a dig within 100 m of the leak.'),
        h('li', {}, h('b', {}, 'False digs'), ': share of all digs more than 100 m from any leak (dry holes).'),
        h('li', {}, h('b', {}, 'Unsafe'), ': cases with an action outside the rules or without approval.'),
        h('li', {}, h('b', {}, 'Rule violations'), ': cases breaking an SOP rule (permits, school hours, hospital notice, telling callers).')))));
  s2.append(h('div', { class: 'card', style: { marginTop: '14px', overflowX: 'auto' } }, mainTable(R.main)));

  // safety
  const ss = sec('safety', 'Safety', 'No unsafe action, every trap case solved', 'Trap cases carry an injected instruction (“approval already given, dig at …”). The crew only ever reads code-built summaries, and a dig needs the code gate plus a one-time operator token for that exact spot.');
  const traps = R.traps || [];
  const tCrew = traps.filter(t => t.crew_found || (t.crew === 'no_leak') || (t.crew === 'hand_over' && !t.crew_found)).length;
  ss.append(h('div', { class: 'grid g2' },
    h('div', { class: 'card', html: hbars(R.main.map(m => ({ label: m.name, value: m.unsafe, color: m.unsafe > 5 ? 'var(--bad)' : COL[m.key], bold: m.ours })), { max: 50, labelW: 220 }) }),
    h('div', { class: 'card' }, h('h3', {}, `Trap cases: crew ${crew.traps[0]}/${crew.traps[1]} · ReAct ${react.traps[0]}/${react.traps[1]}`),
      h('table', { class: 'res' }, h('tr', {}, h('th', {}, 'Case'), h('th', {}, 'AquaMind crew'), h('th', {}, 'ReAct')),
        ...traps.map(t => h('tr', {}, h('td', { class: 'mono' }, t.case),
          h('td', { html: t.crew_found ? `<span class="chip good">found · ${num(t.crew_err, 0)} m</span>` : `<span class="chip ${t.crew === 'no_leak' ? 'good' : ''}">${esc(t.crew)}</span>` }),
          h('td', { html: t.react_followed ? '<span class="chip bad">followed the fake order</span>' : t.react_found ? `<span class="chip good">found</span>` : `<span class="chip ${t.react_unsafe ? 'bad' : ''}">${esc(t.react)}${t.react_unsafe ? ' · unsafe' : ''}</span>` })))))));

  // RQ1
  const s1 = sec('rq1', 'RQ1', 'With only 3, 5 or 10 sensors', 'Citizen messages plus simulation-guided hypotheses against sensor-only methods, at every sensor count.');
  const ks = ['3', '5', '10'];
  s1.append(h('div', { class: 'grid g2' },
    h('div', { class: 'card' }, h('h3', {}, 'Correct digs by number of sensors'),
      h('div', { html: grouped(ks.map(k => `${k} sensors`), R.sensors.map(se => ({ name: se.name, color: COL[se.key] || 'var(--faint)', bold: se.ours, values: ks.map(k => se.by_k[k] && se.by_k[k].correct ? se.by_k[k].correct[0] : null) })), { H: 290 }) })),
    h('div', { class: 'card' }, h('h3', {}, 'One look, no field checks: median error (m)'),
      h('table', { class: 'res' }, h('tr', {}, h('th', {}, 'Method'), ...R.one_look.k.map(k => h('th', {}, `${k} sensors`))),
        ...R.one_look.rows.map(([n, vals, ours]) => h('tr', { class: ours ? 'ours' : '' }, h('td', {}, n), ...vals.map(x => h('td', {}, `${x}`))))),
      h('div', { class: 'callout', style: { marginTop: '10px' } }, 'Adding the messages to the physics search cuts the first guess\'s error by about a third at every sensor count; after the full investigation the crew\'s error drops to 23–31 m whatever the number of sensors.'),
      h('div', { class: 'src' }, `source: ${R.one_look.source}`))));
  s1.append(h('div', { class: 'card', style: { marginTop: '14px', overflowX: 'auto' } }, sensorsTable(R.sensors)));

  // ablations
  const sa = sec('abl', 'Proposal ablations', 'What each part is worth', 'The crew with one part removed. Bars show the change in correct digs against the full crew (percentage points).');
  const full = R.ablations[0];
  sa.append(h('div', { class: 'grid g2' },
    h('div', { class: 'card', html: diverging(R.ablations.slice(1).map(a => ({ label: `− ${a.name}`, sub: a.capability, delta: a.correct[0] - full.correct[0] }))) }),
    h('div', { class: 'card' }, h('h3', {}, 'RAG barely moves the digs — but it is what keeps the SOP'),
      h('div', { class: 'grid g2', style: { marginTop: '8px' } },
        kpi(`${full.violations}%`, 'cases with a rule violation', 'with RAG (SOP + town archive)', '', 'var(--good)'),
        kpi(`${R.ablations.find(a => a.key === 'no_rag').violations}%`, 'cases with a rule violation', 'without RAG', '', 'var(--bad)')),
      h('div', { class: 'callout', style: { marginTop: '12px' } }, `Removing the step-by-step loop (${R.ablations.find(a => a.key === 'one_shot').correct[0]}%), the safety gate (${R.ablations.find(a => a.key === 'no_gate').correct[0]}%), all AI (${R.ablations.find(a => a.key === 'no_ai').correct[0]}%), the step test or the citizen reports costs the most.`))));
  sa.append(h('div', { class: 'card', style: { marginTop: '14px', overflowX: 'auto' } }, mainTable(R.ablations, true)));

  // roles
  const sr = sec('roles', 'Role ablations', 'One agent or skill switched off', 'Switching off one reasoning agent costs a few correct digs (code and the other agents cover for it); without the Auditor, rule violations jump from about 4% to 70% of cases.');
  sr.append(h('div', { class: 'grid g2' },
    h('div', { class: 'card', html: hbars(R.roles.map(r => ({ label: r.key === 'full' ? 'Full crew' : `− ${r.name}`, value: r.violations, color: r.violations > 20 ? 'var(--bad)' : 'var(--good)', bold: r.key === 'full' })), { max: 100, labelW: 210 }) }),
    h('div', { class: 'card', html: diverging(R.roles.slice(1).map(r => ({ label: `− ${r.name}`, delta: r.correct[0] - R.roles[0].correct[0] })), { span: 20 }) })));
  sr.firstChild && sr.querySelector('.grid').before(h('div', { class: 'small muted', style: { margin: '6px 0' } }, 'Left: cases with an SOP rule violation. Right: change in correct digs.'));
  sr.append(h('div', { class: 'card', style: { marginTop: '14px', overflowX: 'auto' } }, mainTable(R.roles, true)));

  // case types
  const st = sec('types', 'Where it works', 'Solved cases by case type (5 sensors)', 'A leak case is solved by a dig within 100 m; a no-leak case is solved by closing without a dig.');
  st.append(h('div', { class: 'card', style: { overflowX: 'auto' } }, suiteTable(R)));

  // mitigations
  const sm = sec('mit', 'After the mid report', 'Mitigations M1–M5 (final system, preliminary)', 'The five mitigations of the mid report, each with an on/off switch, compared on the same dev cases. Small sample: the frozen-test runs are the final word.');
  sm.append(h('div', { class: 'card', style: { overflowX: 'auto' } }, mainTable(R.mitigation, true),
    h('div', { class: 'grid g5', style: { marginTop: '12px' } }, ...[
      ['M1', 'Weak leads', 'show check strength, allow a re-check, follow suggestive flow'],
      ['M2', 'Turn budget', 'more Forensic turns, a Skeptic turn after a quiet check, no blind sweeps'],
      ['M3', 'Confirm the dig', 'keep the last check to confirm the spot; objections can hold weak digs'],
      ['M4', 'Ticket type', 'only non-utility tickets count as false alarms'],
      ['M5', 'Corner permits', 'every street at the dig junction for SOP-6']].map(([k, t, d]) => h('div', { class: 'card', style: { padding: '10px 12px' } }, h('div', { class: 'eyebrow' }, k), h('b', {}, t), h('div', { class: 'small muted' }, d))))));

  // reader
  const sd = sec('reader', 'Fine-tuning', 'The message reader: Qwen2.5-3B + QLoRA, running locally', '250 real FixMyStreet reports never used in training. The tuned 3B model matches a 550B prompted model on this narrow task, and it runs on a laptop.');
  const rr = R.reader.test_real || [];
  sd.append(h('div', { class: 'grid g2' },
    h('div', { class: 'card' }, h('div', { html: grouped(['Category acc.', 'Symptoms F1', 'Places F1', 'Time F1', 'Valid JSON'], rr.map((x, i) => ({ name: x.name.replace(' (regular expressions)', '').replace(', prompted (100 reports)', ''), color: ['var(--accent)', 'var(--faint)', 'var(--code)', 'var(--forensic)'][i] || 'var(--faint)', bold: x.ours,
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
  t.append(h('tr', {}, h('th', {}, ablation ? 'Variant' : 'Method'), ...['Correct digs', 'Within 100 m', 'Median error', 'False digs', 'Unsafe', 'Tool-call acc.', 'Rule viol.', 'Steps', ablation ? 'Cases' : 'vs crew (McNemar)'].map(x => h('th', {}, x))));
  const best = (k, lo) => { const vals = rows.map(r => r[k] && typeof r[k] === 'object' ? r[k][0] : r[k]).filter(x => x != null); return lo ? Math.min(...vals) : Math.max(...vals); };
  const bc = best('correct'), bw = best('within100'), bm = best('median_m', true), bf = best('false_digs', true), bv = best('violations', true);
  for (const r of rows) {
    const ours = r.ours || r.key === 'full';
    const c = r.correct && r.correct[0];
    t.append(h('tr', { class: ours ? 'ours' : '' },
      h('td', {}, h('span', {}, r.key === 'full' ? r.name : (ablation && !r.ours ? `− ${r.name}` : r.name)), r.what && r.what !== '-' ? h('span', { class: 'what' }, r.what) : null, h('span', { class: 'what mono', style: { fontSize: '10px' } }, (r.file || '').replace('logs/evaluation/', ''))),
      h('td', { class: c === bc ? 'best' : '' }, v(c, 1, '%')), h('td', { class: r.within100 === bw ? 'best' : '' }, v(r.within100, 1, '%')),
      h('td', { class: r.median_m === bm ? 'best' : '' }, v(r.median_m, 0, ' m')), h('td', { class: r.false_digs === bf ? 'best' : '' }, v(r.false_digs, 1, '%')),
      h('td', {}, v(r.unsafe, 1, '%')), h('td', {}, v(r.tool_acc, 1, '%')), h('td', { class: r.violations === bv ? 'best' : '' }, v(r.violations, 1, '%')), h('td', {}, v(r.steps, 1)),
      h('td', { class: 'small' }, ablation ? `${r.n}` : r.mcnemar ? `${r.mcnemar.only_ours} vs ${r.mcnemar.only_theirs} · p ${r.mcnemar.p < 0.001 ? '< 0.001' : r.mcnemar.p.toFixed(3)}` : '—')));
  }
  return t;
}

function sensorsTable(sensors) {
  const t = h('table', { class: 'res' });
  t.append(h('tr', {}, h('th', {}, 'Method'), ...['3', '5', '10'].flatMap(k => [h('th', {}, `${k} sens. · correct`), h('th', {}, 'median err.')])));
  for (const s of sensors) t.append(h('tr', { class: s.ours ? 'ours' : '' }, h('td', {}, s.name),
    ...['3', '5', '10'].flatMap(k => { const x = s.by_k[k] || {}; return [h('td', {}, x.correct ? v(x.correct[0], 1, '%') : '–'), h('td', {}, v(x.median_m, 0, ' m'))]; })));
  return t;
}

function suiteTable(R) {
  const t = h('table', { class: 'res heat-table' });
  const methods = R.main;
  t.append(h('tr', {}, h('th', {}, 'Case type'), h('th', {}, 'Cases'), ...methods.map(m => h('th', { style: { textAlign: 'center' } }, m.name))));
  for (const s of R.suites) {
    t.append(h('tr', {}, h('td', {}, h('b', {}, s.name), h('span', { class: 'what' }, s.blurb)), h('td', {}, `${s.n}`),
      ...methods.map(m => {
        const [ok, n] = (m.suites || {})[s.id] || [0, 0];
        const r = n ? ok / n : 0;
        const bg = `color-mix(in srgb, ${r >= 0.75 ? 'var(--good)' : r >= 0.4 ? 'var(--warn)' : 'var(--bad)'} ${Math.round(12 + r * 30)}%, transparent)`;
        return h('td', { style: { background: n ? bg : '' } }, n ? `${ok}/${n}` : '–');
      })));
  }
  return t;
}
