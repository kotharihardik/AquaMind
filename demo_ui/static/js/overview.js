// Overview: the problem, the system, the five capabilities and the headline numbers, with links into the live views.
import { get } from './api.js';
import { NetworkMap } from './map.js';
import { h, esc, num, ROLES } from './ui.js';
import { icon } from './icons.js';

export async function initOverview(root, boot, go) {
  root.innerHTML = '';
  const R = await get('/api/results', { memo: true }).catch(() => null);
  const page = h('div', { class: 'page' });
  root.append(page);
  const crew = R && R.main.find(m => m.ours), react = R && R.main.find(m => m.key === 'react_agent'), prac = R && R.main.find(m => m.key === 'crew_practice');
  const net = boot.network;

  // hero
  const hero = h('div', { class: 'hero' });
  const mapBox = h('div', { style: { position: 'absolute', inset: '0 0 0 34%', pointerEvents: 'none' } });
  hero.append(mapBox, h('div', { class: 'hero-in' },
    h('span', { class: 'tag', html: `${icon('dropFill', 13)} AgentiX · LMA group project 2026` }),
    h('h1', {}, 'Where sensors are blind, ', h('span', { class: 'grad' }, 'a crew of AI agents'), ' finds the leak.'),
    h('p', { class: 'lead' }, `A town of ${net.stats.junctions} junctions watched by only 5 pressure sensors. Citizens notice what sensors miss — damp ground, a hiss at night, brown tap water — but their messages are vague, point downhill and are sometimes fake. AquaMind reads them with a fine-tuned local model, tests every hypothesis with physics in code, and lets four LLM agents investigate before anyone digs.`),
    h('div', { class: 'row', style: { gap: '10px', marginTop: '22px' } },
      h('button', { class: 'btn primary', onclick: () => go('mission'), html: `${icon('play', 15)} Watch an investigation` }),
      h('button', { class: 'btn', onclick: () => go('h2h'), html: `${icon('swords', 15)} Crew vs ReAct` }),
      h('button', { class: 'btn', onclick: () => go('results'), html: `${icon('chart', 15)} Results` })),
    h('div', { class: 'small muted', style: { marginTop: '18px' } }, 'Hardik Kothari · Gaurav Patel · Harshal Karangale · Rohit Rohon')));
  page.append(hero);
  const hm = new NetworkMap(mapBox, net, { interactive: false, compact: true, labels: false, sensorsAll: true });
  hm.setLayer('landmarks', false);
  try {
    const rec = boot.recordings && boot.recordings[0] ? await get(`/api/recording?file=${encodeURIComponent(boot.recordings[0].file)}`) : null;
    if (rec) {
      const evs = rec.events, last = [...evs].reverse().find(e => e.state && e.state.heat), end = evs.find(e => e.kind === 'end');
      const msgs = {}; const open = evs.find(e => e.tool === 'open');
      for (const m of (open && open.messages) || []) if (m.j != null) msgs[m.rid] = { j: m.j, eps: 0.5, usable: true };
      hm.setState({ heat: last && last.state.heat, p_leak: 1, sensors: rec.meta.sensors, msgs, truth: end && end.truth, cand: null });
    } else hm.setState({ sensors: net.layouts.spread_k5 || [] });
  } catch (e) { /* the hero art is optional */ }

  // KPIs
  if (crew) {
    page.append(h('div', { class: 'grid g5', style: { marginTop: '16px' } },
      kpi(`${crew.correct[0]}%`, 'correct digs', `leak cases dug within 100 m · ${crew.n} cases`, `ReAct ${react.correct[0]}% · crews today ${prac.correct[0]}%`, 'var(--accent)'),
      kpi(`${crew.median_m} m`, 'median error', 'from the final guess to the leak', `ReAct ${react.median_m} m`, 'var(--good)'),
      kpi(`${crew.unsafe}%`, 'unsafe actions', 'outside the rules or unapproved', `ReAct ${react.unsafe}%`, 'var(--good)'),
      kpi(`${crew.traps[0]}/${crew.traps[1]}`, 'trap cases solved', 'fake “approval given” messages', `ReAct ${react.traps[0]}/${react.traps[1]}`, 'var(--forensic)'),
      kpi(`${R.reader.test_real[0] ? (100 * R.reader.test_real[0].category).toFixed(1) : '–'}%`, 'reader accuracy', 'fine-tuned 3B on unseen real reports', `untuned ${R.reader.test_real[1] ? (100 * R.reader.test_real[1].category).toFixed(1) : '–'}%`, 'var(--reader)')));
  }

  // problem
  page.append(h('div', { class: 'eyebrow', style: { marginTop: '34px' } }, 'The problem'), h('h2', { style: { marginTop: '4px' } }, 'A small leak fits many places when sensors are few'),
    h('div', { class: 'grid g4' },
      fact('782', 'junctions in L-Town (BattLeDIM 2020)', 'wave'), fact('5 of 33', 'pressure sensors a typical utility can afford', 'gauge'),
      fact('0.7–8.5 L/s', 'leak sizes; more than half under 2 L/s', 'dropFill'), fact('5–36', 'citizen messages per case — vague, downhill, sometimes fake', 'chat')));

  // architecture
  page.append(h('div', { class: 'eyebrow', style: { marginTop: '34px' } }, 'The system'), h('h2', { style: { marginTop: '4px' } }, 'Code does every number and every safety check. The agents do the language and judgement.'),
    h('div', { class: 'card', style: { padding: '10px' }, html: archSVG() }));
  page.querySelectorAll('[data-go]').forEach(n => n.addEventListener('click', () => go(n.dataset.go)));

  // capabilities
  const cap = (ic, color, name, where, metric, mm, view) => h('div', { class: 'card cap' },
    h('div', { class: 'ic', style: { background: color }, html: icon(ic, 20) }), h('h3', { style: { marginTop: '6px' } }, name), h('div', { class: 'small muted' }, where),
    h('div', { class: 'm', style: { color } }, metric), h('div', { class: 'mm' }, mm),
    h('div', { class: 'go' }, h('button', { class: 'btn sm', onclick: () => go(view) }, 'See it ', h('span', { html: icon('next', 11) }))));
  const abl = k => R && R.ablations.find(a => a.key === k);
  page.append(h('div', { class: 'eyebrow', style: { marginTop: '34px' } }, 'Five capabilities, each with a job and a measurement'), h('h2', { style: { marginTop: '4px' } }, 'Fine-tuning · RAG · Reasoning · Agents · Tool calling'),
    h('div', { class: 'grid g5' },
      cap('chip', 'var(--reader)', 'Fine-tuning', 'Qwen2.5-3B + QLoRA reads raw messages into fixed fields, locally', R ? `${(100 * R.reader.test_real[0].category).toFixed(1)}%` : '91.6%', `category accuracy on real reports (untuned ${R ? (100 * R.reader.test_real[1].category).toFixed(1) : '65.2'}%)`, 'lab'),
      cap('book', 'var(--operator)', 'RAG', 'BM25 over the town archive (88 tickets) and the SOP (10 clauses)', abl('no_rag') ? `${crew.violations}%` : '3.8%', `cases breaking an SOP rule (${abl('no_rag') ? abl('no_rag').violations : 98.6}% without RAG)`, 'lab'),
      cap('search', 'var(--forensic)', 'Reasoning', 'Hypothesis slate, Skeptic debate, triangulation of witnesses', crew ? `${crew.correct[0]}%` : '74.6%', `correct digs step by step (one shot: ${abl('one_shot') ? abl('one_shot').correct[0] : 9.2}%)`, 'mission'),
      cap('radar', 'var(--strategist)', 'Agentic behaviour', 'Four agents, a shared evidence board, a fixed turn order', crew ? `${crew.correct[0]}%` : '74.6%', `with agents (code only: ${abl('no_ai') ? abl('no_ai').correct[0] : 35.1}%)`, 'h2h'),
      cap('key', 'var(--auditor)', 'Tool calling', 'Call-backs, step tests, listening, citations, a dig with no location argument', crew ? `${crew.unsafe}%` : '0%', `unsafe actions · tool-call accuracy ${crew ? crew.tool_acc : 91.2}%`, 'safety')));

  // guard rails
  page.append(h('div', { class: 'eyebrow', style: { marginTop: '34px' } }, 'Rules we keep'), h('div', { class: 'grid g4' },
    rail('cpu', 'The LLM never produces a number a decision depends on', 'Physics (WNTR), probabilities, chances of each check and both gates are code.'),
    rail('lock', 'Citizen text never controls a tool', 'Only the fine-tuned reader sees raw text; agents read code-built lines; instructions are quarantined.'),
    rail('shieldCheck', 'Every claim is checked by code', 'Archive citations, failing mains, triangulations and SOP actions are verified before they touch the board.'),
    rail('flask', 'A fair, honest benchmark', 'Truth runs on L-TOWN_Real, methods on the nominal model; 150 test cases frozen with a hash; same luck for every method.')));
}

function kpi(v, k, c, vs, color) {
  return h('div', { class: 'card kpi' }, h('div', { class: 'v', style: { color } }, v), h('div', { class: 'k' }, k), h('div', { class: 'c' }, c), vs ? h('span', { class: 'vs' }, vs) : null);
}
function fact(v, k, ic) {
  return h('div', { class: 'card', style: { display: 'flex', gap: '14px', alignItems: 'center' } }, h('div', { style: { color: 'var(--accent)' }, html: icon(ic, 28) }),
    h('div', {}, h('div', { style: { fontFamily: 'var(--display)', fontSize: '26px', fontWeight: 800, letterSpacing: '-.5px' } }, v), h('div', { class: 'small muted' }, k)));
}
function rail(ic, t, d) {
  return h('div', { class: 'card' }, h('div', { style: { color: 'var(--good)' }, html: icon(ic, 22) }), h('h3', { style: { marginTop: '8px' } }, t), h('div', { class: 'small muted' }, d));
}

function archSVG() {
  const box = (x, y, w, hh, title, sub, color, go, ic) => `
    <g class="node" ${go ? `data-go="${go}" style="cursor:pointer"` : ''}>
      <rect x="${x}" y="${y}" width="${w}" height="${hh}" rx="14" fill="var(--panel-2)" stroke="${color}" stroke-width="1.6"/>
      <rect x="${x}" y="${y}" width="6" height="${hh}" rx="3" fill="${color}"/>
      <text x="${x + 20}" y="${y + 26}" font-size="14.5" font-weight="750" fill="var(--text)">${esc(title)}</text>
      ${sub.map((s, i) => `<text x="${x + 20}" y="${y + 46 + i * 16}" font-size="11.5" fill="var(--muted)">${esc(s)}</text>`).join('')}
    </g>`;
  const agent = (x, y, role) => { const R = ROLES[role]; return `<g data-go="mission" style="cursor:pointer"><rect x="${x}" y="${y}" width="196" height="40" rx="11" fill="${R.hex}" opacity=".14" stroke="${R.hex}"/><circle cx="${x + 20}" cy="${y + 20}" r="11" fill="${R.hex}"/><text x="${x + 40}" y="${y + 25}" font-size="12.5" font-weight="700" fill="var(--text)">${R.name}</text></g>`; };
  const flow = (d, color = 'var(--accent)') => `<path class="flow" d="${d}" fill="none" stroke="${color}" stroke-width="2" marker-end="url(#ah)" opacity=".85"/>`;
  return `<svg class="arch" viewBox="0 0 1200 440">
    <defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" fill="var(--accent)"/></marker></defs>
    ${box(20, 40, 200, 92, 'Citizens', ['WhatsApp · SMS · app · calls', 'Hinglish, typos, wrong streets', 'some are fake orders'], 'var(--citizen)', 'lab')}
    ${box(20, 290, 200, 92, 'SCADA', ['5 of 33 pressure sensors', 'zone inflow + tank pump', 'a small leak barely shows'], 'var(--accent)', null)}
    ${box(262, 40, 210, 92, 'Fine-tuned reader', ['Qwen2.5-3B + QLoRA, local', 'raw text → reader-2.0 fields', 'flags instructions'], 'var(--reader)', 'lab')}
    ${box(420, 190, 300, 112, 'Evidence board (code)', ['WNTR physics on every hypothesised', 'junction · Bayesian leak map over 782', 'junctions · P(leak) · P(success)', 'value of each field check'], 'var(--code)', 'mission')}
    <rect x="762" y="14" width="420" height="214" rx="16" fill="none" stroke="var(--line-2)" stroke-dasharray="5 5"/>
    <text x="780" y="38" font-size="11" font-weight="800" letter-spacing="1.4" fill="var(--muted)">INVESTIGATION CREW · 4 LLM AGENTS</text>
    ${agent(780, 52, 'forensic')}${agent(980, 52, 'skeptic')}${agent(780, 104, 'strategist')}${agent(980, 104, 'auditor')}
    <text x="780" y="172" font-size="11.5" fill="var(--muted)">questions to callers · hypotheses · attacks · next check · SOP</text>
    <text x="780" y="190" font-size="11.5" fill="var(--muted)">JSON only · every claim verified by code · cannot open a gate</text>
    ${box(780, 252, 190, 76, 'RAG (BM25)', ['town archive: 88 tickets', 'SOP: 10 clauses'], 'var(--operator)', 'lab')}
    ${box(420, 340, 190, 84, 'Code gate', ['P(success) ≥ 0.80', '+ a field check supports it'], 'var(--good)', 'safety')}
    ${box(640, 352, 170, 72, 'Operator token', ['one-time, bound to', 'this action and spot'], 'var(--operator)', 'safety')}
    ${box(840, 352, 340, 72, 'Field crew', ['night step test · listening survey · dig', 'max 5 checks and 2 digs per case'], 'var(--field)', 'mission')}
    ${flow('M220,86 L258,86')}
    ${flow('M472,86 C560,86 560,190 570,190')}
    ${flow('M220,336 C320,336 330,262 416,262')}
    ${flow('M720,220 C740,220 750,140 776,140', 'var(--forensic)')}
    ${flow('M776,96 C700,96 700,200 720,205', 'var(--forensic)')}
    ${flow('M875,252 L875,232', 'var(--operator)')}
    ${flow('M515,302 L515,336', 'var(--good)')}
    ${flow('M610,388 L636,388', 'var(--operator)')}
    ${flow('M810,388 L836,388', 'var(--field)')}
    ${flow('M1000,350 C975,338 770,338 724,296', 'var(--field)')}
    <text x="988" y="342" font-size="11" fill="var(--field)">field results become evidence</text>
  </svg>`;
}
