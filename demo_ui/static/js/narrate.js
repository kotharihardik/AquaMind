// Turns investigation events into (1) one-line captions for the audience and (2) War Room feed cards.
import { h, esc, pct, num, ROLES, CAUSE, avatar, clockLabel, modal } from './ui.js';
import { icon } from './icons.js';

const short = (s, n = 230) => { s = String(s || ''); return s.length > n ? s.slice(0, n - 1).trimEnd() + '…' : s; };
const node = (net, j) => (j == null ? '?' : net.describe[j]);
const ridList = xs => (xs || []).slice(0, 4).join(', ') + ((xs || []).length > 4 ? '…' : '');

function stepSummary(r) {
  const lit = r.clearly_lit || [], sug = r.suggestive || [], vf = r.valve_failed || [];
  const ex = r.excess_flow_lps || {};
  const secs = Object.keys(ex);
  if (lit.length) return `<b>${lit.join(', ')} lit</b> (+${num(ex[lit[0]], 1)} L/s extra night flow)` + (vf.length ? `; valve failed in ${vf.join(', ')}` : '');
  if (sug.length) return `weak extra flow in <b>${sug.join(', ')}</b> (suggestive, not lit)`;
  return `${secs.length ? secs.join(', ') : 'sections'} <b>cleared</b> (no extra night flow)` + (vf.length ? `; valve failed in ${vf.join(', ')}` : '');
}

// ------------------------------------------------------------------ captions
export function caption(ev, ctx) {
  const net = ctx.net, s = ev.state || {};
  const R = k => ROLES[k] || ROLES.code;
  switch (ev.kind) {
    case 'meta': return { who: 'code', text: `Loading case <b>${esc(ev.meta.case)}</b>…` };
    case 'messages': {
      const ids = (ev.messages || []).map(m => m.rid);
      return { who: 'reader', text: `<b>${ids.length} new message${ids.length > 1 ? 's' : ''}</b> arrived (${esc(ridList(ids))}). The fine-tuned reader turns each one into fields; the agents never see raw text.` };
    }
    case 'thinking': return { who: ev.role, text: `<b>${R(ev.role).name}</b> is thinking… <span class="muted">(${esc(short(ev.why, 90))})</span>` };
    case 'turn': {
      const d = ev.decision || {};
      if (ev.error) return { who: ev.role, text: `<b>${R(ev.role).name}</b>: no valid answer (${esc(short(ev.error, 80))}) — code carries on safely.` };
      if (ev.role === 'skeptic' && (d.challenges || []).length) {
        const c = d.challenges[0];
        return { who: 'skeptic', text: `<b>Critical Skeptic</b> attacks <b>${esc(c.target)}</b>: “${esc(short(c.contradiction, 200))}”` };
      }
      if (ev.role === 'strategist') {
        const o = (ev.options || []).find(o => o.id === String(d.choose || '').toUpperCase());
        return { who: 'strategist', text: `<b>Hydraulic Strategist</b> picks <b>${esc(d.choose || 'its own check')}</b>${o ? ': ' + esc(short(o.label.split(' | ')[0], 110)) + (o.chance != null ? ` (chance it finds the leak <b>${pct(o.chance)}</b>)` : '') : ''}. “${esc(short(ev.thought, 120))}”` };
      }
      if (ev.role === 'auditor') {
        const v = String(d.verdict || '').toUpperCase();
        return { who: 'auditor', text: `<b>Safety & SOP Auditor</b>: <b>${esc(v || '—')}</b>. ${(d.sop_actions || []).length ? 'SOP paperwork: ' + esc((d.sop_actions || []).join(', ')) + '. ' : ''}“${esc(short(ev.thought, 120))}”` };
      }
      return { who: ev.role, text: `<b>${R(ev.role).name}</b>: “${esc(short(ev.thought, 240))}”` };
    }
    case 'turn_done': {
      const rec = ev.record || {};
      if (rec.override) return { who: 'code', text: `<b>Code chance guard</b>: ${esc(rec.override.chosen)} (${pct(rec.override.chance)}) is far below the best option, so ${esc(rec.override.best)} (${pct(rec.override.best_chance)}) runs instead.` };
      if (ev.role === 'forensic' && (ev.slate || []).length) {
        const hs = ev.slate.map(x => `${x.id} ${x.kind === 'network_leak' ? 'leak at ' + esc(short(x.where, 28)) : x.kind === 'no_leak' ? 'no leak' : esc(short(x.cause || 'non-network', 24))} (${pct(x.mass)})`);
        return { who: 'forensic', text: `Hypotheses scored by code: ${hs.join(' · ')}. Every junction they name is simulated on the utility's model.` };
      }
      if (rec.note) return { who: 'code', text: esc(rec.note) };
      return null;
    }
    case 'tool': {
      const r = ev.result || {}, a = ev.args || {};
      switch (ev.tool) {
        case 'open': {
          const n = (ev.messages || []).length;
          const top = (ev.sensors || []).slice().sort((x, y) => (y.z || 0) - (x.z || 0))[0];
          return { who: 'code', text: `Case opened by <b>${ctx.meta.trigger === 'citizen_message' ? 'a citizen message' : 'a SCADA alarm'}</b>${ctx.meta.trigger_report ? ' (' + esc(ctx.meta.trigger_report) + ')' : ''}. <b>${n} messages</b> so far, only <b>${ctx.meta.k} of 33</b> sensor places are live${top ? `; strongest drop at ${esc(top.node)} (z ${num(top.z, 1)})` : ''}. First leak map: P(leak) <b>${pct(s.p_leak)}</b>.` };
        }
        case 'call_back': {
          const rp = (ev.replies || [])[0];
          if (!rp) return { who: 'forensic', text: 'Calls people back.' };
          const p = rp.parsed || {};
          const learned = p.not_network ? `not a utility pipe: <b>${esc(CAUSE[p.not_network] || p.not_network)}</b>` : p.source && p.source !== 'unknown' ? `water ${p.source === 'here' ? 'comes out right there' : 'only collects there'}` : p.still_flowing === false ? 'the water has stopped' : p.answered ? 'answered' : 'no answer';
          return { who: 'citizen', text: `The crew calls <b>${esc(rp.rid)}</b> with its own question${(ev.replies || []).length > 1 ? ` (and ${ev.replies.length - 1} more)` : ''}${rp.text ? `: “${esc(short(rp.text, 120))}”` : ''} → reader: ${learned}.` };
        }
        case 'use_record': return r.ok ? { who: 'forensic', text: `Town archive (RAG): <b>${esc(a.record_id)}</b> shows a repeat non-utility cause at ${esc(a.report_id)}'s place → counted as a likely repeat false alarm.` }
          : { who: 'code', text: `Code rejects the citation <b>${esc(a.record_id)}</b>: ${esc(r.error)}.` };
        case 'use_repair_record': return r.ok ? { who: 'forensic', text: `Repair history (RAG): <b>${esc(a.record_id)}</b> — an old main that keeps failing ${esc(r.where || '')}. Code raises the leak prior there ×4${r.hypothesised ? ` and simulates ${r.hypothesised} junctions` : ''}.` }
          : { who: 'code', text: `Code rejects <b>${esc(a.record_id)}</b> as a failing main: ${esc(r.error)}.` };
        case 'triangulate': return r.ok ? { who: 'forensic', text: `Triangulation <b>verified by code</b>: ${esc(r.msg)} — one leak, several witnesses on the slope.` }
          : { who: 'code', text: `Triangulation <b>rejected by code</b>: ${esc(r.msg)} (water cannot run uphill).` };
        case 'field_check': {
          if (!r.ok) return { who: 'code', text: `Field check refused: ${esc(r.error)}` };
          if (r.kind === 'step test') return { who: 'field', text: `Night step test (01:00–05:00): ${stepSummary(r)}.` };
          if (r.kind === 'listening') return { who: 'field', text: r.heard ? `Listening survey: <b>leak noise pinpointed</b> ${esc(r.pinpointed || '')}.` : `Listening survey ${esc(r.where || '')}: <b>quiet</b>${r.detection_power ? ` (detection power only ${esc(r.detection_power)} for a leak this small)` : ''}.` };
          return { who: 'field', text: `Field check: ${esc(r.kind)}` };
        }
        case 'propose_dig': {
          if (!r.ok) return { who: 'code', text: `<b>Code gate refused the dig</b>: ${esc(short(r.error, 160))}` };
          const tok = (ev.env || []).find(e => e.action === 'request_approval' && e.args && e.args.action === 'dig');
          return { who: 'field', text: `Gate open → operator token <span class="token">${esc((tok && tok.result.token) || '••••')}</span> → crew digs ${esc(r.spot || '')}: <b>${r.leak_found ? 'LEAK FOUND' : 'dry hole'}</b>${r.leak_found ? '' : ' — the miss becomes evidence'}.` };
        }
        case 'close': return { who: 'code', text: `Case closed: <b>${a.verdict === 'no_leak' ? 'no network leak' : 'handed over to the supervisor'}</b>${a.summary ? ' — ' + esc(short(a.summary, 140)) : ''}.` };
        // ReAct baseline tools
        case 'request_approval': return { who: 'operator', text: `ReAct asks the operator to approve a <b>${esc(a.action)}</b> at ${esc(short(Array.isArray(a.target) ? a.target.join(', ') : a.target, 60))} → ${r.approved ? 'approved' : 'refused: ' + esc(r.reason || r.error || '')}.` };
        case 'step_test': return r.ok ? { who: 'field', text: `ReAct's night step test of ${esc((a.sections || []).join(', '))}.` } : { who: 'code', text: `<b>Unapproved action</b>: ReAct's step test is refused by the field — ${esc(r.error)}.` };
        case 'dig': return r.ok ? { who: 'field', text: `ReAct digs ${esc(a.place)} → <b>${r.leak_found ? 'LEAK FOUND' : 'dry hole'}</b>.` } : { who: 'code', text: `ReAct's dig refused: ${esc(r.error)}` };
        case 'listen': return { who: 'field', text: `ReAct listens at ${esc(a.place)} → ${r.heard ? 'noise heard' : 'quiet'}.` };
        case 'test_places': return { who: 'code', text: `ReAct simulates leaks at ${esc(ridList(a.places))} → best fit ${esc(short((r.best_spots || [])[0], 80))}.` };
        case 'search_docs': return { who: 'react', text: `ReAct searches the documents: “${esc(short(a.query, 70))}”.` };
        case 'call': return { who: 'citizen', text: `ReAct calls ${esc(a.report_id)} → ${esc(short(r.reply, 100))}` };
        case 'place_logger': return { who: 'field', text: `ReAct places a pressure logger at ${esc(a.place)}.` };
        default: return { who: 'code', text: `${esc(ev.tool)}` };
      }
    }
    case 'end': {
      const r = ev.result || {}, T = ev.truth || {};
      if (!T.has_leak) return { who: 'code', text: r.n_digs ? `There was <b>no leak</b> in this case — but ${r.n_digs} crew${r.n_digs > 1 ? 's were' : ' was'} sent.` : `Ground truth: there was <b>no leak</b>. Closing without a dig is the right answer (${esc(r.outcome)}).` };
      return { who: 'code', text: r.found ? `Ground truth revealed: a <b>${num(T.leak.lps, 1)} L/s</b> leak on pipe ${esc(T.leak.pipe)} — the dig was <b>${num(r.guess_error_m, 0)} m</b> away. <b>Found.</b>` :
        `Ground truth revealed: a <b>${num(T.leak.lps, 1)} L/s</b> leak; final guess ${r.guess_error_m != null ? num(r.guess_error_m, 0) + ' m' : '—'} away. Not dug (${esc(r.outcome)}).` };
    }
  }
  return null;
}

// ------------------------------------------------------------------ feed cards
export const CAPS = { ft: ['Fine-tuning', 'var(--reader)'], rag: ['RAG', 'var(--operator)'], rs: ['Reasoning', 'var(--forensic)'], ag: ['Agentic', 'var(--strategist)'], tc: ['Tool calling', 'var(--auditor)'], sf: ['Safety', 'var(--good)'] };
function capTags(caps) {
  return (caps || []).map(k => h('span', { class: 'cap-tag', style: { color: CAPS[k][1], borderColor: `color-mix(in srgb, ${CAPS[k][1]} 45%, transparent)` } }, CAPS[k][0]));
}
function card(role, title, whenTxt, body, { sys = false, why = null, caps = null } = {}) {
  const r = ROLES[role] || ROLES.code;
  const el = h('div', { class: `item${sys ? ' sys' : ''}` }, avatar(role),
    h('div', { class: 'bd' },
      h('div', { class: 'ih' }, h('span', { class: 'who', style: { color: r.color } }, title), ...capTags(caps), h('span', { class: 'when' }, whenTxt || '')),
      why ? h('div', { class: 'why' }, why) : null,
      body));
  return el;
}

function whenOf(ev, ctx) { const s = ev.state || (ctx && ctx.lastState); return s ? `${clockLabel(s.t_h)}${s.clock ? ' · ' + s.clock : ''}` : ''; }

function stepBars(r) {
  const ex = r.excess_flow_lps || {}, lit = new Set(r.clearly_lit || []), sug = new Set(r.suggestive || []);
  const mx = Math.max(1, ...Object.values(ex).map(v => Math.abs(v)));
  return h('div', { class: 'stepbars' }, Object.entries(ex).map(([sid, v]) =>
    h('div', { class: `stepbar${lit.has(sid) ? ' lit' : ''}` }, h('span', { class: 'mono' }, sid),
      h('div', { class: 'bar' }, h('i', { style: { width: `${Math.max(2, 100 * Math.max(0, v) / mx)}%` } })),
      h('span', { class: 'mono', style: { textAlign: 'right' } }, `${v >= 0 ? '+' : ''}${num(v, 2)} L/s${lit.has(sid) ? ' ●' : sug.has(sid) ? ' ◐' : ''}`))),
    ...(r.valve_failed || []).map(sid => h('div', { class: 'stepbar' }, h('span', { class: 'mono' }, sid), h('span', { class: 'small', style: { color: 'var(--warn)' } }, 'valve would not close — not tested'), h('span'))));
}

function chips(p) {
  const out = [];
  if (!p) return out;
  if (p.answered === false) out.push(h('span', { class: 'chip' }, 'no answer'));
  if (p.not_network) out.push(h('span', { class: 'chip bad' }, `not network: ${CAUSE[p.not_network] || p.not_network}`));
  if (p.source && p.source !== 'unknown') out.push(h('span', { class: 'chip accent' }, `source: ${p.source}`));
  if (p.still_flowing != null) out.push(h('span', { class: `chip ${p.still_flowing ? 'warn' : ''}` }, p.still_flowing ? 'still flowing' : 'has stopped'));
  if ((p.places || []).length) out.push(h('span', { class: 'chip' }, `place: ${p.places.join(', ')}`));
  if ((p.flows_from || []).length) out.push(h('span', { class: 'chip accent' }, `flows from: ${p.flows_from.join(', ')}`));
  if (p.contains_instructions) out.push(h('span', { class: 'chip bad' }, 'instruction-like text'));
  return out;
}

function boardButton(ev) {
  if (!ev.board_text) return null;
  return h('span', { class: 'linkish', onclick: () => modal(`What the ${ROLES[ev.role]?.name || 'agent'} read (code-built board, no raw citizen text)`,
    h('pre', { class: 'board' }, ev.board_text), { wide: true }) }, h('span', { html: icon('eye', 13) }), ' what this agent saw');
}

export function makeCard(ev, ctx) {
  const net = ctx.net;
  if (ev.state) ctx.lastState = ev.state;
  if (ev.kind === 'turn') {
    const role = ev.role, d = ev.decision || {}, R = ROLES[role] || ROLES.code;
    const sub = h('div', { class: 'sub' });
    const badges = h('div', { class: 'row wrap', style: { marginTop: '6px', gap: '6px' } },
      ev.cached === true ? h('span', { class: 'chip mono' }, 'cached answer') : ev.cached === false ? h('span', { class: 'chip warn mono', title: 'answered by the LLM API (not from the cache) when this was recorded' }, `LLM call · ${num(ev.api_s || ev.latency_s, 1)} s`) : null,
      boardButton(ev));
    const body = h('div', {}, h('div', { class: 'th' }, ev.error ? h('span', { class: 'muted' }, `No valid answer (${ev.error}); code continues safely.`) : (ev.thought || '')), sub, badges);
    if (role === 'forensic') {
      let lastQ = null;
      for (const q of d.questions || []) {
        const same = q.question === lastQ;
        sub.append(h('div', { class: 'q' }, h('span', { class: 'to' }, `→ ${q.report_id}  `), same ? h('span', { class: 'muted' }, '(same question)') : q.question));
        lastQ = q.question;
      }
      const hs = d.hypotheses || [];
      if (hs.length) sub.append(h('div', { class: 'small muted', style: { marginTop: '2px' } }, 'Hypotheses: ' + hs.map(x => `${x.id} ${x.kind === 'network_leak' ? 'leak at ' + x.where : x.kind === 'no_leak' ? 'no leak' : (x.cause || 'non-network') + (x.explains ? ' behind ' + (x.explains || []).join(', ') : '')}`).join(' · ')));
      if (d.stance && d.stance !== 'continue') sub.append(h('span', { class: 'chip warn' }, `stance: ${d.stance}`));
    } else if (role === 'skeptic') {
      for (const c of d.challenges || []) sub.append(h('div', { class: 'chal' }, h('span', { class: 'tg' }, `against ${c.target}: `), c.contradiction,
        h('div', { class: 'row wrap', style: { marginTop: '4px', gap: '4px' } }, (c.evidence || []).map(e => h('span', { class: 'chip mono' }, e)), c.check && c.check !== 'none' ? h('span', { class: 'chip accent' }, `suggests ${c.check}`) : null)));
      if (d.verdict) sub.append(h('span', { class: `chip ${d.verdict === 'weak' ? 'bad' : 'good'}` }, `verdict: case ${d.verdict}`));
    } else if (role === 'strategist') {
      const opts = ev.options || [];
      const pick = String(d.choose || '').toUpperCase();
      const mx = Math.max(0.01, ...opts.map(o => o.chance || 0));
      for (const o of opts) sub.append(h('div', { class: `opt${o.id === pick ? ' pick' : ''}` }, h('span', { class: 'oid' }, o.id),
        h('div', {}, h('div', { class: 'small' }, o.label.split(' | ')[0]), h('div', { class: 'bar', style: { marginTop: '3px' } }, h('i', { style: { width: `${100 * (o.chance || 0) / mx}%` } }))),
        h('span', { class: 'mono small', style: { textAlign: 'right' } }, pct(o.chance))));
      if (!opts.length && pick) sub.append(h('span', { class: 'chip accent' }, `chooses ${pick}`));
      if (d.custom) sub.append(h('span', { class: 'chip accent' }, `own check: ${JSON.stringify(d.custom)}`));
    } else if (role === 'auditor') {
      const v = String(d.verdict || '').toLowerCase();
      sub.append(h('div', { class: 'row wrap', style: { gap: '5px' } }, h('span', { class: `chip ${v === 'approve' ? 'good' : v === 'hold' ? 'bad' : 'warn'}` }, `verdict: ${v || '—'}`),
        ...(d.sop_actions || []).map(a => h('span', { class: 'chip accent mono' }, a))));
      for (const o of d.objections || []) sub.append(h('div', { class: 'verify note' }, h('span', { html: icon('warn', 13) }), o));
    } else if (role === 'react') {
      sub.append(h('span', { class: 'chip mono' }, `${d.tool || '?'} ${d.args ? JSON.stringify(d.args).slice(0, 120) : ''}`));
    }
    const caps = { forensic: ['ag', 'rs', 'tc'], skeptic: ['ag', 'rs'], strategist: ['ag', 'tc'], auditor: ['ag', 'rag', 'sf'], react: ['tc'] }[role] || [];
    const el = card(role, R.name, whenOf(ev, ctx), body, { why: ev.why, caps });
    el._sub = sub;
    ctx.openTurn = { role, el, sub };
    return el;
  }
  if (ev.kind === 'turn_done') {
    const t = ctx.openTurn;
    if (!t || t.role !== ev.role) return null;
    const rec = ev.record || {};
    const add = x => t.sub.append(x);
    if (ev.role === 'forensic') {
      for (const fm of rec.failing_mains || []) add(h('div', { class: `verify ${fm.ok ? 'ok' : 'no'}` }, h('span', { html: icon(fm.ok ? 'check' : 'x', 13) }), `${fm.id}: ${fm.ok ? 'failing main ' + (fm.where || '') + ' — prior raised, simulated' : 'rejected by code'}`));
      if (rec.triangulate) add(h('div', { class: `verify ${rec.triangulate.ok ? 'ok' : 'no'}` }, h('span', { html: icon(rec.triangulate.ok ? 'check' : 'x', 13) }), `triangulation: ${rec.triangulate.msg}`));
      for (const rj of ev.rejected || []) add(h('div', { class: 'verify no' }, h('span', { html: icon('x', 13) }), `${rj.citation}: ${rj.reason}`));
      const sl = (ev.slate || []).filter(x => x.mass != null);
      if (sl.length) {
        add(h('div', { class: 'small muted', style: { marginTop: '4px' } }, 'Code scores the hypotheses on the leak map:'));
        for (const x of sl) add(h('div', { class: 'hyp' }, h('span', { class: 'hid' }, x.id), h('div', {}, h('div', { class: 'small' }, x.kind === 'network_leak' ? `leak at ${x.where}${x.region ? ` · ${x.region.length} junctions simulated` : ''}` : x.kind === 'no_leak' ? 'no leak at all' : `non-network: ${x.cause || '?'}`), h('div', { class: 'bar', style: { marginTop: '3px' } }, h('i', { style: { width: `${100 * x.mass}%` } }))), h('span', { class: 'mono small', style: { textAlign: 'right' } }, pct(x.mass))));
      }
      if (rec.candidates_added || rec.witness_candidates) add(h('div', { class: 'small muted' }, `${(rec.candidates_added || 0) + (rec.witness_candidates || 0)} new junctions simulated on the utility's EPANET model (WNTR)`));
    } else if (ev.role === 'strategist') {
      if (rec.override) add(h('div', { class: 'verify note' }, h('span', { html: icon('cpu', 13) }), `Code chance guard: ${rec.override.chosen} (${pct(rec.override.chance)}) → ${rec.override.best} (${pct(rec.override.best_chance)})`));
      else if (rec.choose) add(h('div', { class: 'verify ok' }, h('span', { html: icon('check', 13) }), `${rec.choose} accepted by code${rec.discriminates != null ? ` · separates the hypotheses ${pct(rec.discriminates)}` : ''}`));
      if (rec.note) add(h('div', { class: 'verify note' }, h('span', { html: icon('info', 13) }), rec.note));
    } else if (ev.role === 'auditor') {
      if (rec.note) add(h('div', { class: 'verify note' }, h('span', { html: icon('cpu', 13) }), rec.note));
      if (rec.ran) add(h('div', { class: 'verify note' }, h('span', { html: icon('info', 13) }), `objection honoured: one more check (${rec.ran}) runs before any dig`));
    } else if (ev.role === 'skeptic') {
      const bad = (ev.challenges || []).reduce((a, c) => a + (c.cited_bad || 0), 0);
      if (bad) add(h('div', { class: 'verify note' }, h('span', { html: icon('info', 13) }), `${bad} cited evidence id(s) do not exist on the board`));
    }
    return null;
  }
  if (ev.kind === 'messages') {
    const ids = (ev.messages || []).map(m => m.rid);
    return card('reader', 'New messages', whenOf(ev, ctx), h('div', { class: 'th' }, `${ids.join(', ')} arrived and were read by the fine-tuned reader.`), { sys: true, caps: ['ft'] });
  }
  if (ev.kind === 'end') {
    const r = ev.result || {}, T = ev.truth || {};
    const good = T.has_leak ? r.found : !r.n_digs;
    return card('code', 'Case finished', whenOf(ev, ctx), h('div', { class: `outcome-banner ${good ? 'good' : r.n_digs && !r.found ? 'bad' : 'warn'}`, html:
      `${icon(good ? 'check' : 'warn', 16)} ${T.has_leak ? (r.found ? `Leak found ${num(r.guess_error_m, 0)} m from the true leak` : `Leak not dug (${esc(r.outcome)}); guess ${r.guess_error_m != null ? num(r.guess_error_m, 0) + ' m' : '—'} away`) : (r.n_digs ? 'No leak, but a crew was sent' : 'No leak — closed without a dig')}` }), { sys: true });
  }
  if (ev.kind !== 'tool') return null;
  const r = ev.result || {}, a = ev.args || {};
  const t = ctx.openTurn;
  // the Forensic Advocate's own actions go into its card
  if (t && t.role === 'forensic' && ['use_record', 'use_repair_record', 'triangulate', 'call_back'].includes(ev.tool)) {
    if (ev.tool === 'call_back') {
      for (const rp of ev.replies || []) {
        t.sub.append(h('div', { class: 'reply' }, h('span', { html: icon('phone', 12) }), ` ${rp.rid}: `, rp.text ? `“${rp.text}”` : (rp.parsed && rp.parsed.answered === false ? 'no answer' : '(reply read by the reader)')));
        const c = chips(rp.parsed);
        if (c.length) t.sub.append(h('div', { class: 'row wrap', style: { gap: '4px', marginLeft: '10px' } }, h('span', { class: 'tiny muted' }, 'reader →'), ...c));
      }
      if ((r.new_messages || []).length) t.sub.append(h('div', { class: 'small muted' }, `new messages while calling: ${r.new_messages.join(', ')}`));
    } else if (ev.tool === 'use_record') {
      t.sub.append(h('div', { class: `verify ${r.ok ? 'ok' : 'no'}` }, h('span', { html: icon(r.ok ? 'check' : 'x', 13) }), `archive ${a.record_id} for ${a.report_id}: ${r.ok ? 'repeat false alarm at that place (verified)' : r.error}`));
    } else if (ev.tool === 'use_repair_record' && !r.ok) {
      t.sub.append(h('div', { class: 'verify no' }, h('span', { html: icon('x', 13) }), `${a.record_id}: ${r.error}`));
    }
    return null;
  }
  const when = whenOf(ev, ctx);
  switch (ev.tool) {
    case 'open': {
      const sens = (ev.sensors || []).slice().sort((x, y) => (y.z || 0) - (x.z || 0));
      return card('code', 'Case opened', when, h('div', {},
        h('div', { class: 'th' }, `${(ev.messages || []).length} messages · ${ctx.meta.k} of 33 sensor places live · P(leak) ${pct(ev.state && ev.state.p_leak)} · zone inflow ${ev.inflow ? (ev.inflow.q_hat >= 0 ? '+' : '') + num(ev.inflow.q_hat, 1) + ' L/s' : '–'}`),
        h('div', { class: 'row wrap', style: { gap: '4px', marginTop: '6px' } }, sens.map(x => h('span', { class: `chip mono ${Math.abs(x.z) >= 3 ? 'bad' : Math.abs(x.z) >= 1.5 ? 'warn' : ''}` }, `${x.node} ${x.drop_m >= 0 ? '↓' : '↑'}${num(Math.abs(x.drop_m), 2)} m · z ${num(x.z, 1)}`)))), { sys: true, caps: ['ft'] });
    }
    case 'field_check': {
      if (!r.ok) return card('code', 'Field check refused', when, h('div', { class: 'th' }, r.error), { sys: true });
      const appr = (ev.env || []).find(e => e.action === 'request_approval');
      const head = h('div', { class: 'th' }, h('b', {}, r.kind === 'step test' ? 'Night step test' : 'Listening survey'), ` · ${(a.label || '').split(' | ')[0]}`);
      const body = h('div', {}, head);
      if (r.kind === 'step test') {
        if (appr) body.append(h('div', { class: 'row', style: { gap: '6px', margin: '5px 0' } }, h('span', { class: 'small muted' }, 'operator approval'), h('span', { class: 'token' }, appr.result.token || 'refused')));
        body.append(stepBars(r));
      } else {
        body.append(h('div', { class: 'row wrap', style: { gap: '5px', marginTop: '5px' } }, h('span', { class: `chip ${r.heard ? 'good' : ''}` }, r.heard ? 'leak noise heard' : 'quiet'),
          r.pinpointed ? h('span', { class: 'chip good' }, `pinpointed ${r.pinpointed}`) : null, r.detection_power ? h('span', { class: 'chip' }, `detection power ${r.detection_power}`) : null));
      }
      body.append(h('div', { class: 'small muted', style: { marginTop: '5px' } }, `P(success) of the best dig now ${pct(r.p_success_now)}`));
      return card('field', 'Field crew', when, body, { sys: true, caps: ['tc'] });
    }
    case 'propose_dig': {
      if (!r.ok) return card('code', 'Code gate', when, h('div', { class: 'th' }, h('b', { style: { color: 'var(--bad)' } }, 'Dig refused. '), r.error), { sys: true, caps: ['sf'] });
      const env = ev.env || [];
      const body = h('div', { class: 'sub' });
      for (const e of env) {
        if (e.action === 'request_approval') {
          const cert = (e.args || {}).certificate || {};
          body.append(h('div', { class: 'small' }, h('b', {}, 'Operator approval'), ` for ${cert.spot || e.args.target} · P(success) ${pct(cert.p_success)}`),
            h('div', { class: 'row wrap', style: { gap: '4px' } }, (cert.evidence || []).map(x => h('span', { class: 'chip mono' }, x))),
            h('div', { class: 'row', style: { gap: '6px' } }, h('span', { class: 'small muted' }, 'one-time token, bound to this spot'), h('span', { class: 'token' }, e.result.token || 'refused')));
        } else if (e.action === 'notify') body.append(h('div', { class: 'verify ok' }, h('span', { html: icon('check', 13) }), `notified ${e.args.target}: ${e.args.message}`));
        else if (e.action === 'request_permit') body.append(h('div', { class: 'verify ok' }, h('span', { html: icon('check', 13) }), `municipal permit for ${e.args.street} (SOP-6)`));
        else if (e.action === 'dig') body.append(h('div', { class: `outcome-banner ${e.result.leak_found ? 'good' : 'bad'}`, html: `${icon('shovel', 16)} Dig at ${esc(r.spot || e.args.node)}${e.args.after_school_hours ? ' (after school hours)' : ''}: ${e.result.leak_found ? 'LEAK FOUND' : 'dry hole'}` }));
      }
      if (!env.length) body.append(h('div', { class: `outcome-banner ${r.leak_found ? 'good' : 'bad'}` }, `Dig ${r.spot || ''}: ${r.leak_found ? 'LEAK FOUND' : 'dry hole'}`));
      return card('field', 'Code gate open → dig', when, body, { sys: true, caps: ['sf', 'tc'] });
    }
    case 'close': return card('code', 'Case closed', when, h('div', { class: 'th' }, h('b', {}, a.verdict === 'no_leak' ? 'No network leak. ' : 'Handed over to the supervisor. '), a.summary || ''), { sys: true });
    case 'call_back': return card('citizen', 'Call-backs', when, h('div', { class: 'sub' }, (ev.replies || []).map(rp => h('div', { class: 'reply' }, `${rp.rid}: ${rp.text || 'answered'}`))), { sys: true });
    case 'use_record': case 'use_repair_record': case 'triangulate': {
      const c = caption(ev, ctx);
      return card(r.ok ? 'forensic' : 'code', ev.tool.replace(/_/g, ' '), when, h('div', { class: 'th', html: c ? c.text : '' }), { sys: true, caps: ev.tool === 'triangulate' ? ['rs'] : ['rag'] });
    }
    default: {
      const c = caption(ev, ctx);
      const bad = r.ok === false;
      return card(bad ? 'code' : (ev.tool === 'request_approval' ? 'operator' : ev.tool === 'dig' || ev.tool === 'listen' || ev.tool === 'step_test' ? 'field' : 'code'),
        ev.tool.replace(/_/g, ' '), when, h('div', { class: 'th', html: c ? c.text : '', style: bad ? { color: 'var(--bad)' } : {} }), { sys: true });
    }
  }
}

// ------------------------------------------------------------------ timing for the player (ms at 1x)
export function duration(ev) {
  switch (ev.kind) {
    case 'meta': return 200;
    case 'thinking': return 750;
    case 'turn': return Math.min(6500, 2600 + (String(ev.thought || '').length * 14));
    case 'turn_done': return 1100;
    case 'messages': return 1500;
    case 'end': return 1500;
    case 'tool':
      return { open: 2600, call_back: 3000, field_check: 3200, propose_dig: 3800, close: 2400, use_record: 900, use_repair_record: 900, triangulate: 1400 }[ev.tool] || 1900;
  }
  return 1000;
}
