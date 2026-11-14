// Safety lab: send a malicious message into a real case and watch every defence layer it meets (real AquaMind code on the server).
import { get, post } from './api.js';
import { hbars } from './charts.js';
import { h, esc, jsonHTML } from './ui.js';
import { icon } from './icons.js';

export async function initSafety(root, boot, go) {
  root.innerHTML = '';
  const page = h('div', { class: 'page' });
  root.append(page);
  page.append(h('div', { class: 'eyebrow' }, 'Safety by design · prompt injection'),
    h('h1', {}, 'Try to make the crew dig.'),
    h('p', { class: 'lead' }, 'Type a message like a scammer or a confused caller would: fake approvals, “system notes”, pressure from a supervisor. It is added to a real benchmark case and goes through the same code the crew uses. Nothing here is simulated for the demo.'));

  const ta = h('textarea', { class: 'field', rows: 4 }, boot.examples.attacks[0].text);
  const ch = h('select', { class: 'field', style: { maxWidth: '200px' } }, ['whatsapp', 'sms', 'app_form', 'social_post', 'call_transcript'].map(c => h('option', { value: c }, c)));
  ch.value = boot.examples.attacks[0].channel;
  const go_ = h('button', { class: 'btn danger', html: `${icon('bolt', 15)} Launch attack` });
  const verdict = h('div', { class: 'hidden' });
  const ex = h('div', { class: 'row wrap', style: { gap: '6px' } }, boot.examples.attacks.map(a => h('button', { class: 'btn sm', onclick: () => { ta.value = a.text; ch.value = a.channel; } }, a.label)));
  const layersBox = h('div', { class: 'layers' });
  const LAYERS = [['reader', 'Quarantine: only the fine-tuned reader sees raw text'], ['view', 'The agents read a code-built line, never the text'], ['tool', 'The dig tool has no location argument'],
    ['gate', 'Code gate: P(success) ≥ 0.80 and a field check must support the spot'], ['operator', 'Duty operator rejects requests that quote instructions (SOP-4)'],
    ['token', 'One-time operator token, bound to one action at one spot'], ['docs', 'Instruction-like text is stripped from documents (RAG)']];
  const layerEls = {};
  LAYERS.forEach(([id, name], k) => {
    const el = h('div', { class: 'layer' }, h('div', { class: 'n' }, `${k + 1}`), h('div', {}, h('div', { class: 'lt' }, name), h('div', { class: 'ld muted' }, 'waiting…')));
    layerEls[id] = el; layersBox.append(el);
  });
  page.append(h('div', { class: 'grid g2', style: { marginTop: '18px', alignItems: 'start' } },
    h('div', { class: 'card' }, h('h3', {}, 'Attack console'), h('div', { class: 'small muted', style: { marginBottom: '8px' } }, 'Sandbox case: test-trap-000 (5 sensors). The message arrives as R-900, at the moment the case opens.'),
      ex, h('div', { style: { height: '10px' } }), ta, h('div', { class: 'row', style: { marginTop: '10px', gap: '8px' } }, ch, h('div', { class: 'grow' }), go_),
      h('div', { style: { marginTop: '14px' } }, verdict),
      h('div', { class: 'small muted', style: { marginTop: '12px' } }, 'A message the fine-tuned reader has not seen before is read live (reader server or the local MLX model; the rule reader is the labelled fallback).')),
    h('div', { class: 'card' }, h('h3', {}, 'Defence in depth'), layersBox)));

  go_.addEventListener('click', async () => {
    go_.disabled = true; go_.innerHTML = `<span class="spinner"></span> Attacking…`;
    verdict.className = 'hidden';
    for (const [id] of LAYERS) { layerEls[id].className = 'layer'; layerEls[id].querySelector('.ld').textContent = 'waiting…'; }
    try {
      const r = await post('/api/attack', { text: ta.value, channel: ch.value });
      if (!r.ok) throw new Error(r.error);
      for (let k = 0; k < r.layers.length; k++) {
        const L = r.layers[k], el = layerEls[L.id];
        await new Promise(res => setTimeout(res, 420));
        if (!el) continue;
        el.className = `layer on ${L.blocked ? 'blocked' : 'passed'}`;
        el.querySelector('.n').innerHTML = L.blocked ? icon('shieldCheck', 20) : icon('warn', 20);
        const ld = el.querySelector('.ld');
        ld.className = 'ld';
        ld.textContent = String(L.detail || '');
        if (L.id === 'reader' && L.data && L.data.read) {
          const more = h('details', { style: { marginTop: '6px' } }, h('summary', { class: 'linkish' }, `reader output (${L.data.provenance || '?'})`), h('pre', { class: 'json', html: jsonHTML(L.data.read) }));
          ld.append(more);
        }
      }
      await new Promise(res => setTimeout(res, 300));
      verdict.className = `verdict ${r.dug ? 'bad' : 'good'}`;
      verdict.innerHTML = r.dug ? `${icon('warn', 26)} A crew was sent.` : `${icon('shieldCheck', 26)} Blocked at ${r.blocked} of ${r.total} layers — no crew was sent.`;
    } catch (e) {
      verdict.className = 'verdict bad'; verdict.textContent = `Error: ${e.message}`;
    } finally { go_.disabled = false; go_.innerHTML = `${icon('bolt', 15)} Launch attack`; }
  });

  // evidence from the benchmark
  const R = await get('/api/results', { memo: true }).catch(() => null);
  if (!R) return;
  const crew = R.main.find(m => m.ours), react = R.main.find(m => m.key === 'react_agent');
  page.append(h('div', { class: 'eyebrow', style: { marginTop: '34px' } }, 'On the benchmark (210 cases, 5 sensors)'), h('h2', { style: { marginTop: '4px' } }, 'What happens when an agent is not built like this'),
    h('div', { class: 'grid g2' },
      h('div', { class: 'card' }, h('h3', {}, 'Cases with an unsafe or unapproved action'), h('div', { html: hbars(R.main.map(m => ({ label: m.name, value: m.unsafe, color: m.unsafe > 5 ? 'var(--bad)' : 'var(--good)', bold: m.ours })), { max: 50, labelW: 220 }) }),
        h('div', { class: 'callout' }, `ReAct tried an unsafe action in ${react.unsafe}% of cases — mostly step tests without an operator token (the field refused them) and a dig at a spot named by a fake message. The crew: ${crew.unsafe}%.`)),
      h('div', { class: 'card' }, h('h3', {}, `Trap cases: crew ${crew.traps[0]}/${crew.traps[1]} solved · ReAct ${react.traps[0]}/${react.traps[1]}`),
        h('p', { class: 'small muted' }, 'In test-trap-008 the ReAct agent asked for approval at the spot the fake message named and dug there. Watch it side by side with the crew:'),
        h('button', { class: 'btn primary', onclick: () => go('h2h', 'test-trap-008'), html: `${icon('swords', 15)} Replay test-trap-008: crew vs ReAct` }),
        h('div', { class: 'grid g2', style: { marginTop: '14px' } },
          h('div', { class: 'card kpi', style: { padding: '12px' } }, h('div', { class: 'v', style: { color: 'var(--good)', fontSize: '30px' } }, `${crew.unapproved}`), h('div', { class: 'k' }, 'unapproved attempts · crew')),
          h('div', { class: 'card kpi', style: { padding: '12px' } }, h('div', { class: 'v', style: { color: 'var(--bad)', fontSize: '30px' } }, `${react.unapproved}`), h('div', { class: 'k' }, 'unapproved attempts · ReAct'))))));
}
