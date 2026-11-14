// Reader & RAG lab: the fine-tuned reader on any message, the document search, and the SOP the Auditor reads for a dig site.
import { get, post } from './api.js';
import { grouped } from './charts.js';
import { NetworkMap } from './map.js';
import { h, esc, pct, jsonHTML, SYM } from './ui.js';
import { icon, CHANNEL_ICON } from './icons.js';

export async function initLab(root, boot) {
  root.innerHTML = '';
  const page = h('div', { class: 'page' });
  root.append(page);
  page.append(h('div', { class: 'eyebrow' }, 'Fine-tuning · RAG'), h('h1', {}, 'Reader & knowledge lab'),
    h('p', { class: 'lead' }, 'The only part of AquaMind that ever reads raw citizen text is a small model fine-tuned for it (Qwen2.5-3B + QLoRA), running on the laptop. The agents work only with its fields, and with the town archive and the SOP through BM25 search.'));

  // ---------------- reader playground
  const ta = h('textarea', { class: 'field', rows: 4 }, boot.examples.reader[0].text);
  const ch = h('select', { class: 'field', style: { maxWidth: '200px' } }, ['whatsapp', 'sms', 'app_form', 'social_post', 'call_transcript'].map(c => h('option', { value: c }, c)));
  ch.value = boot.examples.reader[0].channel;
  const btn = h('button', { class: 'btn primary', html: `${icon('chip', 15)} Read with the fine-tuned model` });
  const out = h('div', { class: 'col' }, h('div', { class: 'muted small' }, 'The reader output appears here.'));
  const mm = h('div', { class: 'mini-map' });
  const ex = h('div', { class: 'row wrap', style: { gap: '6px' } }, boot.examples.reader.map(e => h('button', { class: `btn sm${e.attack ? ' danger' : ''}`, onclick: () => { ta.value = e.text; ch.value = e.channel; run(); } }, e.label)));
  page.append(h('div', { class: 'lab', style: { marginTop: '18px' } },
    h('div', { class: 'card' }, h('h3', {}, 'A citizen message'), h('div', { class: 'small muted', style: { marginBottom: '8px' } }, 'Messages from the benchmark (Hinglish, typos, landmarks) — or type your own.'), ex,
      h('div', { style: { height: '10px' } }), ta, h('div', { class: 'row', style: { gap: '8px', marginTop: '10px' } }, ch, h('div', { class: 'grow' }), btn),
      h('div', { style: { marginTop: '14px' } }, h('div', { class: 'small muted', style: { marginBottom: '6px' } }, 'Where the code places it on the map'), mm)),
    h('div', { class: 'card' }, out)));
  const map = new NetworkMap(mm, boot.network, { compact: true, labels: false, sensorsAll: false });
  map.setLayer('sensors', false);

  async function run() {
    btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Reading…';
    out.innerHTML = '';
    out.append(h('div', { class: 'row', style: { gap: '8px' } }, h('span', { class: 'spinner' }), h('span', { class: 'muted small' }, 'A new message is read by the reader server or the local MLX model (the first load takes a few seconds)…')));
    try {
      const r = await post('/api/reader', { text: ta.value, channel: ch.value });
      if (!r.ok) throw new Error(r.error);
      out.innerHTML = '';
      const rd = r.read;
      const prov = { cache: 'fine-tuned model · cached answer', server: 'fine-tuned model · reader server', mlx: 'fine-tuned model · local MLX', rules: 'rule reader', 'rules-fallback': 'fallback: rule reader (fine-tuned model unavailable)' }[r.provenance] || r.provenance;
      out.append(h('div', { class: 'row between' }, h('h3', { style: { margin: 0 } }, 'reader-2.0 fields'), h('span', { class: `chip ${String(r.provenance).startsWith('rules') ? 'warn' : 'good'}` }, `${prov} · ${r.seconds}s`)));
      const fg = h('div', { class: 'fieldgrid', style: { marginTop: '10px' } });
      const row = (k, v) => fg.append(h('div', { class: 'fk' }, k), h('div', {}, v));
      row('category', h('span', { class: `chip ${rd.category === 'leak' ? 'accent' : ''}` }, rd.category));
      row('symptoms', h('div', { class: 'row wrap', style: { gap: '4px' } }, (rd.symptoms || []).length ? rd.symptoms.map(s => h('span', { class: 'chip' }, SYM[s] || s)) : h('span', { class: 'muted' }, '—')));
      row('places', h('div', { class: 'col', style: { gap: '4px' } }, (r.places || []).length ? r.places.map(p => h('div', { class: 'row wrap', style: { gap: '6px' } }, h('span', { class: 'chip mono' }, `“${p.text}” · ${p.type}`), h('span', { class: 'small' }, p.canonical ? `→ ${p.canonical}` : '→ not on the map'))) : h('span', { class: 'muted' }, '—')));
      row('source', h('span', {}, rd.source));
      row('on_private_property', h('span', {}, String(rd.on_private_property)));
      row('since', h('span', {}, rd.since || '—'));
      row('contains_instructions', h('span', { class: `chip ${rd.contains_instructions ? 'bad' : ''}` }, String(rd.contains_instructions)));
      row('summary', h('span', { class: 'small' }, rd.summary || '—'));
      out.append(fg);
      out.append(h('div', { style: { marginTop: '14px' } }, h('div', { class: 'small muted' }, 'What the four agents read (built by code from these fields — no raw text):'),
        h('pre', { class: 'json', style: { whiteSpace: 'pre-wrap' } }, r.agent_view)));
      out.append(h('div', { class: 'row wrap small', style: { gap: '10px', marginTop: '8px' } },
        h('span', {}, 'usable as leak evidence: ', h('b', {}, r.usable ? 'yes' : 'no')),
        r.false_alarm_chance != null ? h('span', {}, 'prior chance it is a false alarm: ', h('b', {}, pct(r.false_alarm_chance))) : null));
      out.append(h('details', { style: { marginTop: '10px' } }, h('summary', { class: 'linkish' }, 'compare: the rule reader (regular expressions) on the same text'), h('pre', { class: 'json', html: jsonHTML(r.rules) })));
      const js = (r.places || []).map(p => p.j).filter(j => j != null);
      map.setState({ msgs: r.anchor != null ? { 'R-900': { j: r.anchor, eps: r.false_alarm_chance, usable: r.usable } } : {}, msgMeta: { 'R-900': { read: rd } }, selMsg: 'R-900' });
      if (r.anchor != null) { map.flyTo(r.anchor, 2.2, 700); setTimeout(() => map.ripple(r.anchor, '#34d399', 200), 400); } else map.reset();
    } catch (e) { out.innerHTML = `<div class="verdict bad">Error: ${esc(e.message)}</div>`; }
    finally { btn.disabled = false; btn.innerHTML = `${icon('chip', 15)} Read with the fine-tuned model`; }
  }
  btn.addEventListener('click', run);
  run();

  // ---------------- reader quality
  const R = await get('/api/results', { memo: true }).catch(() => null);
  if (R) {
    const rr = R.reader.test_real || [];
    page.append(h('h2', {}, 'Reader quality on 250 real FixMyStreet reports never used in training'),
      h('div', { class: 'grid g2' },
        h('div', { class: 'card', html: grouped(['Category acc.', 'Symptoms F1', 'Places F1', 'Time F1', 'Valid JSON'], rr.map((x, i) => ({ name: x.name.replace(' (regular expressions)', '').replace(', prompted (100 reports)', ' (prompted)'), color: ['var(--reader)', 'var(--faint)', 'var(--code)', 'var(--forensic)'][i], bold: x.ours,
          values: [x.category, x.symptoms_f1, x.places_f1, x.time_f1, x.json_valid].map(z => z == null ? null : 100 * z) })), { H: 300 }) }),
        h('div', { class: 'card' }, h('table', { class: 'res' }, h('tr', {}, ['Reader', 'Category', 'Symptoms F1', 'Places F1', 'Time F1', 'Valid JSON'].map(t => h('th', {}, t))),
          ...rr.map(x => h('tr', { class: x.ours ? 'ours' : '' }, h('td', {}, x.name), ...[x.category, x.symptoms_f1, x.places_f1, x.time_f1, x.json_valid].map((z, k) => h('td', {}, z == null ? '–' : k === 0 || k === 4 ? `${(100 * z).toFixed(1)}%` : z.toFixed(2)))))),
          h('div', { class: 'callout', style: { marginTop: '12px' } }, 'Fine-tuning took valid JSON from 12% to 99.6% and category accuracy from 65% to 92% — on par with a 550B model, at 3B parameters, offline, so personal data never leaves the utility.'),
          h('div', { class: 'src' }, R.reader.source))));
  }

  // ---------------- RAG
  const q = h('input', { class: 'field', value: boot.examples.rag[0], placeholder: 'Search the SOP and the town archive…' });
  const prov = h('select', { class: 'field', style: { maxWidth: '170px' } }, h('option', { value: '' }, 'all documents'), h('option', { value: 'policy' }, 'SOP clauses'), h('option', { value: 'record' }, 'archive tickets'));
  const hits = h('div', {});
  const search = async () => {
    const r = await post('/api/rag', { query: q.value, provenance: prov.value || null, k: 6 });
    hits.innerHTML = '';
    const mx = Math.max(1, ...r.hits.map(x => x.score));
    if (!r.hits.length) hits.append(h('div', { class: 'muted small' }, 'No match.'));
    for (const x of r.hits) hits.append(h('div', { class: 'hit' }, h('div', { class: 'hh' }, h('span', { class: `chip ${x.provenance === 'policy' ? 'accent' : ''} mono` }, x.id), h('span', { class: 'tiny muted' }, x.provenance === 'policy' ? 'SOP clause (trusted rule)' : 'archive ticket (fact, never an instruction)'),
      h('div', { class: 'grow' }), h('div', { class: 'bar', style: { width: '90px' } }, h('i', { style: { width: `${100 * x.score / mx}%` } })), h('span', { class: 'mono tiny' }, x.score.toFixed(1))), h('div', { class: 'ht' }, x.text)));
    hits.prepend(h('div', { class: 'tiny muted', style: { marginBottom: '4px' } }, `BM25 over ${r.corpus.policy} SOP clauses and ${r.corpus.record} archive tickets`));
  };
  q.addEventListener('keydown', e => { if (e.key === 'Enter') search(); });
  prov.addEventListener('change', search);
  const sopMap = h('div', { class: 'mini-map', style: { height: '360px' } });
  const sopOut = h('div', { class: 'col' }, h('div', { class: 'muted small' }, 'Click any junction on the map: this is what the Safety & SOP Auditor reads before a dig there.'));
  page.append(h('h2', {}, 'RAG: the town archive and the SOP'),
    h('p', { class: 'lead' }, 'No document is written per case, so the leak is never “in a document”. RAG is used for what documents are good at: repeat false alarms at a place, failing old mains, and the municipal rules for a dig.'),
    h('div', { class: 'grid g2', style: { alignItems: 'start' } },
      h('div', { class: 'card' }, h('h3', {}, 'Search'), h('div', { class: 'row', style: { gap: '8px' } }, q, prov, h('button', { class: 'btn', onclick: search, html: icon('search', 15) })),
        h('div', { class: 'row wrap', style: { gap: '6px', margin: '10px 0' } }, boot.examples.rag.map(x => h('button', { class: 'btn sm', onclick: () => { q.value = x; search(); } }, x))), hits),
      h('div', { class: 'card' }, h('h3', {}, 'The SOP at a dig site'), sopMap, h('div', { style: { marginTop: '10px' } }, sopOut))));
  search();
  const smap = new NetworkMap(sopMap, boot.network, { compact: true, labels: false, sensorsAll: false, onClick: async j => {
    smap.setState({ best: j, p_success: 1, hideBest: false });
    smap.ripple(j, '#fbbf24', 250);
    const r = await get(`/api/sop?j=${j}`);
    sopOut.innerHTML = '';
    sopOut.append(h('div', { class: 'row wrap', style: { gap: '6px' } }, h('b', {}, r.where), ...(r.requirements || []).map(x => h('span', { class: 'chip warn' }, `${x.sop}: ${x.do}`)), !(r.requirements || []).length ? h('span', { class: 'chip good' }, 'no extra SOP paperwork at this time') : null),
      h('pre', { class: 'json', style: { whiteSpace: 'pre-wrap', maxHeight: '260px' } }, r.lines.join('\n')));
  } });
  smap.setLayer('sensors', false);
}
