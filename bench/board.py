"""Leaderboard page: one row per experiment, qualified runs ranked by realism, disqualified runs below.

Writes experiments/board/index.html plus web-sized images in experiments/board/img/. Publish the folder as
a private page; it holds client imagery, so never commit it.
"""
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

import benchlib as bl
import crops as cr

THUMB = 960


def _web(img, path, long=THUMB):
    h, w = img.shape[:2]
    s = min(1.0, long / max(h, w))
    im = Image.fromarray(img)
    if s < 1:
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, 'WEBP', quality=82)


def build(bench, root, rows):
    board = root / 'board'
    img_dir = board / 'img'
    show = {}
    for v in bench.views:
        cs = bench.crops['views'][v.id] if bench.crops else []
        show[v.id] = [c for c in cs if c['kind'] == 'plant'][:2] + [c for c in cs if c['kind'] == 'ground'][:1]
    entries = []
    for d, s in rows:
        imgs = {}
        for v in bench.views:
            p = d / 'out' / f'{v.id}.png'
            if not p.exists():
                continue
            out = bl.load_rgb(p)
            rel = f"img/{s['id']}-{v.id}.webp"
            _web(out, board / rel, 1600)
            imgs[v.id] = {'overview': rel, 'crops': []}
            for c in show[v.id]:
                r2 = f"img/{s['id']}-{c['id']}.webp"
                _web(cr.cut(out, c), board / r2, 640)
                imgs[v.id]['crops'].append({'src': r2, 'kind': c['kind'], 'label': (c.get('focus') or {}).get('species', c['kind'])})
        entries.append(dict(s, images=imgs))
    entries.sort(key=lambda e: (e.get('qualified') is not True, -(e.get('win_vs_shipped') or 0), -(e.get('realism') or -1)))
    data = json.dumps({'bench': bench.name, 'views': [v.id for v in bench.views], 'entries': entries})
    page = TEMPLATE.replace('__DATA__', data.replace('</', '<\\/')).replace('__BENCH__', html.escape(bench.name))
    (board / 'index.html').write_text(page, encoding='utf-8')
    return board / 'index.html'


TEMPLATE = r"""<title>Photoreal Lab Board</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Layout: leaderboard. One band per experiment: rank and scores on the left, both views and close crops on the right. */
:root { --paper:#f3f5f0; --sheet:#fbfcf9; --ink:#1d2a24; --muted:#5f6e66; --rule:#d3dbd3; --ok:#2f7650; --ok-soft:#dfeee5;
  --bad:#9a3426; --bad-soft:#f6e0dc; --warn:#8a5a12; --warn-soft:#f5e9d2;
  --display:"Barlow Condensed","Arial Narrow",system-ui,sans-serif; --body:"IBM Plex Sans","Segoe UI",system-ui,sans-serif; --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --paper:#121815; --sheet:#18201c; --ink:#e2e9e4; --muted:#9daba3; --rule:#2d3832;
  --ok:#7fca9d; --ok-soft:#1b3125; --bad:#ec9a8c; --bad-soft:#3a211d; --warn:#e3b76a; --warn-soft:#33281a; color-scheme:dark; } }
:root[data-theme="dark"] { --paper:#121815; --sheet:#18201c; --ink:#e2e9e4; --muted:#9daba3; --rule:#2d3832;
  --ok:#7fca9d; --ok-soft:#1b3125; --bad:#ec9a8c; --bad-soft:#3a211d; --warn:#e3b76a; --warn-soft:#33281a; color-scheme:dark; }
* { box-sizing:border-box; }
body { background:var(--paper); color:var(--ink); font:15px/1.5 var(--body); }
.wrap { max-width:1400px; margin:0 auto; padding-left:max(16px, env(safe-area-inset-left, 0px)); padding-right:max(16px, env(safe-area-inset-right, 0px)); padding-block:24px 48px; display:grid; gap:22px; }
h1 { font:700 clamp(2rem,4.5vw,3rem)/1 var(--display); text-transform:uppercase; margin:0; }
.sub { color:var(--muted); }
.exp { background:var(--sheet); border:1px solid var(--rule); display:grid; grid-template-columns:minmax(0,300px) minmax(0,1fr); }
.exp.dq { opacity:.78; }
.meta { padding:14px 16px; border-right:1px solid var(--rule); display:grid; gap:8px; align-content:start; }
.rank { font:700 2.2rem/1 var(--display); color:var(--muted); }
.id { font:600 1rem/1.2 var(--mono); }
.hyp { font-size:.88rem; color:var(--muted); }
.score { display:flex; gap:14px; flex-wrap:wrap; }
.score div { display:grid; } .score b { font:600 1.4rem/1.1 var(--mono); font-variant-numeric:tabular-nums; } .score span { font-size:.75rem; color:var(--muted); }
.chips { display:flex; flex-wrap:wrap; gap:4px; }
.chip { font:500 .72rem/1.6 var(--mono); padding:0 7px; border-radius:3px; background:var(--ok-soft); color:var(--ok); }
.chip.no { background:var(--bad-soft); color:var(--bad); } .chip.gw { background:var(--warn-soft); color:var(--warn); }
.views { padding:12px; display:grid; gap:10px; min-width:0; }
.view { display:grid; grid-template-columns:minmax(0,2fr) repeat(3,minmax(0,1fr)); gap:8px; align-items:start; }
figure { margin:0; display:grid; gap:3px; min-width:0; } figure img { width:100%; height:auto; display:block; border-radius:4px; background:var(--rule); }
figcaption { font-size:.72rem; color:var(--muted); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
@media (max-width:900px) { .exp { grid-template-columns:minmax(0,1fr); } .meta { border-right:0; border-bottom:1px solid var(--rule); } .view { grid-template-columns:repeat(2,minmax(0,1fr)); } .view figure:first-child { grid-column:1 / -1; } }
</style>
<div class="wrap">
  <header><h1>Photoreal Lab Board</h1><p class="sub">Bench __BENCH__. Qualified runs (every gate passed) ranked by how often the judge picks them over the shipped output side by side (50% = no better), then by realism; disqualified runs below. Realism = weighted mean of the judge's photo probability over fixed crops.</p></header>
  <main id="list"></main>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function () {
  const D = JSON.parse(document.getElementById('data').textContent);
  const el = (t, a, ...k) => { const n = document.createElement(t); for (const [x, v] of Object.entries(a || {})) { if (v == null) continue; if (x === 'class') n.className = v; else if (x === 'text') n.textContent = v; else n.setAttribute(x, v); } k.flat().forEach(c => c != null && n.append(c.nodeType ? c : document.createTextNode(String(c)))); return n; };
  const list = document.getElementById('list'); let rank = 0;
  D.entries.forEach(e => {
    const q = e.qualified === true; if (q) rank++;
    const gates = [];
    Object.entries(e.gates || {}).forEach(([v, g]) => Object.entries(g).forEach(([name, ok]) => gates.push(el('span', { class: 'chip' + (ok ? '' : ' no'), text: v + ' ' + name }))));
    if (e.bloom) gates.push(el('span', { class: 'chip' + (e.bloom.pass ? '' : ' no'), text: 'in bloom ' + e.bloom.in_bloom + '/' + e.bloom.checked }));
    if (e.species) gates.push(el('span', { class: 'chip' + (e.species.pass ? '' : ' no'), text: 'species ' + (e.species.pairs - e.species.different.length) + '/' + e.species.pairs + ((e.species.extra_plants || []).length ? ' · ' + e.species.extra_plants.length + ' invented' : '') }));
    const gw = Object.entries(e.giveaways || {}).slice(0, 4).map(([k, n]) => el('span', { class: 'chip gw', text: k.replace(/_/g, ' ') + ' ' + n }));
    const meta = el('div', { class: 'meta' },
      el('div', { class: 'rank', text: q ? '#' + rank : 'DQ' }), el('div', { class: 'id', text: e.id }),
      el('div', { class: 'hyp', text: (e.hypothesis || '') + (e.method ? ' · ' + e.method : '') }),
      el('div', { class: 'score' },
        el('div', {}, el('b', { text: e.win_vs_shipped != null ? Math.round(e.win_vs_shipped * 100) + '%' : '–' }), el('span', { text: 'beats shipped' })),
        el('div', {}, el('b', { text: e.realism != null ? e.realism : '–' }), el('span', { text: 'realism' })),
        el('div', {}, el('b', { text: e.photo_share != null ? Math.round(e.photo_share * 100) + '%' : '–' }), el('span', { text: 'crops judged photo' })),
        el('div', {}, el('b', { text: e.gpu_seconds != null ? e.gpu_seconds : '–' }), el('span', { text: 'GPU-s' })),
        el('div', {}, el('b', { text: e.judge_anchor_accuracy != null ? Math.round(e.judge_anchor_accuracy * 100) + '%' : '–' }), el('span', { text: 'judge on anchors' }))),
      el('div', { class: 'chips' }, gates), el('div', { class: 'chips' }, gw));
    const views = el('div', { class: 'views' });
    D.views.forEach(v => { const im = (e.images || {})[v]; if (!im) return;
      views.append(el('div', { class: 'view' }, el('figure', {}, el('img', { src: im.overview, alt: e.id + ' ' + v, loading: 'lazy' }), el('figcaption', { text: v })),
        im.crops.map(c => el('figure', {}, el('img', { src: c.src, alt: e.id + ' ' + c.label, loading: 'lazy' }), el('figcaption', { text: c.label })))));
    });
    list.append(el('article', { class: 'exp' + (q ? '' : ' dq') }, meta, views));
  });
})();
</script>
"""
