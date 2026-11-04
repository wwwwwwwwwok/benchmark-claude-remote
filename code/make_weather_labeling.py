"""Build a local page for labelling the weather of every segment of the main experiment (user, 2026-11-03: the user
labels the weather; the exploratory analysis of the risk scores' own rule uses it as a scene cue).

One item per segment: the 2 x 3 grid image (all five cameras) and the larger front image of the segment's middle
target frame (of its target frames sorted, the lower middle one), both copied byte for byte from vlm_inputs (never
opened or re-encoded). The labeller picks one of WEATHER (keys 1-5) and may add a note. The page shows neither the
segment name nor Waymo's time of day, place or weather. --repeats segments come a second time, at least --gap items
after their first showing, to see how consistently they are labelled (the analysis uses the first label). Answers stay
in the browser (localStorage) until the CSV is downloaded; the page sends nothing anywhere.

usage: python -I make_weather_labeling.py <vlm_inputs> <out_dir> [--targets grid/targets_main.csv] [--repeats 20]
                                          [--gap 50] [--seed 20261103]
Output: <out_dir>/index.html, <out_dir>/img/ (two images per segment), <out_dir>/key.csv (item, segment, frame_idx,
image, repeat_of). <out_dir> must not exist yet.
"""
import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys

sys.stdout.reconfigure(encoding='utf-8')
# code written to the CSV, button text, explanation shown on hover (keys 1-5 in this order)
WEATHER = [('dry', '干', '路面干，没在下雨（晴天、阴天都算）'),
           ('wet', '路面湿', '路面湿，但没在下雨'),
           ('rain', '下雨', '在下雨：镜头上有雨点、雨刷在动、看得见雨丝或溅起的水'),
           ('fog', '雾', '有雾或霾，远处发白看不清'),
           ('unclear', '看不清', '判断不了，请在备注里写原因')]

PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>天气标注</title>
<style>
:root { --bg:#f6f6f4; --card:#ffffff; --ink:#1d1d1b; --ink2:#5f5f5a; --line:#d9d9d4; --sel:#ffe08a; --ok:#2f7d4f; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink);
       font:14px/1.5 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }
header { position:sticky; top:0; z-index:2; display:flex; flex-wrap:wrap; gap:8px; align-items:center;
         padding:8px 16px; background:var(--card); border-bottom:1px solid var(--line); }
header h1 { font-size:16px; margin:0 8px 0 0; }
#progress { color:var(--ink2); margin-right:auto; }
button { font:inherit; padding:5px 12px; border:1px solid var(--line); border-radius:8px; background:var(--card);
         color:var(--ink); cursor:pointer; }
button:hover { border-color:#8a8a84; }
button.primary { background:var(--ink); color:#fff; border-color:var(--ink); }
main { max-width:1500px; margin:0 auto; padding:12px 16px 40px; }
details.rules { color:var(--ink2); background:var(--card); border:1px solid var(--line); border-radius:10px;
                padding:10px 14px; margin-bottom:12px; }
details.rules summary { color:var(--ink); font-weight:600; cursor:pointer; }
.rules b { color:var(--ink); }
.meta { margin:4px 0 8px; color:var(--ink2); }
.meta b { color:var(--ink); font-size:16px; }
#lab { margin-left:10px; color:var(--ok); font-weight:600; }
.imgs { display:flex; gap:10px; align-items:flex-start; flex-wrap:wrap; }
.imgs a { display:block; min-width:0; }
.imgs a.grid { flex:3 1 560px; }
.imgs a.front { flex:2 1 380px; }
.imgs img { display:block; width:100%; height:auto; border-radius:6px; background:#ddd; }
.answers { margin-top:12px; display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
.answers button { padding:8px 14px; }
.answers .k { display:inline-block; min-width:1.5em; margin-right:6px; border:1px solid var(--line); border-radius:4px;
              font-size:12px; color:var(--ink2); text-align:center; }
.answers button.on { background:var(--sel); border-color:var(--ink); font-weight:600; }
#note { flex:1 1 260px; min-width:0; border:1px solid var(--line); border-radius:8px; padding:8px 10px; font:inherit; }
.nav { margin-top:10px; display:flex; gap:8px; flex-wrap:wrap; align-items:center; color:var(--ink2); }
#jump { width:90px; font:inherit; padding:4px 8px; border:1px solid var(--line); border-radius:8px; }
#warn { display:none; margin-bottom:12px; padding:8px 14px; border-radius:10px; background:#fde2e1; color:#7a1712; }
details.out { margin-top:24px; }
textarea { width:100%; height:140px; margin-top:8px; font:12px/1.4 ui-monospace,Consolas,monospace; }
</style></head><body>
<header><h1>天气标注</h1><span id="progress"></span>
<button onclick="go(cur - 1)">← 上一个</button><button onclick="go(cur + 1)">下一个 →</button>
<button onclick="nextTodo()">跳到下一个没标的</button>
<button class="primary" onclick="downloadCsv()">下载 CSV</button><button onclick="copyResult()">复制结果</button></header>
<main>
<div id="warn">这个浏览器存不了答案（可能是隐私模式），关掉页面之前一定要先点"下载 CSV"。</div>
<details class="rules" open><summary>怎么标（看过可以点这里收起）</summary>
每一段（约 20 秒的一段行车）标一次天气。左边是 5 个相机同一时刻的拼图，右边是放大的前视相机；点图可以在新标签页里看原图。
只看天气和路面，不用管车和行人。按键盘 <b>1–5</b> 或点按钮来选，选完自动跳到下一个；<b>← →</b> 前后翻，<b>N</b> 跳到下一个没标的。
拿不准的选"看不清"，在备注里写原因；按钮上停一下会显示每一类的说明。
<br>其中有 __REPEATS__ 段会再出现一次（隔得比较远，不告诉你是哪几段），用来看前后标得一不一致：照当下的判断标就行，不用回头找。
答案自动存在这个浏览器里，标完点<b>"下载 CSV"</b>；中途也可以随时下载一份，用最新的那份。</details>
<div class="meta"><b id="num"></b><span id="lab"></span></div>
<div class="imgs"><a class="grid" id="ga" target="_blank" rel="noopener"><img id="gi" alt="5 个相机的拼图"></a>
<a class="front" id="fa" target="_blank" rel="noopener"><img id="fi" alt="前视相机"></a></div>
<div class="answers" id="answers"><input id="note" placeholder="备注（可不填）"></div>
<div class="nav">跳到第 <input id="jump" type="number" min="1" max="__N__"> 个 <button onclick="jumpTo()">去</button></div>
<details class="out"><summary>结果预览（"复制结果"不起作用时，可以在这里全选复制）</summary><textarea id="out" readonly></textarea></details>
</main>
<script>
const ITEMS = __ITEMS__;
const WEATHER = __WEATHER__;
const KEY = 'weather_labels___STAMP__';
let ans = {}, cur = 0, stored = true;
try { ans = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { ans = {}; stored = false; }
const box = document.getElementById('answers'), noteEl = document.getElementById('note');
WEATHER.forEach(([code, text, tip], j) => {
  const b = document.createElement('button');
  b.dataset.v = code; b.title = tip;
  b.innerHTML = '<span class="k">' + (j + 1) + '</span>' + text;
  b.onclick = () => pick(code);
  box.insertBefore(b, noteEl); });
function persist() { try { localStorage.setItem(KEY, JSON.stringify(ans)); } catch (e) { stored = false; }
  document.getElementById('warn').style.display = stored ? 'none' : 'block'; }
function done(it) { const a = ans[it.i]; return !!(a && a.a); }
function csv() { const rows = ['item,segment,frame_idx,repeat_of,weather,note'];
  ITEMS.forEach(it => { const a = ans[it.i] || {};
    rows.push([it.i, it.seg, it.f, it.rep || '', a.a || '', '"' + (a.n || '').replace(/"/g, '""') + '"'].join(',')); });
  return rows.join('\n') + '\n'; }
function render() { const it = ITEMS[cur], a = ans[it.i] || {};
  document.getElementById('num').textContent = '第 ' + (cur + 1) + ' / ' + ITEMS.length + ' 个';
  const w = WEATHER.find(x => x[0] === a.a);
  document.getElementById('lab').textContent = w ? '已选：' + w[1] : '';
  const g = 'img/' + it.img + '_grid.jpg', f = 'img/' + it.img + '_front.jpg';
  document.getElementById('gi').src = g; document.getElementById('ga').href = g;
  document.getElementById('fi').src = f; document.getElementById('fa').href = f;
  box.querySelectorAll('button').forEach(b => b.classList.toggle('on', b.dataset.v === a.a));
  noteEl.value = a.n || '';
  document.getElementById('progress').textContent = '已标 ' + ITEMS.filter(done).length + ' / ' + ITEMS.length;
  document.getElementById('out').value = csv();
  const nx = ITEMS[cur + 1];
  if (nx) { new Image().src = 'img/' + nx.img + '_grid.jpg'; new Image().src = 'img/' + nx.img + '_front.jpg'; } }
function go(j) { cur = Math.max(0, Math.min(ITEMS.length - 1, j)); render(); }
function pick(v) { const it = ITEMS[cur]; ans[it.i] = Object.assign(ans[it.i] || {}, {a: v}); persist();
  if (ITEMS.every(done)) { render(); alert('都标完了，请点"下载 CSV"。'); }
  else if (cur < ITEMS.length - 1) go(cur + 1); else render(); }
noteEl.oninput = () => { const it = ITEMS[cur]; ans[it.i] = Object.assign(ans[it.i] || {}, {n: noteEl.value}); persist();
  document.getElementById('out').value = csv(); };
function nextTodo() { const n = ITEMS.length;
  for (let d = 1; d <= n; d++) { const j = (cur + d) % n; if (!done(ITEMS[j])) { go(j); return; } }
  alert('都标完了，请点"下载 CSV"。'); }
function jumpTo() { const v = parseInt(document.getElementById('jump').value, 10); if (v >= 1 && v <= ITEMS.length) go(v - 1); }
function downloadCsv() { const b = new Blob(['﻿' + csv()], {type: 'text/csv'}); const a = document.createElement('a');
  a.href = URL.createObjectURL(b); a.download = 'weather_labels.csv'; document.body.appendChild(a); a.click(); a.remove(); }
function copyResult() { const t = csv();
  (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(
    () => alert('已复制'),
    () => { const o = document.getElementById('out'); o.closest('details').open = true; o.focus(); o.select();
            alert('自动复制没成功，已选中页面底部的结果框，请按 Ctrl+C 复制'); }); }
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
    if (e.key === 'Escape') e.target.blur();
    else if (e.key === 'Enter' && e.target.id === 'jump') jumpTo();
    return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const j = parseInt(e.key, 10);
  if (j >= 1 && j <= WEATHER.length) { e.preventDefault(); pick(WEATHER[j - 1][0]); }
  else if (e.key === 'ArrowLeft') go(cur - 1);
  else if (e.key === 'ArrowRight') go(cur + 1);
  else if (e.key === 'n' || e.key === 'N') nextTodo(); });
persist();
const first = ITEMS.findIndex(it => !done(it)); go(first < 0 ? 0 : first);
</script></body></html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('vlm_inputs')
    ap.add_argument('out')
    ap.add_argument('--targets', default='grid/targets_main.csv', help='relative to <vlm_inputs>')
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--gap', type=int, default=50)
    ap.add_argument('--seed', type=int, default=20261103)
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit(f'{a.out} exists already; give a new folder (a page being labelled is never overwritten)')
    targets = os.path.join(a.vlm_inputs, a.targets)
    with open(targets, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    by_seg = {}
    for r in rows:
        by_seg.setdefault(r['segment'], []).append(r)
    picked, missing = {}, []
    for s, g in sorted(by_seg.items()):
        r = sorted(g, key=lambda r: int(r['frame_idx']))[(len(g) - 1) // 2]
        k = int(r['frame_idx'])
        grid = os.path.join(os.path.dirname(os.path.abspath(targets)), r['images'].split(';')[-1])   # 'now', the last
        front = os.path.join(a.vlm_inputs, 'separate', s, f'FRONT_{k:03d}.jpg')
        if os.path.basename(grid) != f'step_{k:03d}.jpg':
            sys.exit(f'{s} frame {k}: its last grid image is {grid}, not this frame')
        missing += [p for p in (grid, front) if not os.path.isfile(p)]
        picked[s] = (k, grid, front)
    if missing:
        sys.exit(f'{len(missing)} images missing, e.g. {missing[:3]}')

    rng = random.Random(a.seed)
    order = sorted(picked)
    rng.shuffle(order)
    n = len(order)
    if not 0 <= a.repeats <= n - a.gap:
        sys.exit(f'--repeats {a.repeats} with --gap {a.gap}: only {max(n - a.gap, 0)} of {n} segments leave room')
    keyed = [(float(p), s, False) for p, s in enumerate(order)]
    for s in rng.sample(order[:n - a.gap], a.repeats):         # the second showing at least --gap items later
        keyed.append((rng.uniform(order.index(s) + a.gap, n), s, True))
    keyed.sort(key=lambda x: x[0])
    code = {s: f'{i + 1:03d}' for i, s in enumerate(order)}
    first, items = {}, []
    for i, (_, s, again) in enumerate(keyed, 1):
        items.append(dict(i=i, seg=s, f=picked[s][0], img=code[s], rep=first[s] if again else 0))
        first.setdefault(s, i)

    os.makedirs(os.path.join(a.out, 'img'))
    nbytes = 0
    for s, (k, grid, front) in picked.items():
        for src, kind in ((grid, 'grid'), (front, 'front')):
            dst = os.path.join(a.out, 'img', f'{code[s]}_{kind}.jpg')
            shutil.copyfile(src, dst)
            nbytes += os.path.getsize(dst)
    with open(os.path.join(a.out, 'key.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['item', 'segment', 'frame_idx', 'image', 'repeat_of'])
        for it in items:
            w.writerow([it['i'], it['seg'], it['f'], it['img'], it['rep'] or ''])
    stamp = hashlib.sha256(json.dumps([[it['seg'], it['rep']] for it in items]).encode()).hexdigest()[:12]
    page = (PAGE.replace('__ITEMS__', json.dumps(items)).replace('__WEATHER__', json.dumps(WEATHER, ensure_ascii=False))
            .replace('__STAMP__', stamp).replace('__N__', str(len(items))).replace('__REPEATS__', str(a.repeats)))
    with open(os.path.join(a.out, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(page)
    print(f'{n} segments ({len(rows)} target frames), {a.repeats} of them shown twice: {len(items)} items; '
          f'{2 * n} images copied, {nbytes:,} bytes; page {os.path.join(a.out, "index.html")}')


if __name__ == '__main__':
    main()
