"""Build a blind labelling page for the lead vehicle.

Frames = every frame where the three pairing methods disagree, plus a random sample of frames where
they agree. In each frame every vehicle/cyclist ahead of the ego that is visible in the FRONT image is
numbered (1 = nearest); the page never shows which method picked which car. The labeller picks the
true lead's number (0 = no lead, ? = unclear). The answer key stays in key.csv for scoring.

Usage:
    python -I make_lead_labeling.py <extracted_root> <frame_ssm.csv> <out_dir> [--n-random 50] [--seed 0]
"""
import argparse
import html
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wod_common import (Segment, LaneMap, project, cumlen, ego_future_path,  # noqa: E402
                        EGO_X_FRONT, EGO_X_REAR, EGO_HALF_W)

sys.stdout.reconfigure(encoding='utf-8')
METHODS = ('map', 'path', 'straight')
AHEAD_MAX = 90.0   # cars further ahead than this (m) get no number
BAND = 4.5         # number cars within this lateral distance (m) of the straight line ahead, the ego's
                   # map lane route or the ego's future path: the ego lane plus the neighbouring lanes
VIEW_LAT, VIEW_BACK, VIEW_FWD = 22.0, 10.0, 95.0


def rect(cx, cy, L, W, h):
    c, s = np.cos(h), np.sin(h)
    dx = np.array([1, 1, -1, -1]) * L / 2
    dy = np.array([1, -1, -1, 1]) * W / 2
    return np.stack([cx + dx * c - dy * s, cy + dx * s + dy * c], axis=-1)


def font(size, bold=True):
    for name in (('arialbd.ttf', 'arial.ttf') if bold else ('arial.ttf',)):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def segment_label(seg):
    c = seg.context
    loc = {'location_sf': '旧金山', 'location_phx': '凤凰城', 'location_mtv': '山景城',
           'location_other': '其他地点'}.get(c.get('location'), c.get('location', ''))
    tod = {'Day': '白天', 'Night': '夜间', 'Dawn/Dusk': '黄昏'}.get(c.get('time_of_day'), c.get('time_of_day', ''))
    return f'{loc} {tod}'


def numbered_objects(seg, k, pb_front, must_include, corridors):
    o = seg.objects[(seg.objects['frame_idx'] == k) & seg.objects['type'].isin(['vehicle', 'cyclist'])]
    vis = set(pb_front['laser_id'])
    must = o['id'].isin(must_include)
    o = o[o['id'].isin(vis) | must]
    near = o['cy'].abs().to_numpy() < BAND
    if len(o):
        P, _, _, _ = seg.to_map(k, o)
        for xy in corridors:
            _, d, _ = project(xy, cumlen(xy), P)
            near |= np.abs(d) < BAND
    keep = ((o['cx'] > 0) & (o['cx'] < AHEAD_MAX) & near) | o['id'].isin(must_include)
    o = o[keep].copy()
    o['dist'] = np.hypot(o['cx'], o['cy'])
    o = o.sort_values('dist').reset_index(drop=True)
    o['num'] = np.arange(1, len(o) + 1)
    return o


def draw_front(img_path, pb_front, nums, out_path, width=1280):
    img = Image.open(img_path).convert('RGB')
    d = ImageDraw.Draw(img)
    f, th = font(34), 34
    W, H = img.size
    boxes = []
    for r in pb_front.itertuples():
        n = nums.get(r.laser_id)
        if n is not None:
            boxes.append((r.box_w * r.box_h, n, r.box_cx - r.box_w / 2, r.box_cy - r.box_h / 2,
                          r.box_cx + r.box_w / 2, r.box_cy + r.box_h / 2))
    for _, _, x0, y0, x1, y1 in boxes:
        d.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=(20, 20, 20), width=5)
        d.rectangle([x0, y0, x1, y1], outline=(255, 214, 0), width=3)
    placed = []

    def free(q):
        return all(q[2] < p[0] or q[0] > p[2] or q[3] < p[1] or q[1] > p[3] for p in placed)

    for _, n, x0, y0, x1, y1 in sorted(boxes, reverse=True):     # big (near) boxes first
        t = str(n)
        w = d.textlength(t, font=f) + 12
        h = th + 8
        tries = [(x0, y0 - h - 2), (x1 - w, y0 - h - 2), (x0, y1 + 2), (x0 + 2, y0 + 2)]
        tries += [(x0, y0 - h - 2 - j * (h + 2)) for j in range(1, 6)]
        tries += [(x0 + j * (w + 2), y0 - h - 2) for j in range(1, 4)]
        tries += [(x0 - j * (w + 2), y0 - h - 2) for j in range(1, 4)]
        pos = next(((min(max(bx, 0), W - w), min(max(by, 0), H - h)) for bx, by in tries
                    if free((min(max(bx, 0), W - w), min(max(by, 0), H - h),
                             min(max(bx, 0), W - w) + w, min(max(by, 0), H - h) + h))), None)
        if pos is None:
            pos = (min(max(x0, 0), W - w), min(max(y0 - h - 2, 0), H - h))
        bx, by = pos
        placed.append((bx, by, bx + w, by + h))
        d.rectangle([bx, by, bx + w, by + h], fill=(20, 20, 20))
        d.text((bx + 6, by + 2), t, fill=(255, 214, 0), font=f)
    img.thumbnail((width, width))
    img.save(out_path, quality=85)


def draw_bev(seg, k, objs, out_path):
    ego = seg.ego_map[k, :2]
    c, s = np.cos(-seg.yaw[k]), np.sin(-seg.yaw[k])
    Rm = np.array([[c, -s], [s, c]])

    def to_screen(xy):                      # map frame -> vehicle frame -> screen (left = left, up = forward)
        v = (np.asarray(xy) - ego) @ Rm.T
        return np.column_stack([-v[:, 1], v[:, 0]])

    fig, ax = plt.subplots(figsize=(4.6, 8.4), facecolor='#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    for r in seg.map.itertuples():
        if len(r.xs) < 2:
            continue
        p = to_screen(np.column_stack([r.xs, r.ys]))
        if r.kind == 'lane':
            ax.plot(p[:, 0], p[:, 1], color='#c3c2b7', lw=0.8, zorder=1)
        elif r.kind == 'road_line':
            ax.plot(p[:, 0], p[:, 1], color='#898781', lw=0.8, ls=(0, (4, 3)), zorder=1)
        elif r.kind == 'road_edge':
            ax.plot(p[:, 0], p[:, 1], color='#52514e', lw=1.4, zorder=2)
        elif r.kind in ('crosswalk', 'speed_bump') and len(r.xs) >= 3:
            ax.add_patch(MplPolygon(p, closed=True, fc='#e1e0d9', ec='none', zorder=0))
    allo = seg.objects[(seg.objects['frame_idx'] == k) & seg.objects['type'].isin(['vehicle', 'pedestrian', 'cyclist'])]
    numbered = set(objs['id'])
    for r in allo.itertuples():
        if r.id in numbered:
            continue
        q = rect(r.cx, r.cy, r.length, r.width, r.heading)
        ax.add_patch(MplPolygon(np.column_stack([-q[:, 1], q[:, 0]]), closed=True,
                                fc='#e1e0d9', ec='#c3c2b7', lw=0.5, zorder=3))
    for r in objs.itertuples():
        q = rect(r.cx, r.cy, r.length, r.width, r.heading)
        ax.add_patch(MplPolygon(np.column_stack([-q[:, 1], q[:, 0]]), closed=True,
                                fc='#ffd600', ec='#141414', lw=1.0, zorder=4))
        ax.text(-r.cy + r.width / 2 + 0.8, r.cx, str(r.num), fontsize=13, fontweight='bold',
                color='#141414', va='center', ha='left', zorder=5)
    eq = np.array([[EGO_X_FRONT, EGO_HALF_W], [EGO_X_FRONT, -EGO_HALF_W],
                   [EGO_X_REAR, -EGO_HALF_W], [EGO_X_REAR, EGO_HALF_W]])
    ax.add_patch(MplPolygon(np.column_stack([-eq[:, 1], eq[:, 0]]), closed=True, fc='#2a78d6', ec='none', zorder=6))
    ax.set_xlim(-VIEW_LAT, VIEW_LAT)
    ax.set_ylim(-VIEW_BACK, VIEW_FWD)
    ax.set_aspect('equal')
    ax.tick_params(colors='#898781', labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#e1e0d9')
    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=110, facecolor='#fcfcfb')
    plt.close(fig)


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>前车标注</title>
<style>
:root { --bg:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; --accent:#2a78d6; --sel:#ffd600; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }
header { position:sticky; top:0; z-index:5; background:var(--card); border-bottom:1px solid var(--line); padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
header h1 { font-size:17px; margin:0 12px 0 0; }
#progress { color:var(--ink2); }
button { font:inherit; border:1px solid var(--line); background:#fff; border-radius:8px; padding:5px 12px; cursor:pointer; }
button:hover { border-color:var(--muted); }
button.primary { background:var(--accent); color:#fff; border-color:var(--accent); }
main { max-width:1500px; margin:0 auto; padding:16px; }
.rules { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 16px; margin-bottom:16px; color:var(--ink2); }
.rules b { color:var(--ink); }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 16px; margin-bottom:18px; }
.card.done { border-color:#9ec5f4; }
.meta { color:var(--ink2); margin-bottom:8px; }
.meta b { color:var(--ink); }
.imgs { display:flex; gap:12px; align-items:flex-start; flex-wrap:wrap; }
.imgs img.front { flex:1 1 600px; max-width:100%; min-width:0; border-radius:6px; }
.imgs img.bev { width:380px; max-width:100%; border:1px solid var(--line); border-radius:6px; }
.answers { margin-top:10px; display:flex; flex-wrap:wrap; gap:6px; align-items:center; }
.answers button { min-width:44px; }
.answers button.on { background:var(--sel); border-color:#141414; font-weight:600; }
.answers input { flex:1 1 220px; min-width:0; border:1px solid var(--line); border-radius:8px; padding:5px 10px; font:inherit; }
textarea { width:100%; height:160px; margin-top:8px; font:12px/1.4 ui-monospace,Consolas,monospace; }
@media (max-width:700px) { .imgs img.bev { width:100%; } }
</style></head><body>
<header><h1>前车标注</h1><span id="progress"></span>
<button onclick="nextTodo()">跳到下一个没标的</button>
<button class="primary" onclick="copyResult()">复制结果</button>
<button onclick="downloadCsv()">下载 CSV</button></header>
<main>
<div class="rules"><b>怎么标：</b>每帧里，自车前方的车按距离编了号（1 = 最近）。左图是前视相机，右图是俯视图（蓝色是自车，向上是行驶方向，黄色是编号的车，灰线是车道）。
<br>请选出<b>真正的前车</b>：和自车在<b>同一条车道</b>、<b>同方向</b>、在自车前方<b>最近</b>的那辆车。建议：车身一半以上在本车道就算本车道；自车正在变道时，以车头所在的车道为准。
<br><b>0</b> = 前面没有前车；<b>?</b> = 说不清（可在备注里写原因）。真正的前车没有编号时选"不在编号里"，并在备注里说明。答案自动保存在这个浏览器里，标完点"复制结果"发给 Claude，或者点"下载 CSV"。</div>
__CARDS__
<div class="card"><b>结果预览</b>（"复制结果"不起作用时，可以手动全选复制这里）<textarea id="out" readonly></textarea></div>
</main>
<script>
const ITEMS = __ITEMS__;
const KEY = 'lead_labels___STAMP__';
let ans = {};
try { ans = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) { ans = {}; }
function save() { try { localStorage.setItem(KEY, JSON.stringify(ans)); } catch (e) {} render(); }
function pick(i, v) { ans[i] = Object.assign(ans[i] || {}, {a: v}); save();
  const nx = document.getElementById('c' + (i + 1)); if (nx) nx.scrollIntoView({behavior: 'smooth', block: 'start'}); }
function note(i, el) { ans[i] = Object.assign(ans[i] || {}, {n: el.value}); save(); }
function csv() { const rows = ['item,segment,frame_idx,answer,note'];
  ITEMS.forEach(it => { const a = ans[it.i] || {}; rows.push([it.i, it.seg, it.f, a.a ?? '', '"' + (a.n || '').replace(/"/g, '""') + '"'].join(',')); });
  return rows.join('\\n'); }
function render() { let n = 0;
  ITEMS.forEach(it => { const a = (ans[it.i] || {}).a; const card = document.getElementById('c' + it.i);
    if (a !== undefined && a !== '') { n++; card.classList.add('done'); } else card.classList.remove('done');
    card.querySelectorAll('.answers button').forEach(b => b.classList.toggle('on', b.dataset.v === String(a))); });
  document.getElementById('progress').textContent = '已标 ' + n + ' / ' + ITEMS.length;
  document.getElementById('out').value = csv(); }
function nextTodo() { const it = ITEMS.find(it => !(ans[it.i] && ans[it.i].a !== undefined && ans[it.i].a !== ''));
  if (it) document.getElementById('c' + it.i).scrollIntoView({behavior: 'smooth', block: 'start'}); }
function copyResult() { const t = csv();
  (navigator.clipboard ? navigator.clipboard.writeText(t) : Promise.reject()).then(
    () => alert('已复制，粘贴到对话里发给 Claude 就行'),
    () => { const o = document.getElementById('out'); o.focus(); o.select(); alert('自动复制没成功，已选中页面底部的结果框，请按 Ctrl+C 复制'); }); }
function downloadCsv() { const b = new Blob(['\\ufeff' + csv()], {type: 'text/csv'}); const a = document.createElement('a');
  a.href = URL.createObjectURL(b); a.download = 'lead_labels.csv'; a.click(); }
ITEMS.forEach(it => { const c = document.getElementById('c' + it.i);
  c.querySelectorAll('.answers button').forEach(b => b.onclick = () => pick(it.i, b.dataset.v));
  const inp = c.querySelector('input'); inp.value = (ans[it.i] || {}).n || ''; inp.oninput = () => note(it.i, inp); });
render();
</script></body></html>
"""


def card_html(it):
    btns = ['<button data-v="0">0 没有前车</button>'] + \
           [f'<button data-v="{n}">{n}</button>' for n in range(1, it['n'] + 1)] + \
           ['<button data-v="other">不在编号里</button>', '<button data-v="?">? 说不清</button>']
    return (f'<div class="card" id="c{it["i"]}"><div class="meta"><b>#{it["i"]}</b> · {html.escape(it["label"])} · '
            f'第 {it["f"]} 帧 · t = {it["t"]:.1f} s · 自车 {it["v"]:.1f} m/s · 编号车 {it["n"]} 辆</div>'
            f'<div class="imgs"><img class="front" loading="lazy" src="img/{it["i"]:03d}_front.jpg" alt="前视相机">'
            f'<img class="bev" loading="lazy" src="img/{it["i"]:03d}_bev.png" alt="俯视图"></div>'
            f'<div class="answers">{"".join(btns)}<input placeholder="备注（可不填）"></div></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('frame_csv')
    ap.add_argument('out')
    ap.add_argument('--n-random', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, 'img'), exist_ok=True)

    fr = pd.read_csv(a.frame_csv)
    lead = {m: fr[f'lead_{m}'].fillna('') for m in METHODS}
    disagree = (lead['map'] != lead['path']) | (lead['map'] != lead['straight']) | (lead['path'] != lead['straight'])
    agree = fr[~disagree]
    rng = np.random.default_rng(a.seed)
    share = agree['segment'].value_counts() / len(agree)
    alloc = (share * a.n_random).round().astype(int)
    alloc.iloc[0] += a.n_random - alloc.sum()
    picked = [g.loc[rng.choice(g.index, size=min(alloc[s], len(g)), replace=False)] for s, g in agree.groupby('segment')]
    items = pd.concat([fr[disagree].assign(sample='disagree')] + [p.assign(sample='random') for p in picked])
    items = items.sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    items['item'] = np.arange(1, len(items) + 1)

    page_items, key_rows = [], []
    for seg_name, g in items.groupby('segment', sort=False):
        seg = Segment(a.root, seg_name)
        label = segment_label(seg)
        pb = pd.read_parquet(os.path.join(seg.dir, 'projected_boxes.parquet'))
        pb = pb[pb['camera'] == 'FRONT'].copy()
        pb['laser_id'] = pb['id'].str.replace(r'_FRONT$', '', regex=True)
        lmap = LaneMap(seg.map)
        if lmap.ok:
            ego_lanes, ego_s = lmap.match_sequence(seg.ego_map[:, :2], seg.yaw)
        else:
            ego_lanes, ego_s = [None] * len(seg.frames), None
        for r in g.itertuples():
            k = int(r.frame_idx)
            pbk = pb[pb['frame_idx'] == k]
            picks = {m: getattr(r, f'lead_{m}') for m in METHODS}
            must = {v for v in picks.values() if isinstance(v, str) and v}
            corridors = [ego_future_path(seg, k)]
            if ego_lanes[k] is not None:
                corridors.append(lmap.route(ego_lanes[k], ego_s[k], [l for l in ego_lanes[k:] if l is not None])[1])
            objs = numbered_objects(seg, k, pbk, must, corridors)
            nums = dict(zip(objs['id'], objs['num']))
            img = os.path.join(seg.dir, 'images', 'FRONT', f'{k:03d}_{int(r.timestamp_micros)}.jpg')
            draw_front(img, pbk, nums, os.path.join(a.out, 'img', f'{r.item:03d}_front.jpg'))
            draw_bev(seg, k, objs, os.path.join(a.out, 'img', f'{r.item:03d}_bev.png'))
            page_items.append(dict(i=int(r.item), seg=seg_name, label=label, f=k, t=float(r.t_s),
                                   v=float(r.ego_speed), n=len(objs)))
            row = dict(item=int(r.item), segment=seg_name, frame_idx=k, t_s=r.t_s, sample=r.sample,
                       numbering=';'.join(f'{n}:{i}' for i, n in nums.items()))
            for m, v in picks.items():
                row[f'{m}_pick'] = nums.get(v, -1) if isinstance(v, str) and v else 0
            key_rows.append(row)
            print(f'#{r.item:3d} {label} frame {k:3d} {r.sample:8s} numbered {len(objs)}')

    page_items.sort(key=lambda x: x['i'])
    cards = '\n'.join(card_html(it) for it in page_items)
    items_js = json.dumps([dict(i=it['i'], seg=it['seg'], f=it['f']) for it in page_items])
    stamp = f'{a.seed}_{len(page_items)}'
    page = PAGE.replace('__CARDS__', cards).replace('__ITEMS__', items_js).replace('__STAMP__', stamp)
    with open(os.path.join(a.out, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(page)
    pd.DataFrame(key_rows).sort_values('item').to_csv(os.path.join(a.out, 'key.csv'), index=False)
    print(f'{len(page_items)} frames: {int((items["sample"] == "disagree").sum())} disagreement + '
          f'{int((items["sample"] == "random").sum())} random; page at {os.path.join(a.out, "index.html")}')


if __name__ == '__main__':
    main()
