"""Local HTML page to explore the raw Waymo v1.4.3 HD map of the extracted segments, with a replay of
every labelled road user (recorded tracks, 10 Hz) driving on it.

Pan (drag) and zoom (wheel) the map; hover any feature or road user to see what it is; click a lane to
highlight the lanes it comes from (entry), goes to (exit), its left/right neighbours and its boundary lines;
click a vehicle/pedestrian/cyclist to see its whole track. Play/pause and scrub through the 20 s.
All segments go into one page with a selector.

Usage: python -I make_map_viewer.py <extracted_root> <out_html> [segment ...]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wod_common import footprint_from_lidars  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')

LANE_TYPES = {0: '未定义', 1: '高速公路', 2: '城市道路', 3: '自行车道'}
LINE_TYPES = {0: '未知', 1: '白色虚线', 2: '白色实线', 3: '白色双实线', 4: '黄色虚线', 5: '黄色双虚线',
              6: '黄色实线', 7: '黄色双实线', 8: '黄色虚实线（可借道超车）'}
EDGE_TYPES = {0: '未知', 1: '路缘（道路边界）', 2: '中央分隔带边缘'}
KIND_NAMES = {'lane': '车道中心线', 'road_line': '车道线', 'road_edge': '路缘', 'crosswalk': '人行横道',
              'speed_bump': '减速带', 'driveway': '车库/出入口', 'stop_sign': '停车标志'}
TCODE = {'vehicle': 0, 'pedestrian': 1, 'cyclist': 2}
PREFIX = {'vehicle': '车', 'pedestrian': '人', 'cyclist': '骑'}


def seg_data(root, seg):
    d = os.path.join(root, seg)
    m = pd.read_parquet(os.path.join(d, 'map.parquet'))
    fr = pd.read_parquet(os.path.join(d, 'frames.parquet')).sort_values('frame_idx').reset_index(drop=True)
    pose = np.stack([np.asarray(p, float).reshape(4, 4) for p in fr['world_from_vehicle']])
    off = np.nan_to_num(fr[['map_pose_offset_x', 'map_pose_offset_y']].to_numpy(float))
    ego = pose[:, :2, 3] + off
    x0, y0 = ego[0]
    yaw = np.arctan2(pose[:, 1, 0], pose[:, 0, 0])
    ctx = json.load(open(os.path.join(d, 'context.json'), encoding='utf-8'))

    feats = []
    for r in m.itertuples(index=False):
        xs, ys, zs = np.asarray(r.xs, float), np.asarray(r.ys, float), np.asarray(r.zs, float)
        pts = np.column_stack([xs - x0, ys - y0]).round(2).ravel().tolist()
        length = float(np.hypot(np.diff(xs), np.diff(ys)).sum()) if len(xs) > 1 else 0.0
        f = dict(id=int(r.feature_id), k=r.kind, p=pts)
        info = {'feature_id': int(r.feature_id), '类别': KIND_NAMES.get(r.kind, r.kind)}
        if r.kind == 'lane':
            f.update(st=int(r.subtype), it=bool(r.interpolating), en=[int(v) for v in r.entry_lanes],
                     ex=[int(v) for v in r.exit_lanes], ln=[int(v) for v in r.left_neighbors],
                     rn=[int(v) for v in r.right_neighbors],
                     bd=[int(v) for v in list(r.left_boundaries) + list(r.right_boundaries)])
            info.update({'类型 (type)': f'{int(r.subtype)} = {LANE_TYPES.get(int(r.subtype), "?")}',
                         '限速 (speed_limit_mph)': ('未给出' if not np.isfinite(r.speed_limit_mph) or r.speed_limit_mph == 0
                                                  else f'{r.speed_limit_mph:.0f} mph（{r.speed_limit_mph * 1.609:.0f} km/h）'),
                         '路口内的连接段 (interpolating)': '是' if r.interpolating else '否',
                         '从哪些车道来 (entry_lanes)': list(map(int, r.entry_lanes)) or '无',
                         '通往哪些车道 (exit_lanes)': list(map(int, r.exit_lanes)) or '无',
                         '左侧相邻车道 (left_neighbors)': list(map(int, r.left_neighbors)) or '无',
                         '右侧相邻车道 (right_neighbors)': list(map(int, r.right_neighbors)) or '无',
                         '左边界 (left_boundaries)': list(map(int, r.left_boundaries)) or '无',
                         '右边界 (right_boundaries)': list(map(int, r.right_boundaries)) or '无'})
        elif r.kind == 'road_line':
            f['st'] = int(r.subtype)
            info['类型 (type)'] = f'{int(r.subtype)} = {LINE_TYPES.get(int(r.subtype), "?")}'
        elif r.kind == 'road_edge':
            f['st'] = int(r.subtype)
            info['类型 (type)'] = f'{int(r.subtype)} = {EDGE_TYPES.get(int(r.subtype), "?")}'
        elif r.kind == 'stop_sign':
            f['sl'] = [int(v) for v in r.stop_sign_lanes]
            info['管控的车道 (lane)'] = list(map(int, r.stop_sign_lanes)) or '无'
        info['点数'] = len(xs)
        if r.kind != 'stop_sign':
            info['长度 / 周长'] = f'{length:.1f} m'
        if len(zs):
            info['高程 z'] = f'{zs.min():.1f} ~ {zs.max():.1f} m'
        f['info'] = info
        feats.append(f)

    # ---- replay: every labelled road user, frame by frame, in the same coordinates as the map ----
    obj = pd.read_parquet(os.path.join(d, 'objects.parquet'))
    obj = obj[obj['type'].isin(list(TCODE))]
    first = obj.sort_values('frame_idx').drop_duplicates('id')
    counters, labels, idx = {}, [], {}
    for oid, typ in zip(first['id'], first['type']):
        counters[typ] = counters.get(typ, 0) + 1
        idx[oid] = len(labels)
        labels.append(f'{PREFIX[typ]}{counters[typ]}')
    v_ego = np.hypot(fr['ego_vx'].to_numpy(float), fr['ego_vy'].to_numpy(float))
    by_k = dict(tuple(obj.groupby('frame_idx')))
    t0 = fr['timestamp_micros'].iloc[0]
    frames = []
    for i, k in enumerate(fr['frame_idx']):
        R, t = pose[i, :3, :3], pose[i, :3, 3]
        rows = []
        o = by_k.get(k)
        if o is not None:
            xy = (o[['cx', 'cy', 'cz']].to_numpy(float) @ R.T + t)[:, :2] + off[i] - [x0, y0]
            hd = o['heading'].to_numpy(float) + yaw[i]
            sp = np.hypot(o['speed_x'].to_numpy(float), o['speed_y'].to_numpy(float))
            for oid, typ, (x, y), h, L, W, s in zip(o['id'], o['type'], xy, hd, o['length'], o['width'], sp):
                rows.append([idx[oid], TCODE[typ], round(float(x), 2), round(float(y), 2), round(float(h), 3),
                             round(float(L), 2), round(float(W), 2), None if np.isnan(s) else round(float(s), 2)])
        frames.append(dict(t=round((fr['timestamp_micros'].iloc[i] - t0) / 1e6, 1),
                           e=[round(float(ego[i, 0] - x0), 2), round(float(ego[i, 1] - y0), 2),
                              round(float(yaw[i]), 4), round(float(v_ego[i]), 2)], o=rows))

    counts = m['kind'].value_counts().to_dict()
    return dict(seg=seg, location=ctx.get('location'), time_of_day=ctx.get('time_of_day'),
                counts={KIND_NAMES.get(k, k): int(v) for k, v in counts.items()},
                ego=(ego - [x0, y0]).round(2).ravel().tolist(), feats=feats,
                frames=frames, labels=labels, fp=[round(float(v), 3) for v in footprint_from_lidars(ctx)])


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Waymo 地图浏览</title>
<style>
:root { --bg:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.55 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }
header { background:var(--card); border-bottom:1px solid var(--line); padding:8px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
header h1 { font-size:16px; margin:0 8px 0 0; }
select, button { font:inherit; border:1px solid var(--line); background:#fff; border-radius:8px; padding:3px 10px; cursor:pointer; }
#tk { width:min(360px, 50vw); }
main { display:flex; gap:12px; padding:12px 16px; flex-wrap:wrap; }
#wrap { flex:1 1 640px; min-width:0; background:#fcfcfb; border:1px solid var(--line); border-radius:10px; position:relative; height:calc(100vh - 110px); min-height:420px; overflow:hidden; }
canvas { display:block; width:100%; height:100%; cursor:grab; }
#tip { position:absolute; pointer-events:none; background:rgba(20,20,20,0.88); color:#fff; font-size:12px; padding:4px 8px; border-radius:6px; display:none; white-space:nowrap; }
aside { flex:0 1 360px; min-width:0; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px; margin-bottom:12px; }
.card h2 { font-size:14px; margin:0 0 6px; }
.kv { display:grid; grid-template-columns:max-content 1fr; gap:2px 12px; font-size:13px; }
.kv b { color:var(--ink2); font-weight:500; }
.legend div { display:flex; align-items:center; gap:8px; font-size:13px; color:var(--ink2); margin:2px 0; }
.sw { width:26px; height:0; border-top:3px solid; display:inline-block; }
.bx { width:14px; height:9px; display:inline-block; border-radius:2px; }
.muted { color:var(--muted); font-size:12px; }
label.tg { display:inline-flex; gap:4px; align-items:center; margin-right:10px; font-size:13px; }
</style></head><body>
<header><h1>Waymo 高精地图 + 轨迹回放（原始数据）</h1>
<label>片段 <select id="seg"></select></label>
<button id="fit">全图</button>
<button id="play">▶ 播放</button>
<input type="range" id="tk" min="0" value="0"><span id="tlab" class="muted"></span>
<label>速度 <select id="spd"><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
<label class="tg"><input type="checkbox" id="trail" checked>轨迹尾巴（2 秒）</label>
<label class="tg"><input type="checkbox" id="follow">跟随自车</label>
</header>
<main>
<div id="wrap"><canvas id="cv"></canvas><div id="tip"></div></div>
<aside>
<div class="card"><h2>显示</h2><div id="tgs"></div><div id="cnt" class="muted" style="margin-top:6px"></div></div>
<div class="card legend"><h2>图例</h2>
<div><i class="bx" style="background:#141414"></i>自车（按雷达位置算出的车身）</div>
<div><i class="bx" style="background:#2a78d6"></i>车辆　<i class="bx" style="background:#eb6834"></i>行人　<i class="bx" style="background:#1baf7a"></i>骑车人</div>
<div><i class="sw" style="border-color:#9ec5f4"></i>车道中心线（虚线 = 路口内的连接段）</div>
<div><i class="sw" style="border-color:#a8a79f;border-top-style:dashed"></i>白色车道线　<i class="sw" style="border-color:#eda100"></i>黄色车道线</div>
<div><i class="sw" style="border-color:#52514e"></i>路缘　<i class="sw" style="border-color:#e1e0d9;border-top-width:8px"></i>人行横道 / 出入口</div>
<div><i class="sw" style="border-color:#e34948;border-top-width:6px;width:10px"></i>停车标志　<i class="sw" style="border-color:#4a3aa7"></i>自车整段轨迹</div>
<div class="muted" style="margin-top:6px">点一条车道：红 = 选中，深蓝 = 从哪来，橙 = 往哪去，紫色虚线 = 左右相邻车道。点一辆车 / 一个人：红线 = 它整段的轨迹。</div></div>
<div class="card"><h2>选中的对象</h2><div class="kv" id="info"><span class="muted">鼠标移上去看名字，点一下看详细信息。</span></div></div>
<div class="card muted">这些是真实记录的轨迹（人工标注的 3D 框，每秒 10 帧），不是模拟。只有离自车 75 m 以内的物体有标注，所以远处的车会突然出现或消失。坐标以自车起点为原点（米），x 向东、y 向北。</div>
</aside></main>
<script>
const SEGS = __DATA__;
const KINDS = [['lane','车道中心线'],['road_line','车道线'],['road_edge','路缘'],['crosswalk','人行横道'],['driveway','车库/出入口'],['speed_bump','减速带'],['stop_sign','停车标志'],['ego','自车整段轨迹'],['o0','车辆'],['o1','行人'],['o2','骑车人']];
const OCOL = ['#2a78d6', '#eb6834', '#1baf7a'], ONAME = ['车辆', '行人', '骑车人'];
const show = Object.fromEntries(KINDS.map(([k]) => [k, true]));
let S = null, byId = {}, tracks = [], sel = null, selObj = null, hover = null, k = 0, timer = null, view = {x: 0, y: 0, s: 3};
const $ = id => document.getElementById(id);
const cv = $('cv'), ctx = cv.getContext('2d'), tip = $('tip'), tk = $('tk');
SEGS.forEach((s, i) => { const o = document.createElement('option'); o.value = i; o.textContent = s.seg.slice(0, 20) + '… · ' + s.location + ' · ' + s.time_of_day; $('seg').appendChild(o); });
$('tgs').innerHTML = KINDS.map(([kk, n]) => '<label class="tg"><input type="checkbox" data-k="' + kk + '" checked>' + n + '</label>').join('');
document.querySelectorAll('#tgs input').forEach(c => c.onchange = () => { show[c.dataset.k] = c.checked; draw(); });
$('seg').onchange = e => load(+e.target.value);
$('fit').onclick = () => { fit(); draw(); };
$('trail').onchange = draw; $('follow').onchange = draw;
tk.oninput = () => { k = +tk.value; draw(); refreshObjInfo(); };
$('play').onclick = () => timer ? stop() : start();
$('spd').onchange = () => { if (timer) { stop(); start(); } };
function start() { $('play').textContent = '❚❚ 暂停'; timer = setInterval(() => { k = (k + 1) % S.frames.length; tk.value = k; draw(); refreshObjInfo(); }, 100 / +$('spd').value); }
function stop() { clearInterval(timer); timer = null; $('play').textContent = '▶ 播放'; }
function load(i) { S = SEGS[i]; byId = {}; S.feats.forEach(f => byId[f.id] = f); sel = null; selObj = null; hover = null; k = 0;
  tracks = S.labels.map(() => []); S.frames.forEach((F, fi) => F.o.forEach(o => tracks[o[0]].push([fi, o[2], o[3]])));
  tk.max = S.frames.length - 1; tk.value = 0;
  const nobj = [0, 0, 0]; S.labels.forEach(l => { nobj[l[0] === '车' ? 0 : l[0] === '人' ? 1 : 2]++; });
  $('cnt').textContent = '地图：' + Object.entries(S.counts).map(([a, b]) => a + ' ' + b).join('，') + '。整段出现过：车辆 ' + nobj[0] + '，行人 ' + nobj[1] + '，骑车人 ' + nobj[2] + '。';
  info(null); resize(); fit(); draw(); }
function resize() { const r = cv.getBoundingClientRect(); cv.width = r.width * devicePixelRatio; cv.height = r.height * devicePixelRatio; }
function fit() { let x0 = 1e9, x1 = -1e9, y0 = 1e9, y1 = -1e9;
  S.feats.forEach(f => { for (let i = 0; i < f.p.length; i += 2) { x0 = Math.min(x0, f.p[i]); x1 = Math.max(x1, f.p[i]); y0 = Math.min(y0, f.p[i + 1]); y1 = Math.max(y1, f.p[i + 1]); } });
  view.s = Math.min(cv.width / (x1 - x0 + 20), cv.height / (y1 - y0 + 20)); view.x = (x0 + x1) / 2; view.y = (y0 + y1) / 2; }
const W = (x, y) => [cv.width / 2 + (x - view.x) * view.s, cv.height / 2 - (y - view.y) * view.s];
const Wi = (px, py) => [(px - cv.width / 2) / view.s + view.x, -(py - cv.height / 2) / view.s + view.y];
function path(p, close) { ctx.beginPath(); for (let i = 0; i < p.length; i += 2) { const q = W(p[i], p[i + 1]); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); } if (close) ctx.closePath(); }
function stroke(p, col, w, dash, close) { path(p, close); ctx.strokeStyle = col; ctx.lineWidth = w * devicePixelRatio; ctx.setLineDash((dash || []).map(v => v * devicePixelRatio)); ctx.stroke(); ctx.setLineDash([]); }
function lineStyle(f) { const t = f.st; const yellow = t >= 4; const col = yellow ? '#eda100' : '#a8a79f';
  const dash = (t === 1 || t === 4 || t === 5) ? [7, 6] : []; const w = (t === 3 || t === 5 || t === 7 || t === 8) ? 2.6 : 1.3; return [col, w, dash]; }
function box(x, y, h, L, Wd, fill, edge, lw) { const mn = 4 * devicePixelRatio / view.s; L = Math.max(L, mn); Wd = Math.max(Wd, mn);
  const c = Math.cos(h), s = Math.sin(h); ctx.beginPath();
  [[L / 2, Wd / 2], [L / 2, -Wd / 2], [-L / 2, -Wd / 2], [-L / 2, Wd / 2]].forEach(([a, b], i) => { const q = W(x + a * c - b * s, y + a * s + b * c); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); });
  ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); if (edge) { ctx.strokeStyle = edge; ctx.lineWidth = lw * devicePixelRatio; ctx.stroke(); }
  const q1 = W(x + L / 2 * c, y + L / 2 * s), q0 = W(x + L / 4 * c, y + L / 4 * s);
  ctx.beginPath(); ctx.moveTo(q0[0], q0[1]); ctx.lineTo(q1[0], q1[1]); ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 1.2 * devicePixelRatio; ctx.stroke(); }
function draw() { if (!S) return; const F = S.frames[k];
  if ($('follow').checked) { view.x = F.e[0]; view.y = F.e[1]; }
  ctx.clearRect(0, 0, cv.width, cv.height);
  $('tlab').textContent = 't = ' + F.t.toFixed(1) + ' s（第 ' + k + ' 帧）· 自车 ' + F.e[3].toFixed(1) + ' m/s';
  const rel = new Map(); if (sel && sel.k === 'lane') { (sel.en || []).forEach(i => rel.set(i, 'en')); (sel.ex || []).forEach(i => rel.set(i, 'ex'));
    (sel.ln || []).concat(sel.rn || []).forEach(i => rel.set(i, 'nb')); (sel.bd || []).forEach(i => rel.set(i, 'bd')); }
  if (sel && sel.k === 'stop_sign') (sel.sl || []).forEach(i => rel.set(i, 'en'));
  for (const f of S.feats) { if (!show[f.k]) continue;
    if (f.k === 'driveway' || f.k === 'crosswalk' || f.k === 'speed_bump') { if (f.p.length < 6) continue; path(f.p, true);
      ctx.fillStyle = f.k === 'crosswalk' ? 'rgba(195,194,183,0.55)' : f.k === 'speed_bump' ? 'rgba(237,161,0,0.35)' : 'rgba(225,224,217,0.6)'; ctx.fill();
      if (f === hover || f === sel) stroke(f.p, '#e34948', 1.5, [], true); } }
  for (const f of S.feats) { if (!show[f.k]) continue;
    if (f.k === 'road_edge') stroke(f.p, rel.get(f.id) === 'bd' ? '#e34948' : '#52514e', f.st === 2 ? 1.2 : 1.6, f.st === 2 ? [3, 3] : []);
    if (f.k === 'road_line') { const [c, w, d] = lineStyle(f); stroke(f.p, rel.get(f.id) === 'bd' ? '#e34948' : c, w, d); } }
  for (const f of S.feats) { if (!show.lane || f.k !== 'lane') continue; const r = rel.get(f.id);
    let col = f.st === 3 ? '#1baf7a' : f.st === 0 ? '#c3c2b7' : '#9ec5f4', w = 1.2, d = f.it ? [3, 3] : [];
    if (r === 'en' || r === 'ex' || r === 'nb') { col = r === 'en' ? '#2a78d6' : r === 'ex' ? '#eb6834' : '#4a3aa7'; w = 3; d = r === 'nb' ? [6, 4] : []; }
    if (f === sel) { col = '#e34948'; w = 4; d = []; } else if (f === hover) { w = 3; }
    stroke(f.p, col, w, d); if (f === sel || f === hover || r === 'en' || r === 'ex') arrowHead(f.p, col); }
  if (show.stop_sign) S.feats.filter(f => f.k === 'stop_sign').forEach(f => { const q = W(f.p[0], f.p[1]); ctx.beginPath();
    for (let i = 0; i < 8; i++) { const a = Math.PI / 8 + i * Math.PI / 4, r = (f === sel || f === hover ? 9 : 6) * devicePixelRatio; i ? ctx.lineTo(q[0] + r * Math.cos(a), q[1] + r * Math.sin(a)) : ctx.moveTo(q[0] + r * Math.cos(a), q[1] + r * Math.sin(a)); }
    ctx.closePath(); ctx.fillStyle = '#e34948'; ctx.fill(); });
  if (show.ego) { stroke(S.ego, 'rgba(74,58,167,0.55)', 2, []); const q = W(S.ego[0], S.ego[1]); ctx.beginPath(); ctx.arc(q[0], q[1], 4 * devicePixelRatio, 0, 7); ctx.fillStyle = '#4a3aa7'; ctx.fill(); }
  if (selObj !== null) { const tr = tracks[selObj]; const p = []; tr.forEach(r => p.push(r[1], r[2])); stroke(p, '#e34948', 2.2, []); }
  if ($('trail').checked) for (const o of F.o) { if (!show['o' + o[1]]) continue; const tr = tracks[o[0]]; const p = [];
    tr.forEach(r => { if (r[0] <= k && r[0] >= k - 20) p.push(r[1], r[2]); }); if (p.length >= 4) stroke(p, OCOL[o[1]] + '88', 1.4, []); }
  for (const o of F.o) { if (!show['o' + o[1]]) continue; const hi = selObj === o[0] || (hover && hover.obj === o[0]);
    box(o[2], o[3], o[4], o[5], o[6], OCOL[o[1]], hi ? '#e34948' : '#fcfcfb', hi ? 2.5 : 0.6); }
  const fp = S.fp, e = F.e, cx = e[0] + Math.cos(e[2]) * (fp[0] + fp[1]) / 2, cy = e[1] + Math.sin(e[2]) * (fp[0] + fp[1]) / 2;
  box(cx, cy, e[2], fp[0] - fp[1], 2 * fp[2], '#141414', '#fcfcfb', 0.8);
}
function arrowHead(p, col) { if (p.length < 4) return; const n = p.length; const a = W(p[n - 4], p[n - 3]), b = W(p[n - 2], p[n - 1]);
  const ang = Math.atan2(b[1] - a[1], b[0] - a[0]), r = 9 * devicePixelRatio; ctx.beginPath(); ctx.moveTo(b[0], b[1]);
  ctx.lineTo(b[0] - r * Math.cos(ang - 0.45), b[1] - r * Math.sin(ang - 0.45)); ctx.lineTo(b[0] - r * Math.cos(ang + 0.45), b[1] - r * Math.sin(ang + 0.45)); ctx.closePath(); ctx.fillStyle = col; ctx.fill(); }
function nearestObj(px, py) { const [x, y] = Wi(px, py); let best = null, bd = 1e9;
  for (const o of S.frames[k].o) { if (!show['o' + o[1]]) continue; const d = Math.hypot(o[2] - x, o[3] - y); const lim = Math.max(o[5] / 2 + 0.5, 6 * devicePixelRatio / view.s);
    if (d < lim && d < bd) { bd = d; best = o; } } return best; }
function nearest(px, py) { const [x, y] = Wi(px, py); const tol = 7 * devicePixelRatio / view.s; let best = null, bd = tol;
  for (const f of S.feats) { if (!show[f.k]) continue; const p = f.p;
    if (f.k === 'stop_sign') { const d = Math.hypot(p[0] - x, p[1] - y); if (d < bd * 1.5) { bd = d; best = f; } continue; }
    for (let i = 0; i + 3 < p.length; i += 2) { const ax = p[i], ay = p[i + 1], dx = p[i + 2] - ax, dy = p[i + 3] - ay, L = dx * dx + dy * dy || 1e-9;
      const t = Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy) / L)); const d = Math.hypot(ax + t * dx - x, ay + t * dy - y);
      if (d < bd || (f.k === 'lane' && d < bd + 0.3 * tol && best && best.k !== 'lane')) { bd = d; best = f; } } }
  return best; }
function info(f) { const el = $('info'); if (!f) { el.innerHTML = '<span class="muted">鼠标移上去看名字，点一下看详细信息。</span>'; return; }
  el.innerHTML = Object.entries(f).map(([a, b]) => '<b>' + a + '</b><span>' + (Array.isArray(b) ? b.join(', ') : b) + '</span>').join(''); }
function objInfo(i) { const tr = tracks[i]; const cur = S.frames[k].o.find(o => o[0] === i); let len = 0;
  for (let j = 1; j < tr.length; j++) len += Math.hypot(tr[j][1] - tr[j - 1][1], tr[j][2] - tr[j - 1][2]);
  const lab = S.labels[i], ty = lab[0] === '车' ? 0 : lab[0] === '人' ? 1 : 2;
  return {'编号': lab, '类型': ONAME[ty], '这一帧的速度': cur && cur[7] !== null ? cur[7].toFixed(1) + ' m/s（' + (cur[7] * 3.6).toFixed(0) + ' km/h）' : '这一帧不在标注范围内',
    '尺寸（长 × 宽）': cur ? cur[5].toFixed(1) + ' × ' + cur[6].toFixed(1) + ' m' : '—', '出现的帧数': tr.length + ' / ' + S.frames.length,
    '出现的时间': S.frames[tr[0][0]].t.toFixed(1) + ' ~ ' + S.frames[tr[tr.length - 1][0]].t.toFixed(1) + ' s', '整段走过的路程': len.toFixed(1) + ' m'}; }
function refreshObjInfo() { if (selObj !== null) info(objInfo(selObj)); }
let drag = null;
cv.onmousedown = e => { drag = {x: e.clientX, y: e.clientY, vx: view.x, vy: view.y, moved: false}; cv.style.cursor = 'grabbing'; };
window.onmouseup = e => { if (drag && !drag.moved) { const r = cv.getBoundingClientRect(); const px = (e.clientX - r.left) * devicePixelRatio, py = (e.clientY - r.top) * devicePixelRatio;
    const o = nearestObj(px, py); if (o) { selObj = o[0]; sel = null; info(objInfo(selObj)); }
    else { const f = nearest(px, py); sel = f; selObj = null; info(f ? f.info : null); } draw(); }
  drag = null; cv.style.cursor = 'grab'; };
cv.onmousemove = e => { const r = cv.getBoundingClientRect(); const px = (e.clientX - r.left) * devicePixelRatio, py = (e.clientY - r.top) * devicePixelRatio;
  if (drag) { const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
    view.x = drag.vx - dx * devicePixelRatio / view.s; view.y = drag.vy + dy * devicePixelRatio / view.s; draw(); return; }
  const o = nearestObj(px, py); let label = null;
  if (o) { hover = {obj: o[0]}; label = S.labels[o[0]] + ' · ' + ONAME[o[1]] + (o[7] !== null ? ' · ' + o[7].toFixed(1) + ' m/s' : ''); }
  else { const f = nearest(px, py); hover = f; if (f) label = f.info['类别'] + ' #' + f.id + (f.info['类型 (type)'] ? ' · ' + f.info['类型 (type)'].split('= ')[1] : ''); }
  draw();
  if (label) { tip.style.display = 'block'; tip.style.left = (e.clientX - r.left + 12) + 'px'; tip.style.top = (e.clientY - r.top + 12) + 'px'; tip.textContent = label; } else tip.style.display = 'none'; };
cv.onmouseleave = () => { tip.style.display = 'none'; hover = null; draw(); };
cv.onwheel = e => { e.preventDefault(); const r = cv.getBoundingClientRect(); const px = (e.clientX - r.left) * devicePixelRatio, py = (e.clientY - r.top) * devicePixelRatio;
  const [wx, wy] = Wi(px, py); view.s *= Math.exp(-e.deltaY * 0.0015); const [nx, ny] = Wi(px, py); view.x += wx - nx; view.y += wy - ny; draw(); };
window.onresize = () => { resize(); draw(); };
load(0);
</script></body></html>
"""


def main():
    root, out = sys.argv[1], sys.argv[2]
    segs = sys.argv[3:] or sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)))
    data = [seg_data(root, s) for s in segs]
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        f.write(PAGE.replace('__DATA__', json.dumps(data, ensure_ascii=False, separators=(',', ':'))))
    print(f'{len(data)} segments, {os.path.getsize(out) / 1e6:.1f} MB -> {out}')


if __name__ == '__main__':
    main()
