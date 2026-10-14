"""Write a local HTML page to browse the raw Waymo annotations of one extracted segment.

Shows, frame by frame: the camera image with the 2D camera labels and the projected 3D labels, a
bird's-eye view with the map and every 3D-labelled object, and a table of all 3D labels in the frame.
Images are referenced from the extracted folder (file:// links), so the drive must be connected.

Usage: python -I make_scene_viewer.py <extracted_root> <out_dir> [segment ...]
"""
import json
import os
import sys
import urllib.parse

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
CAMS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
TYPE_CODE = {'vehicle': 0, 'pedestrian': 1, 'cyclist': 2, 'sign': 3, 'unknown': 4}
PREFIX = {'vehicle': '车', 'pedestrian': '人', 'cyclist': '骑', 'sign': '标', 'unknown': '?'}


def r(x, n=2):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def file_url(path):
    p = os.path.abspath(path).replace('\\', '/')
    return 'file:///' + urllib.parse.quote(p, safe='/:')


def build(root, seg):
    d = os.path.join(root, seg)
    ctx = json.load(open(os.path.join(d, 'context.json'), encoding='utf-8'))
    fr = pd.read_parquet(os.path.join(d, 'frames.parquet')).sort_values('frame_idx').reset_index(drop=True)
    pose = np.stack([np.asarray(p, float).reshape(4, 4) for p in fr['world_from_vehicle']])
    R = pose[:, :3, :3]
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    off = np.nan_to_num(fr[['map_pose_offset_x', 'map_pose_offset_y']].to_numpy(float))
    t0 = fr['timestamp_micros'].iloc[0]
    frames = []
    for i, f in fr.iterrows():
        frames.append(dict(k=int(f.frame_idx), ts=int(f.timestamp_micros), t=r((f.timestamp_micros - t0) / 1e6, 1),
                           v=r(np.hypot(f.ego_vx, f.ego_vy)),
                           mx=r(pose[i, 0, 3] + off[i, 0], 2), my=r(pose[i, 1, 3] + off[i, 1], 2),
                           gx=r(pose[i, 0, 3], 2), gy=r(pose[i, 1, 3], 2), yaw=r(yaw[i], 5)))
    kidx = {f['k']: i for i, f in enumerate(frames)}

    obj = pd.read_parquet(os.path.join(d, 'objects.parquet'))
    order = obj.sort_values(['frame_idx']).drop_duplicates('id')
    counters, label = {}, {}
    for oid, typ in zip(order['id'], order['type']):
        counters[typ] = counters.get(typ, 0) + 1
        label[oid] = f'{PREFIX.get(typ, "?")}{counters[typ]}'
    objs = [[] for _ in frames]
    for o in obj.itertuples():
        i = kidx[o.frame_idx]
        Rt = R[i].T
        v = Rt @ np.array([o.speed_x, o.speed_y, o.speed_z]) if not np.isnan(o.speed_x) else [np.nan] * 3
        a = Rt @ np.array([o.accel_x, o.accel_y, o.accel_z]) if not np.isnan(o.accel_x) else [np.nan] * 3
        objs[i].append([label[o.id], TYPE_CODE.get(o.type, 4), r(o.cx), r(o.cy), r(o.cz), r(o.length), r(o.width),
                        r(o.height), r(np.degrees(np.arctan2(np.sin(o.heading), np.cos(o.heading))), 1),
                        r(np.hypot(v[0], v[1])), r(v[0]), r(v[1]), r(a[0]), r(a[1]),
                        int(o.num_lidar_points), int(o.num_top_lidar_points), int(o.detection_difficulty),
                        int(o.tracking_difficulty), o.most_visible_camera or '', o.id])

    def boxes(name, with_label):
        b = pd.read_parquet(os.path.join(d, name))
        out = {c: [[] for _ in frames] for c in CAMS}
        for x in b.itertuples():
            if x.camera not in out:
                continue
            row = [TYPE_CODE.get(x.type, 4), r(x.box_cx, 1), r(x.box_cy, 1), r(x.box_w, 1), r(x.box_h, 1)]
            if with_label:
                lid = x.id[:-(len(x.camera) + 1)] if isinstance(x.id, str) and x.id.endswith('_' + x.camera) else x.id
                row.append(label.get(lid, ''))
            out[x.camera][kidx[x.frame_idx]].append(row)
        return out

    proj = boxes('projected_boxes.parquet', True)
    cam2d = boxes('camera_boxes.parquet', False)

    m = pd.read_parquet(os.path.join(d, 'map.parquet'))
    kinds = {'lane': 0, 'road_line': 1, 'road_edge': 2, 'crosswalk': 3, 'speed_bump': 4, 'driveway': 5, 'stop_sign': 6}
    mp = []
    for x in m.itertuples():
        pts = np.column_stack([x.xs, x.ys]).round(1).ravel().tolist()
        mp.append([kinds.get(x.kind, 0), pts])

    nlz = [[] for _ in frames]
    p = os.path.join(d, 'no_label_zones.parquet')
    if os.path.exists(p):
        for z in pd.read_parquet(p).itertuples():
            nlz[kidx[z.frame_idx]].append(np.column_stack([z.xs, z.ys]).round(1).ravel().tolist())

    imgs = {}
    for c in CAMS:
        folder = os.path.join(d, 'images', c)
        if os.path.isdir(folder):
            names = {int(n.split('_')[0]): n for n in os.listdir(folder) if n.endswith('.jpg')}
            imgs[c] = [file_url(os.path.join(folder, names[f['k']])) if f['k'] in names else '' for f in frames]
    calib = {c['name']: [c['width'], c['height']] for c in ctx['camera_calibrations']}
    kind_counts = m['kind'].value_counts().to_dict()
    info = dict(seg=seg, location=ctx.get('location'), time_of_day=ctx.get('time_of_day'), weather=ctx.get('weather'),
                n_frames=len(frames), laser_counts=ctx.get('laser_object_counts'),
                camera_counts=ctx.get('camera_object_counts'), map_counts=kind_counts,
                n_objects=int(obj['id'].nunique()), n_obj_by_type=obj.drop_duplicates('id')['type'].value_counts().to_dict(),
                n_nlz=int(sum(len(z) for z in nlz)))
    return dict(info=info, frames=frames, objs=objs, proj=proj, cam2d=cam2d, map=mp, nlz=nlz, imgs=imgs, calib=calib)


PAGE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Waymo 场景浏览</title>
<style>
:root { --bg:#f9f9f7; --card:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781; --line:#e1e0d9;
  --veh:#2a78d6; --ped:#eb6834; --cyc:#1baf7a; --sign:#898781; --hi:#e34948; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:14px/1.55 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif; }
header { background:var(--card); border-bottom:1px solid var(--line); padding:10px 16px; position:sticky; top:0; z-index:5; }
header h1 { font-size:16px; margin:0 0 6px; }
.ctrl { display:flex; flex-wrap:wrap; gap:10px; align-items:center; }
.ctrl input[type=range] { width:min(420px, 60vw); }
button, select { font:inherit; border:1px solid var(--line); background:#fff; border-radius:8px; padding:3px 10px; cursor:pointer; }
main { padding:12px 16px; max-width:1700px; margin:0 auto; }
.row { display:flex; gap:12px; flex-wrap:wrap; align-items:flex-start; }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px; margin-bottom:12px; }
#imgwrap { position:relative; flex:1 1 760px; min-width:0; }
#imgwrap img { width:100%; display:block; border-radius:6px; }
#imgwrap canvas { position:absolute; left:0; top:0; pointer-events:none; }
#bevcard { flex:0 0 592px; max-width:100%; }
#bev { border:1px solid var(--line); border-radius:6px; background:#fcfcfb; max-width:100%; }
.legend span { display:inline-flex; align-items:center; gap:4px; margin-right:12px; color:var(--ink2); }
.sw { width:12px; height:12px; border-radius:2px; display:inline-block; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { padding:3px 6px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
th { position:sticky; top:0; background:var(--card); color:var(--ink2); font-weight:600; cursor:pointer; }
td:first-child, th:first-child, td:nth-child(2), th:nth-child(2) { text-align:left; }
tr:hover td { background:#fff4e5; }
tr.sel td { background:#fde3e3; }
.tablewrap { max-height:420px; overflow:auto; }
.muted { color:var(--muted); }
.kv { display:grid; grid-template-columns:max-content 1fr; gap:2px 14px; }
details summary { cursor:pointer; color:var(--ink2); }
</style></head><body>
<header><h1 id="title"></h1>
<div class="ctrl">
 <button id="prev">◀</button><input type="range" id="slider" min="0" value="0"><button id="next">▶</button>
 <button id="play">播放</button><span id="finfo" class="muted"></span>
 <label>相机 <select id="cam"></select></label>
 <label><input type="checkbox" id="s3" checked> 3D 标注投影（实线）</label>
 <label><input type="checkbox" id="s2" checked> 相机 2D 标注（虚线）</label>
 <label>俯视范围 <select id="rng"><option value="40">40 m</option><option value="80" selected>80 m</option><option value="120">120 m</option></select></label>
</div></header>
<main>
<div class="card"><div class="kv" id="ctx"></div></div>
<div class="row">
 <div class="card" id="imgwrap"><img id="img" alt="相机画面"><canvas id="ov"></canvas></div>
 <div class="card" id="bevcard"><canvas id="bev" width="560" height="640"></canvas>
  <div class="legend"><span><i class="sw" style="background:var(--veh)"></i>车辆</span><span><i class="sw" style="background:var(--ped)"></i>行人</span>
  <span><i class="sw" style="background:var(--cyc)"></i>骑车人</span><span><i class="sw" style="background:var(--sign)"></i>标志</span>
  <span><i class="sw" style="background:#141414"></i>自车</span></div>
  <div class="muted">俯视图：向上是自车行驶方向。灰线是车道中心线，虚线是车道线，深色线是路缘，浅灰块是人行横道等，斜纹是"不标注区域"。</div></div>
</div>
<div class="card"><div id="counts" class="muted"></div>
 <label>类型 <select id="tf"><option value="-1">全部</option><option value="0">车辆</option><option value="1">行人</option><option value="2">骑车人</option><option value="3">标志</option></select></label>
 <span class="muted">点表头排序，点一行在图上高亮。</span>
 <div class="tablewrap"><table id="tab"><thead><tr>
  <th data-c="0">编号</th><th data-c="1">类型</th><th data-c="20">距离 m</th><th data-c="2">前方 x</th><th data-c="3">左侧 y</th><th data-c="4">高度 z</th>
  <th data-c="5">长</th><th data-c="6">宽</th><th data-c="7">高</th><th data-c="8">朝向°</th><th data-c="9">速度</th><th data-c="10">纵向速度</th><th data-c="11">横向速度</th>
  <th data-c="12">纵向加速度</th><th data-c="13">横向加速度</th><th data-c="14">激光点</th><th data-c="15">顶部激光点</th><th data-c="16">检测难度</th><th data-c="17">跟踪难度</th><th data-c="18">最清楚的相机</th>
 </tr></thead><tbody></tbody></table></div></div>
<div class="card"><details><summary>字段说明</summary><div class="kv" style="margin-top:6px">
 <b>编号</b><span>每个 3D 物体在整段里的跟踪编号（这里按类型重新编成 车1、人1……；原始编号是一串字母数字）</span>
 <b>前方 x / 左侧 y / 高度 z</b><span>物体 3D 框中心在自车坐标系里的位置（米）：x 向前，y 向左，z 向上</span>
 <b>长 / 宽 / 高</b><span>3D 框尺寸（米）</span>
 <b>朝向</b><span>物体车头方向相对自车车头方向的角度（度），0 = 同向，±180 = 对向</span>
 <b>速度、加速度</b><span>标注自带（原始为全局坐标系），这里换算成相对自车方向的纵向 / 横向分量（m/s，m/s²），是物体自己的速度，不是相对自车的速度</span>
 <b>激光点 / 顶部激光点</b><span>框内的激光雷达点数；点越少越难看清</span>
 <b>检测 / 跟踪难度</b><span>0 = 没有单独标难度；2 = LEVEL_2（一般是激光点很少，或标注员认为难）</span>
 <b>最清楚的相机</b><span>这个物体在哪个相机里最清楚；空白 = 不在任何相机画面里（例如在自车后方，Waymo 没有后视相机）</span>
 <b>3D 标注投影（实线）</b><span>激光雷达 3D 框投影到图像上，有编号，能对应到表里</span>
 <b>相机 2D 标注（虚线）</b><span>直接在图像上画的 2D 框，和 3D 标注是分开做的；远处 80 m 以外的车一般只有这种框，没有距离和速度</span>
 <b>不标注区域</b><span>这块区域里的物体没有标注（例如道路对面）</span>
</div></details></div>
</main>
<script>
const D = __DATA__;
const TYPES = ['车辆', '行人', '骑车人', '标志', '未知'];
const COL = ['#2a78d6', '#eb6834', '#1baf7a', '#898781', '#898781'];
let k = 0, cam = 'FRONT', sel = null, sortCol = 20, sortAsc = true, timer = null;
const $ = id => document.getElementById(id);
const I = D.info;
$('title').textContent = 'Waymo 场景浏览 · ' + I.seg;
const fmt = o => Object.entries(o || {}).map(([a, b]) => a + ' ' + b).join('，');
$('ctx').innerHTML = [['地点', I.location], ['时段', I.time_of_day], ['天气', I.weather], ['帧数', I.n_frames + '（10 帧/秒）'],
  ['整段出现过的 3D 物体', I.n_objects + '（' + fmt(I.n_obj_by_type) + '）'], ['官方统计：激光雷达标注', fmt(I.laser_counts)],
  ['官方统计：相机标注', fmt(I.camera_counts)], ['地图要素', fmt(I.map_counts)], ['不标注区域（各帧合计）', I.n_nlz]]
  .map(([a, b]) => '<b>' + a + '</b><span>' + b + '</span>').join('');
const slider = $('slider'); slider.max = D.frames.length - 1;
Object.keys(D.imgs).forEach(c => { const o = document.createElement('option'); o.value = o.textContent = c; $('cam').appendChild(o); });
$('cam').onchange = e => { cam = e.target.value; draw(); };
['s2', 's3'].forEach(id => $(id).onchange = drawOverlay);
$('rng').onchange = drawBev; $('tf').onchange = drawTable;
slider.oninput = () => { k = +slider.value; draw(); };
$('prev').onclick = () => { k = Math.max(0, k - 1); slider.value = k; draw(); };
$('next').onclick = () => { k = Math.min(D.frames.length - 1, k + 1); slider.value = k; draw(); };
$('play').onclick = () => { if (timer) { clearInterval(timer); timer = null; $('play').textContent = '播放'; return; }
  $('play').textContent = '暂停'; timer = setInterval(() => { k = (k + 1) % D.frames.length; slider.value = k; draw(); }, 150); };
document.querySelectorAll('th').forEach(th => th.onclick = () => { const c = +th.dataset.c; sortAsc = (sortCol === c) ? !sortAsc : true; sortCol = c; drawTable(); });
$('img').onload = drawOverlay;
window.onresize = drawOverlay;

function draw() {
  const f = D.frames[k];
  $('finfo').textContent = '第 ' + f.k + ' 帧 · t = ' + f.t + ' s · 自车 ' + f.v + ' m/s';
  const src = (D.imgs[cam] || [])[k]; if (src) $('img').src = src;
  drawBev(); drawTable(); drawOverlay();
}
function drawOverlay() {
  const img = $('img'), cv = $('ov');
  cv.width = img.clientWidth; cv.height = img.clientHeight;
  const ctx = cv.getContext('2d'); ctx.clearRect(0, 0, cv.width, cv.height);
  const wh = D.calib[cam] || [img.naturalWidth, img.naturalHeight]; const s = cv.width / wh[0];
  if ($('s2').checked) { ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
    (D.cam2d[cam] || [])[k].forEach(b => { ctx.strokeStyle = 'rgba(255,255,255,0.95)'; ctx.strokeRect((b[1] - b[3] / 2) * s, (b[2] - b[4] / 2) * s, b[3] * s, b[4] * s); });
    ctx.setLineDash([]); }
  if ($('s3').checked) { ctx.font = 'bold 12px system-ui,sans-serif';
    (D.proj[cam] || [])[k].forEach(b => { const hi = sel && b[5] === sel; ctx.lineWidth = hi ? 3.5 : 2; ctx.strokeStyle = hi ? '#e34948' : COL[b[0]];
      const x = (b[1] - b[3] / 2) * s, y = (b[2] - b[4] / 2) * s; ctx.strokeRect(x, y, b[3] * s, b[4] * s);
      if (b[5] && b[3] * s > 18) { ctx.fillStyle = 'rgba(20,20,20,0.75)'; const tw = ctx.measureText(b[5]).width + 6; ctx.fillRect(x, y - 15, tw, 15); ctx.fillStyle = '#fff'; ctx.fillText(b[5], x + 3, y - 3); } }); }
}
function drawBev() {
  const cv = $('bev'), ctx = cv.getContext('2d'), W = cv.width, H = cv.height, R = +$('rng').value;
  const sc = (H - 20) / (R * 1.25), ox = W / 2, oy = H - 20 - 0.25 * R * sc;
  const P = (x, y) => [ox - y * sc, oy - x * sc];
  ctx.clearRect(0, 0, W, H);
  const f = D.frames[k], c = Math.cos(-f.yaw), sn = Math.sin(-f.yaw);
  const toV = (X, Y, bx, by) => { const dx = X - bx, dy = Y - by; return [c * dx - sn * dy, sn * dx + c * dy]; };
  const sty = [['#c3c2b7', 0.8, []], ['#898781', 0.8, [4, 3]], ['#52514e', 1.4, []]];
  D.map.forEach(([kd, p]) => { if (kd >= 3) return; const [col, lw, dash] = sty[kd];
    ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.setLineDash(dash); ctx.beginPath();
    for (let i = 0; i < p.length; i += 2) { const v = toV(p[i], p[i + 1], f.mx, f.my); const q = P(v[0], v[1]); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); } ctx.stroke(); });
  ctx.setLineDash([]);
  D.map.forEach(([kd, p]) => { if (kd < 3 || kd > 5 || p.length < 6) return; ctx.fillStyle = 'rgba(195,194,183,0.45)'; ctx.beginPath();
    for (let i = 0; i < p.length; i += 2) { const v = toV(p[i], p[i + 1], f.mx, f.my); const q = P(v[0], v[1]); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); } ctx.closePath(); ctx.fill(); });
  (D.nlz[k] || []).forEach(p => { ctx.fillStyle = 'rgba(137,135,129,0.18)'; ctx.strokeStyle = 'rgba(137,135,129,0.6)'; ctx.beginPath();
    for (let i = 0; i < p.length; i += 2) { const v = toV(p[i], p[i + 1], f.gx, f.gy); const q = P(v[0], v[1]); i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]); } ctx.closePath(); ctx.fill(); ctx.stroke(); });
  const box = (x, y, l, w, hd) => { const ch = Math.cos(hd), sh = Math.sin(hd);
    return [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([a, b]) => P(x + a * l / 2 * ch - b * w / 2 * sh, y + a * l / 2 * sh + b * w / 2 * ch)); };
  const poly = (pts, fill, stroke, lw) => { ctx.beginPath(); pts.forEach((q, i) => i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])); ctx.closePath();
    if (fill) { ctx.fillStyle = fill; ctx.fill(); } if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); } };
  D.objs[k].forEach(o => { const hi = sel === o[0];
    poly(box(o[2], o[3], Math.max(o[5], 0.4), Math.max(o[6], 0.4), o[8] * Math.PI / 180), COL[o[1]], hi ? '#e34948' : '#fcfcfb', hi ? 3 : 0.8);
    if (hi) { const q = P(o[2], o[3]); ctx.fillStyle = '#e34948'; ctx.font = 'bold 13px system-ui,sans-serif'; ctx.fillText(o[0], q[0] + 8, q[1] - 6); } });
  poly(box(1.45, 0, 5.1, 2.0, 0), '#141414', null, 0);
  ctx.fillStyle = '#898781'; ctx.font = '11px system-ui,sans-serif';
  for (let d = 20; d <= R; d += 20) { const q = P(d, 0); ctx.fillText(d + ' m', 4, q[1]); ctx.strokeStyle = 'rgba(225,224,217,0.9)'; ctx.lineWidth = 0.5; ctx.beginPath(); ctx.moveTo(30, q[1]); ctx.lineTo(W, q[1]); ctx.stroke(); }
}
function drawTable() {
  const tf = +$('tf').value; const tb = document.querySelector('#tab tbody');
  let rows = D.objs[k].map(o => { const r = o.slice(); r[20] = Math.round(Math.hypot(o[2], o[3]) * 10) / 10; return r; }).filter(o => tf < 0 || o[1] === tf);
  rows.sort((a, b) => { const x = a[sortCol], y = b[sortCol]; const v = (x > y) - (x < y); return sortAsc ? v : -v; });
  const n = [0, 0, 0, 0, 0]; D.objs[k].forEach(o => n[o[1]]++);
  const n3 = ((D.proj[cam] || [])[k] || []).length, n2 = ((D.cam2d[cam] || [])[k] || []).length;
  $('counts').textContent = '这一帧的 3D 标注：车辆 ' + n[0] + '，行人 ' + n[1] + '，骑车人 ' + n[2] + '，标志 ' + n[3] +
    '。' + cam + ' 相机里：3D 标注投影 ' + n3 + ' 个，相机 2D 标注 ' + n2 + ' 个。';
  const v = x => x === null || x === undefined ? '' : x;
  tb.innerHTML = rows.map(o => '<tr data-id="' + o[0] + '"' + (sel === o[0] ? ' class="sel"' : '') + '><td>' + o[0] + '</td><td>' + TYPES[o[1]] + '</td><td>' + o[20] +
    '</td><td>' + o.slice(2, 18).map(v).join('</td><td>') + '</td><td>' + v(o[18]) + '</td></tr>').join('');
  tb.querySelectorAll('tr').forEach(tr => tr.onclick = () => { sel = sel === tr.dataset.id ? null : tr.dataset.id; drawBev(); drawTable(); drawOverlay(); });
}
draw();
</script></body></html>
"""


def main():
    root, out = sys.argv[1], sys.argv[2]
    segs = sys.argv[3:] or sorted(x for x in os.listdir(root) if os.path.isdir(os.path.join(root, x)))
    os.makedirs(out, exist_ok=True)
    for seg in segs:
        data = build(root, seg)
        js = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
        path = os.path.join(out, f'scene_viewer_{seg}.html')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(PAGE.replace('__DATA__', js))
        print(f'{seg}: {os.path.getsize(path) / 1e6:.1f} MB -> {path}')


if __name__ == '__main__':
    main()
