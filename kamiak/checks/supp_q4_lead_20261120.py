"""Follow-up on 20f9f43 (user, 2026-11-20; exploratory): Q4 frame 3270384983482134275_3220_000_3240_000 / 160, its lead.
Reads tables only, opens no image. Rebuilds the lead pairing of frame 160 with the repository's own functions
(wod_common: Segment, LaneMap, lead_map, lead_path, lateral_overlap, moving_tracks; waymo_ssm_frames.use_ego) and checks
that it gives the lead and gap recorded in results_lead_final_excl/frame_ssm_validation_training.csv.
  1. lead track over the whole segment: first/last frame, max speed, frames above 0.5 m/s (all frames; and the frames
     at least KIN_WINDOW // 2 from both track ends, which the slow-lead rule uses)
  2. ego speed at frames 140, 145, ..., 195 (the smoothed speed used in the SSMs; the Waymo-reported speed beside it)
  3. lead box relative to the reference line used for the pairing at frame 160, and each condition of the same-lane rule
  4. every other vehicle or cyclist ahead with a gap <= 30 m at frame 160, with each condition of the same-lane rule
Writes one long CSV: item, quantity, object_id, frame_idx, value, unit, note.
usage: python -I supp_q4_lead_20261120.py <out csv (must not exist)>"""
import os
import sys

import numpy as np
import pandas as pd

REPO = '/home/shuyi.wang/benchmark-claude-remote'
sys.path.insert(0, f'{REPO}/code')
import waymo_ssm_frames as wsf
import wod_common as wc

L = '/data/lab/cejli/sy/waymo_v143'
SEG, K = '3270384983482134275_3220_000_3240_000', 160
CAMS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']   # the default of waymo_ssm_frames.py
OUT = sys.argv[1]
if os.path.exists(OUT):
    sys.exit(f'{OUT} exists: stop')
rows, failed = [], []


def put(item, quantity, value, unit='', object_id='', frame_idx='', note=''):
    if isinstance(value, (float, np.floating)):
        value = repr(float(value))
    elif isinstance(value, (bool, np.bool_)):
        value = str(bool(value))
    elif isinstance(value, (int, np.integer)):
        value = str(int(value))
    rows.append(dict(item=item, quantity=quantity, object_id=object_id, frame_idx=frame_idx, value=value, unit=unit,
                     note=note))


def check(ok, msg):
    print(('ok    ' if ok else 'FAIL  ') + msg)
    if not ok:
        failed.append(msg)


def rel(a, b):
    return abs(a - b) / abs(b) if b else abs(a - b)


fs = pd.read_csv(f'{L}/results_lead_final_excl/frame_ssm_validation_training.csv', float_precision='round_trip',
                 low_memory=False)
fs = fs[fs.segment == SEG].set_index('frame_idx')
r160 = fs.loc[K]
lead_id = r160.lead_map
split = [d for d in ('training', 'validation') if os.path.isdir(f'{L}/tables/{d}/{SEG}')]
check(len(split) == 1, f'segment in one split: {split}')
seg = wc.Segment(f'{L}/tables/{split[0]}', SEG)
wsf.use_ego(*wc.footprint_from_lidars(seg.context))
n = len(seg.frames)
check(n == len(fs), f'{n} frames in frames.parquet = rows of frame_ssm for this segment ({len(fs)})')
print(f'ego footprint: front {wc.EGO_X_FRONT!r}, rear {wc.EGO_X_REAR!r}, half width {wc.EGO_HALF_W!r}')

# ---------------------------------------------------------------- 1. lead track
o = seg.objects
tr = o[o['id'] == lead_id].sort_values('frame_idx')
spd = np.hypot(tr.speed_x.to_numpy(float), tr.speed_y.to_numpy(float))
fr = tr.frame_idx.to_numpy(int)
half = wc.KIN_WINDOW // 2
inner = (fr - fr.min() >= half) & (fr.max() - fr >= half)
check(len(set(tr['type'])) == 1, f'lead track has one type: {sorted(set(tr["type"]))}')
check(np.all(np.diff(fr) == 1), f'lead tracked in consecutive frames ({len(fr)} frames, {fr.min()}-{fr.max()})')
i160 = int(np.flatnonzero(fr == K)[0])
check(rel(spd[i160], float(r160.v_lead)) <= 1e-13, f'lead speed at frame {K} {float(spd[i160])!r} = frame_ssm v_lead '
      f'{float(r160.v_lead)!r} (speed = magnitude of the horizontal label velocity, as wod_common.speed_and_accel)')
I = '1 lead speed over the segment'
put(I, 'lead type (Waymo 3D box class)', tr['type'].iloc[0], object_id=lead_id)
put(I, 'segment frames', n, 'frames', note='frame_idx 0 to n - 1')
put(I, 'first frame tracked', int(fr.min()), 'frame_idx', lead_id)
put(I, 'last frame tracked', int(fr.max()), 'frame_idx', lead_id)
put(I, 'frames tracked', len(fr), 'frames', lead_id, note='consecutive' if np.all(np.diff(fr) == 1) else 'with gaps')
put(I, 'max speed, all tracked frames', spd.max(), 'm/s', lead_id, int(fr[spd.argmax()]),
    'speed = hypot(speed_x, speed_y) of the label (global frame); frame_idx = frame of the max')
put(I, 'min speed, all tracked frames', spd.min(), 'm/s', lead_id, int(fr[spd.argmin()]))
put(I, 'frames with speed > 0.5 m/s, all tracked frames', int((spd > 0.5).sum()), 'frames', lead_id)
put(I, 'speed at frame 160', spd[i160], 'm/s', lead_id, K)
put(I, 'frames used by the slow-lead rule (>= 9 frames from both track ends)', int(inner.sum()), 'frames', lead_id,
    note=f'frames {fr[inner].min()}-{fr[inner].max()}; wod_common.moving_tracks, KIN_WINDOW // 2 = {half}')
put(I, 'max speed, frames used by the slow-lead rule', spd[inner].max(), 'm/s', lead_id, int(fr[inner][spd[inner].argmax()]))
put(I, 'frames with speed > 0.5 m/s, frames used by the slow-lead rule', int((spd[inner] > 0.5).sum()), 'frames', lead_id)
moving = wc.moving_tracks(seg)
put(I, 'counts as moving in the slow-lead rule (wod_common.moving_tracks)', lead_id in moving, '', lead_id)
put(I, 'lead_path pairing picks the same id at frame 160 (frame_ssm lead_path)', r160.lead_path == lead_id, '', lead_id, K,
    'the slow-lead rule keeps a map lead slower than 0.5 m/s if the path pairing picks it too, or it moves')
ex = pd.read_csv(f'{REPO}/labels/lead_exclusions.csv', dtype={'segment': str, 'id': str})
put(I, 'in labels/lead_exclusions.csv', bool(((ex.segment == SEG) & (ex['id'] == lead_id)).any()), '', lead_id)
put(I, 'num_lidar_points at frame 160', int(tr.num_lidar_points.iloc[i160]), 'points', lead_id, K)
print('lead speed by frame (all tracked frames):', ', '.join(f'{f}:{float(s)!r}' for f, s in zip(fr, spd)))

# ---------------------------------------------------------------- 2. ego speed
I = '2 ego speed'
mx = 0.0
for k in range(140, 200, 5):
    if k >= n:
        put(I, 'ego speed', 'no such frame', 'm/s', '', k, f'the segment has frames 0-{n - 1}')
        continue
    v = float(np.hypot(*seg.ego_v[k, :2]))
    vr = float(np.hypot(*seg.ego_v_reported[k, :2]))
    mx = max(mx, rel(v, float(fs.loc[k, 'ego_speed'])))
    edge = bool(k < half or k > n - 1 - half)
    put(I, 'ego speed (Savitzky-Golay, used in the SSMs)', v, 'm/s', '', k,
        f'edge_ego {edge}' + ('; within 9 frames of the segment end, less accurate' if edge else ''))
    put(I, 'ego speed as reported by Waymo (check only)', vr, 'm/s', '', k)
check(mx <= 1e-13, f'ego speeds equal frame_ssm ego_speed (max relative difference {mx!r})')

# ---------------------------------------------------------------- 3. lead box at frame 160 (pairing rebuilt)
lmap = wc.LaneMap(seg.map)
ego_lanes, ego_s = lmap.match_sequence(seg.ego_map[:, :2], seg.yaw) if lmap.ok else ([None] * n, np.full(n, np.nan))
o_all = seg.objects_at(K, cameras=CAMS, types=['vehicle', 'pedestrian', 'cyclist'])
o_rng = o_all[wc.box_nearest_dist(o_all['cx'], o_all['cy'], o_all['length'], o_all['width'], o_all['heading'])
              <= wc.LABEL_RANGE]
o_lead = o_rng[o_rng['type'].isin(wc.LEAD_TYPES)].reset_index(drop=True)
r = wc.lead_map(seg, K, o_lead, lmap, ego_lanes, ego_s)
check(r is not None and o_lead.loc[r['i'], 'id'] == lead_id and rel(r['gap'], float(r160.gap)) <= 1e-13,
      f'rebuilt map pairing at frame {K} picks {o_lead.loc[r["i"], "id"] if r else None} with gap '
      f'{float(r["gap"]) if r else None!r} (frame_ssm: {lead_id}, {float(r160.gap)!r})')
check(bool(r['fallback']) == bool(r160.map_fallback), f'fallback {r["fallback"]} = frame_ssm map_fallback {r160.map_fallback}')
lane0 = ego_lanes[K]
if lane0 is None:
    xy, s_ego, ref = wc.ego_future_path(seg, K), 0.0, 'the ego future path (no ego lane matched; lead_map falls back to lead_path)'
else:
    _, xy = lmap.route(lane0, ego_s[K], [l for l in ego_lanes[K:] if l is not None])
    s_ego, ref = float(ego_s[K]), f'the ego lane route from map lane {lane0}'
I = '3 lead box at frame 160'
put(I, 'ego lane matched at frame 160', 'none' if lane0 is None else lane0, '', '', K,
    'LaneMap.match_sequence (lane centre line within 2.5 m of the ego origin, heading within 45 deg)')
put(I, 'reference line used for the pairing', ref, '', '', K)
# why no lane: nearest lane segment to the ego origin, and its heading difference
p_ego, h_ego = seg.ego_map[K, :2], seg.yaw[K]
t = np.clip(((p_ego - lmap.A) * lmap.AB).sum(1) / lmap.L2, 0, 1)
dist = np.hypot(*(p_ego - (lmap.A + t[:, None] * lmap.AB)).T)
dh_l = np.degrees(np.abs(wc.wrap(lmap.hdg - h_ego)))
j = int(dist.argmin())
put(I, 'nearest map lane centre line to the ego origin: distance', dist[j], 'm', '', K,
    f'lane {lmap.lane_of[j]}; heading difference {float(dh_l[j])!r} deg; the matching needs < 2.5 m and < 45 deg')
ok_h = dh_l < 45
if ok_h.any():
    jj = int(np.flatnonzero(ok_h)[dist[ok_h].argmin()])
    put(I, 'nearest map lane centre line within 45 deg of the ego heading: distance', dist[jj], 'm', '', K,
        f'lane {lmap.lane_of[jj]}; heading difference {float(dh_l[jj])!r} deg')
put(I, 'map lanes in this segment', len(lmap.lanes), 'lanes')
s_cum = wc.cumlen(xy)
P, hdg, v, a = seg.to_map(K, o_lead)
s, d, ph = wc.project(xy, s_cum, P)
dh = wc.wrap(hdg - ph)
s_rel = s - s_ego
Lb = o_lead['length'].to_numpy(float)
gap = s_rel - Lb / 2 - wc.EGO_X_FRONT
ov = wc.lateral_overlap(seg, K, xy, P, hdg, o_lead)
ext = wc.extend_back(xy, 10.0)
s_ext = wc.cumlen(ext)
_, dce, _ = wc.project(ext, s_ext, wc.ego_centre_map(seg, K))
_, dc, _ = wc.project(ext, s_ext, wc.box_corners(P[:, 0], P[:, 1], o_lead['length'], o_lead['width'], hdg).reshape(-1, 2))
dc = dc.reshape(-1, 4)
i = int(r['i'])
put(I, 'lead box centre: signed lateral offset from the reference line (left +)', d[i], 'm', lead_id, K,
    'same-lane rule: |offset| < 1.75 m (wod_common.LANE_HALF)')
put(I, 'lead box corners: lateral offsets, min', dc[i].min(), 'm', lead_id, K, 'left +')
put(I, 'lead box corners: lateral offsets, max', dc[i].max(), 'm', lead_id, K, 'left +')
put(I, 'ego box centre: lateral offset', dce[0], 'm', '', K, f'ego interval = this +- half width {wc.EGO_HALF_W!r} m')
put(I, 'ego interval, min', dce[0] - wc.EGO_HALF_W, 'm', '', K)
put(I, 'ego interval, max', dce[0] + wc.EGO_HALF_W, 'm', '', K)
put(I, 'lead heading minus reference-line heading', np.degrees(dh[i]), 'deg', lead_id, K, 'same-lane rule: |.| < 30 deg')
put(I, 'lead centre along the reference line, ahead of the ego origin', s_rel[i], 'm', lead_id, K, 'rule: > 0')
put(I, 'gap (ego front bumper to lead rear, along the line)', gap[i], 'm', lead_id, K, 'rule: > 0')
put(I, 'lead box length', Lb[i], 'm', lead_id, K)
put(I, 'lead box width', float(o_lead['width'].iloc[i]), 'm', lead_id, K)
put(I, 'condition |lateral offset| < 1.75 m', bool(abs(d[i]) < wc.LANE_HALF), '', lead_id, K)
put(I, 'condition |heading difference| < 30 deg', bool(abs(dh[i]) < wc.LEAD_MAX_DHEADING), '', lead_id, K)
put(I, 'condition ahead and gap > 0', bool(s_rel[i] > 0 and gap[i] > 0), '', lead_id, K)
put(I, 'condition lateral overlap with the ego interval', bool(ov[i]), '', lead_id, K)
put(I, 'inside the ego lane by the lead-selection rule (all four conditions)',
    bool(abs(d[i]) < wc.LANE_HALF and abs(dh[i]) < wc.LEAD_MAX_DHEADING and s_rel[i] > 0 and gap[i] > 0 and ov[i]), '',
    lead_id, K, 'the rule has no lane polygon: it is these four conditions on the reference line')
check(rel(gap[i], float(r160.gap)) <= 1e-13, 'gap rebuilt by hand = frame_ssm gap')

# ---------------------------------------------------------------- 4. other road users ahead within 30 m
I = '4 other vehicles ahead within 30 m at frame 160'
allv = seg.objects_at(K, types=list(wc.LEAD_TYPES)).reset_index(drop=True)
vis = set(o_lead['id'])
P2, h2, v2, _ = seg.to_map(K, allv)
s2, d2, ph2 = wc.project(xy, s_cum, P2)
dh2 = wc.wrap(h2 - ph2)
sr2 = s2 - s_ego
g2 = sr2 - allv['length'].to_numpy(float) / 2 - wc.EGO_X_FRONT
ov2 = wc.lateral_overlap(seg, K, xy, P2, h2, allv)
near = (sr2 > 0) & (g2 > 0) & (g2 <= 30.0) & (allv['id'] != lead_id).to_numpy()
inlane = (np.abs(d2) < wc.LANE_HALF) & (np.abs(dh2) < wc.LEAD_MAX_DHEADING) & (sr2 > 0) & (g2 > 0) & ov2
nin = 0
for q in np.flatnonzero(near):
    oid = allv.loc[q, 'id']
    put(I, 'object', f'type {allv.loc[q, "type"]}; gap {float(g2[q])!r} m; lateral offset {float(d2[q])!r} m; heading difference '
        f'{float(np.degrees(dh2[q]))!r} deg; lateral overlap {bool(ov2[q])}; speed {float(np.hypot(*v2[q]))!r} m/s; '
        f'in a camera view and within 77 m {oid in vis}; same-lane rule {bool(inlane[q])}', '', oid, K)
    nin += bool(inlane[q]) and oid in vis
put(I, 'labelled vehicles and cyclists ahead (0 < gap <= 30 m), lead excluded',
    int(near.sum()), 'objects', '', K, 'gap measured along the reference line, as for the lead')
put(I, 'of these, in the ego lane by the lead-selection rule and in a camera view', nin, 'objects', '', K)
put(I, 'of these, in the ego lane by the rule ignoring camera visibility', int((near & inlane).sum()), 'objects', '', K)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
pd.DataFrame(rows).to_csv(OUT, index=False)
print(pd.DataFrame(rows).to_string(max_colwidth=230))
print(f'wrote {OUT}')
print(f'checks not met: {len(failed)}', *failed, sep='\n  ')
