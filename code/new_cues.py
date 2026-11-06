"""EXPLORATORY, decided after the words of the replies and the map were seen (user, 2026-11-05; CLAUDE.md "补线索"):
ten more cues per target frame, now, in the ego frame (x forward, y left, origin the rear axle, as wod_common).

 1 speed_limit_mps: the speed limit of the ego's lane (map.parquet speed_limit_mph x 0.44704). The lane is the one
   LaneMap.match_sequence gives for the frame, as in waymo_ssm_frames.process for the lead selection (checked against
   frame_ssm's ego_lane). Missing when the ego is on no mapped lane or the lane has no limit (NaN or 0).
 2 dist_crosswalk, 3 dist_stop_sign, 4 dist_lane_crossing: along the ego's lane forward from the ego's position on it
   (ego_s), through every chain of exit lanes (each branch of a fork), up to MAX_AHEAD = 100 m; the nearest over the
   branches. Crosswalk: the first point of the lane centre line inside a crosswalk polygon (0 when it starts inside
   one). Stop sign: a stop sign controlling a lane of the branch (stop_sign_lanes), its position projected onto that
   lane (wod_common.project), ahead of the ego. Lane crossing: the first point where the branch's centre line crosses
   another lane's centre line (a segment intersection) farther than CROSS_TOL from both ends of the other lane: a lane
   that only starts or ends on the branch (a split or a merge) does not count; the branch's own lanes are left out.
   Nothing within 100 m: 100. The ego on no mapped lane: missing.
 5 n_crossing_ped_cyc: pedestrians and cyclists with 0 <= x <= 30 m, |y| <= 10 m whose ground velocity in the ego frame
   has a lateral component of at least 0.5 m/s, larger than the longitudinal one (absolute values).
 6 n_crossing_vehicle: vehicles in the same box, heading 45-135 deg from the ego's (absolute), ground speed >= 1 m/s.
 7 n_oncoming_vehicle: vehicles with 0 <= x <= 50 m, |y| <= 10 m, heading more than 135 deg from the ego's, ground
   speed >= 1 m/s.
 8 n_ped_cyc_on_road: pedestrians and cyclists with 0 <= x <= 30 m, |y| <= 2.5 m.
 9 heading_change_deg: the ego's yaw now minus 2.0 s before (the frame nearest t - 2.0 s, within 0.15 s), wrapped,
   degrees, left positive.
 10 n_vehicle_moving_<R>m, n_vehicle_stationary_<R>m: count_nearby.py's vehicle counts (in the cameras' view, occluded
   ones included, the frame's lead left out, box centre within R = 10, 20, 30, 50 m) split by ground speed >= 1 m/s.
   Checked: moving + stationary = nearby_counts.csv's n_vehicle_<R>m on every frame (those counts have no speed rule,
   so stationary vehicles are in them).
Objects of 5-8 and 10 as count_nearby.py: objects_at(k, cameras=the five cameras), the lead of the target list left
out; position the box centre (cx, cy) and heading the box heading, both in the vehicle frame; velocity the label's
speed_x/y/z (global frame, wod_common.SPEED_FRAME) turned into the vehicle frame; ground speed its horizontal size.
Output <out_csv> (must not exist): one row per target frame; <out_csv minus .csv>_params.txt beside it; exit code 1 when
a check is not met.
usage: python -I new_cues.py <segments folder (vlm_inputs/segments)> <targets csv (vlm_frames/main_all9_targets.csv)>
       <frame_ssm.csv> <nearby_counts.csv> <out_csv>
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wod_common as wc  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
CAMERAS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']        # as count_nearby.py
TYPES = ['pedestrian', 'cyclist', 'vehicle']
RADII = [10, 20, 30, 50]
MAX_AHEAD = 100.0
CROSS_TOL = 1.0
MPH = 0.44704
MOVING = 1.0
HEADING_DT, HEADING_TOL = 2.0, 0.15
NEW = ['speed_limit_mps', 'dist_crosswalk', 'dist_stop_sign', 'dist_lane_crossing', 'n_crossing_ped_cyc',
       'n_crossing_vehicle', 'n_oncoming_vehicle', 'n_ped_cyc_on_road', 'heading_change_deg'] + \
      [f'n_vehicle_{m}_{r}m' for r in RADII for m in ('moving', 'stationary')]


# ------------------------------------------------------------------ geometry (tested on their own)
def subline(xy, s, a, b):
    """The polyline xy (arc lengths s) between arc lengths a < b, the end points interpolated."""
    a, b = max(a, 0.0), min(b, s[-1])
    inner = xy[(s > a) & (s < b)]
    pa = np.array([np.interp(a, s, xy[:, 0]), np.interp(a, s, xy[:, 1])])
    pb = np.array([np.interp(b, s, xy[:, 0]), np.interp(b, s, xy[:, 1])])
    return np.vstack([pa, inner, pb])


def forward_paths(lanes, lane0, s0, max_len=MAX_AHEAD):
    """Every chain of lanes forward from (lane0, s0) through exit lanes, cut at max_len (a lane is not entered twice in
    one chain). Returns [(xy, s, pieces)]: the branch's polyline from the ego's lane position, its arc length from 0,
    and pieces [(lane, a, b, i0, i1)] (the lane between its arc lengths a and b, rows i0..i1 of xy)."""
    out = []

    def rec(parts, pieces, length, lane, a, seen):
        L = lanes[lane]['s'][-1]
        b = min(L, a + (max_len - length))
        part = subline(lanes[lane]['xy'], lanes[lane]['s'], a, b)
        i0 = sum(len(p) for p in parts)
        parts2, pieces2 = parts + [part], pieces + [(lane, a, b, i0, i0 + len(part) - 1)]
        xy = np.vstack(parts2)
        length2 = float(wc.cumlen(xy)[-1])
        exits = [e for e in lanes[lane]['exits'] if e in lanes and e not in seen]
        if length2 >= max_len - 1e-6 or not exits or b < L - 1e-9:
            out.append((xy, wc.cumlen(xy), pieces2))
            return
        for e in exits:
            rec(parts2, pieces2, length2, e, 0.0, seen | {e})

    rec([], [], 0.0, lane0, s0, {lane0})
    return out


def seg_hits(P, Q):
    """Intersections of the segments of polyline P (n points) with segments Q0 -> Q1 (m x 2 x 2): rows (i, j, t, u),
    t along P's segment i, u along Q's segment j, both in [0, 1]; parallel segments and zero-length ones none."""
    if len(P) < 2 or len(Q) == 0:
        return np.zeros((0,), int), np.zeros((0,), int), np.zeros(0), np.zeros(0)
    P0, r = P[:-1], np.diff(P, axis=0)
    Q0, q = Q[:, 0], Q[:, 1] - Q[:, 0]
    den = r[:, None, 0] * q[None, :, 1] - r[:, None, 1] * q[None, :, 0]
    w = Q0[None] - P0[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (w[..., 0] * q[None, :, 1] - w[..., 1] * q[None, :, 0]) / den
        u = (w[..., 0] * r[:, None, 1] - w[..., 1] * r[:, None, 0]) / den
    ok = (np.abs(den) > 1e-12) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    i, j = np.nonzero(ok)
    return i, j, t[i, j], u[i, j]


def first_crosswalk(xy, s, polygons):
    """Arc length of the first point of the polyline inside one of the polygons (0 when it starts inside); NaN none."""
    best = np.nan
    for poly in polygons:
        if len(poly) < 3:
            continue
        if Path(poly).contains_point(xy[0]):
            return 0.0
        edges = np.stack([poly, np.roll(poly, -1, axis=0)], axis=1)
        i, j, t, u = seg_hits(xy, edges)
        if len(i):
            d = float((s[i] + t * (s[i + 1] - s[i])).min())
            best = d if np.isnan(best) else min(best, d)
    return best


def first_lane_crossing(xy, s, own, other_xy, tol=CROSS_TOL, min_angle=0.0, with_angle=False):
    """Arc length of the first proper crossing of the polyline with another lane's centre line: other_xy {lane: xy};
    lanes in `own` left out; a hit closer than tol to either end of the other lane does not count; with min_angle > 0
    (--min_cross_angle, user 2026-11-05) a hit where the two centre lines meet at less than min_angle degrees (their
    directions' angle folded to 0-90) does not count either. NaN none. with_angle: (arc length, that hit's angle)."""
    lo, hi = xy.min(0) - tol, xy.max(0) + tol
    best, best_ang = np.nan, np.nan
    for lane, q in other_xy.items():
        if lane in own or len(q) < 2:
            continue
        if (q.max(0) < lo).any() or (q.min(0) > hi).any():
            continue
        i, j, t, u = seg_hits(xy, np.stack([q[:-1], q[1:]], axis=1))
        if not len(i):
            continue
        pts = xy[i] + t[:, None] * (xy[i + 1] - xy[i])
        far = (np.hypot(*(pts - q[0]).T) > tol) & (np.hypot(*(pts - q[-1]).T) > tol)
        r_, q_ = xy[i + 1] - xy[i], q[j + 1] - q[j]
        cosv = np.abs((r_ * q_).sum(1)) / np.maximum(np.hypot(*r_.T) * np.hypot(*q_.T), 1e-12)
        ang = np.degrees(np.arccos(np.clip(cosv, 0.0, 1.0)))
        if min_angle > 0:
            far = far & (ang >= min_angle)
        if far.any():
            dd = s[i] + t * (s[i + 1] - s[i])
            k = int(np.flatnonzero(far)[np.argmin(dd[far])])
            if np.isnan(best) or dd[k] < best:
                best, best_ang = float(dd[k]), float(ang[k])
    return (best, best_ang) if with_angle else best


def first_stop_sign(xy, s, pieces, lanes, signs, s0):
    """Arc length to the projection of a stop sign onto a lane of the branch that it controls: signs [(point, lanes)];
    the projection onto the whole lane (wod_common.project), kept when it lies within the lane's piece (on the ego's
    lane: not behind the ego). NaN none."""
    best = np.nan
    for lane, a, b, i0, i1 in pieces:
        ln = lanes[lane]
        for p, ctl in signs:
            if lane not in ctl:
                continue
            sl = float(wc.project(ln['xy'], ln['s'], np.asarray(p, float)[None])[0][0])
            if a - 1e-9 <= sl <= b + 1e-9:
                d = float(s[i0]) + (sl - a)
                best = d if np.isnan(best) else min(best, d)
    return best


def events_ahead(lanes, lane0, s0, polys, signs, other_xy, max_len=MAX_AHEAD, min_angle=0.0, with_angle=False):
    """Cues 2-4 from the ego's lane position: the nearest over the branches; nothing within max_len -> max_len.
    Also the number of branches; with_angle: also cross_angle_deg, the angle at the crossing used (NaN if none)."""
    paths = forward_paths(lanes, lane0, s0, max_len)
    dc, ds, dx, da = [], [], [], []
    for xy, s, pieces in paths:
        own = {p[0] for p in pieces}
        dc.append(first_crosswalk(xy, s, polys))
        ds.append(first_stop_sign(xy, s, pieces, lanes, signs, s0))
        d_, a_ = first_lane_crossing(xy, s, own, other_xy, min_angle=min_angle, with_angle=True)
        dx.append(d_)
        da.append(a_)
    out = {}
    for name, v in (('dist_crosswalk', dc), ('dist_stop_sign', ds), ('dist_lane_crossing', dx)):
        v = np.array(v, float)
        v = v[~np.isnan(v) & (v <= max_len)]
        out[name] = float(v.min()) if len(v) else max_len
    out['n_branches'] = len(paths)
    if with_angle:
        v = np.array(dx, float)
        ok = ~np.isnan(v) & (v <= max_len)
        out['cross_angle_deg'] = float(np.array(da, float)[ok][np.argmin(v[ok])]) if ok.any() else np.nan
    return out


def object_counts(types, x, y, heading, vx, vy, speed):
    """Counts 5-8 from objects already in the ego frame (types; centre x, y; heading; velocity vx, vy; ground speed)."""
    pc = np.isin(types, ['pedestrian', 'cyclist'])
    veh = types == 'vehicle'
    box30 = (x >= 0) & (x <= 30) & (np.abs(y) <= 10)
    dh = np.degrees(np.abs(wc.wrap(heading)))
    return dict(n_crossing_ped_cyc=int((pc & box30 & (np.abs(vy) >= 0.5) & (np.abs(vy) > np.abs(vx))).sum()),
                n_crossing_vehicle=int((veh & box30 & (dh >= 45) & (dh <= 135) & (speed >= MOVING)).sum()),
                n_oncoming_vehicle=int((veh & (x >= 0) & (x <= 50) & (np.abs(y) <= 10) & (dh > 135) &
                                        (speed >= MOVING)).sum()),
                n_ped_cyc_on_road=int((pc & (x >= 0) & (x <= 30) & (np.abs(y) <= 2.5)).sum()))


def heading_change(yaw, t, k, dt=HEADING_DT, tol=HEADING_TOL):
    j = int(np.argmin(np.abs(t - (t[k] - dt))))
    if abs((t[k] - t[j]) - dt) > tol:
        return np.nan
    return float(np.degrees(wc.wrap(yaw[k] - yaw[j])))


def main():
    ap = argparse.ArgumentParser()
    for name in ('segments', 'targets', 'frame_ssm', 'nearby', 'out_csv'):
        ap.add_argument(name)
    ap.add_argument('--min_cross_angle', type=float, default=0.0,
                    help='user 2026-11-05: lane crossings at less than this angle (deg) do not count (0: all count)')
    ap.add_argument('--with_angle', action='store_true', help='also write cross_angle_deg, the angle at the crossing used')
    a = ap.parse_args()
    if os.path.exists(a.out_csv):
        sys.exit(f'{a.out_csv} exists already')
    t0 = time.time()
    tg = pd.read_csv(a.targets, comment='#')[['segment', 'frame_idx', 'lead_map']]
    fs = pd.read_csv(a.frame_ssm, usecols=['segment', 'frame_idx', 'ego_lane']).set_index(['segment', 'frame_idx'])
    nb = pd.read_csv(a.nearby).set_index(['segment', 'frame_idx'])
    rows, lane_bad, sum_bad = [], 0, 0
    signs_no_lane = n_signs = 0
    for sname, g in tg.groupby('segment', sort=True):
        seg = wc.Segment(a.segments, sname)
        lmap = wc.LaneMap(seg.map)
        if lmap.ok:
            ego_lanes, ego_s = lmap.match_sequence(seg.ego_map[:, :2], seg.yaw)
        else:
            ego_lanes, ego_s = [None] * len(seg.frames), np.full(len(seg.frames), np.nan)
        m = seg.map
        lanes_df = m[m['kind'] == 'lane'].set_index('feature_id')
        polys = [np.column_stack([r.xs, r.ys]).astype(float) for r in m[m['kind'] == 'crosswalk'].itertuples()]
        signs = [((r.xs[0], r.ys[0]), set(r.stop_sign_lanes)) for r in m[m['kind'] == 'stop_sign'].itertuples()]
        n_signs += len(signs)
        signs_no_lane += sum(1 for _, c in signs if not c)
        other_xy = {lid: v['xy'] for lid, v in lmap.lanes.items()} if lmap.ok else {}
        for r in g.itertuples():
            k = int(r.frame_idx)
            row = dict(segment=sname, frame_idx=k, ego_lane=ego_lanes[k])
            ref = fs.loc[(sname, k), 'ego_lane'] if (sname, k) in fs.index else np.nan
            same = (ego_lanes[k] is None and pd.isna(ref)) or (ego_lanes[k] is not None and not pd.isna(ref)
                                                                and int(ref) == int(ego_lanes[k]))
            lane_bad += int(not same)
            lane0 = ego_lanes[k]
            if lane0 is None or lane0 not in lmap.lanes:
                row.update(speed_limit_mps=np.nan, dist_crosswalk=np.nan, dist_stop_sign=np.nan,
                           dist_lane_crossing=np.nan, n_branches=0)
                if a.with_angle:
                    row['cross_angle_deg'] = np.nan
            else:
                lim = lanes_df.loc[lane0, 'speed_limit_mph'] if lane0 in lanes_df.index else np.nan
                row['speed_limit_mps'] = float(lim) * MPH if (not pd.isna(lim) and float(lim) > 0) else np.nan
                row.update(events_ahead(lmap.lanes, lane0, float(ego_s[k]), polys, signs, other_xy,
                                        min_angle=a.min_cross_angle, with_angle=a.with_angle))
            o = seg.objects_at(k, cameras=CAMERAS, types=TYPES)
            o = o[o['id'] != r.lead_map]
            v = o[['speed_x', 'speed_y', 'speed_z']].to_numpy(float)
            if wc.SPEED_FRAME == 'global':
                v = v @ seg.R[k]                      # global -> vehicle frame (R maps vehicle to global)
            speed = np.hypot(o['speed_x'].to_numpy(float), o['speed_y'].to_numpy(float))
            x, y = o['cx'].to_numpy(float), o['cy'].to_numpy(float)
            types = o['type'].to_numpy()
            row.update(object_counts(types, x, y, o['heading'].to_numpy(float), v[:, 0], v[:, 1], speed))
            row['heading_change_deg'] = heading_change(seg.yaw, seg.t, k)
            dist = np.hypot(x, y)
            for R in RADII:
                inside = (types == 'vehicle') & (dist <= R)
                row[f'n_vehicle_moving_{R}m'] = int((inside & (speed >= MOVING)).sum())
                row[f'n_vehicle_stationary_{R}m'] = int((inside & (speed < MOVING)).sum())
                tot = int(nb.loc[(sname, k), f'n_vehicle_{R}m'])
                sum_bad += int(row[f'n_vehicle_moving_{R}m'] + row[f'n_vehicle_stationary_{R}m'] != tot)
            row['speed_nan_objects'] = int(np.isnan(speed).sum())
            rows.append(row)
        print(f'{sname}: {len(g)} frames ({time.time() - t0:.0f} s)', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(a.out_csv, index=False)
    failed = []
    checks = [f'ego lane the same as frame_ssm\'s ego_lane: {len(out) - lane_bad} of {len(out)} frames',
              f'moving + stationary vehicles = nearby_counts.csv n_vehicle_<R>m: {len(out) * len(RADII) - sum_bad} of '
              f'{len(out) * len(RADII)}',
              f'objects without a speed: {int(out.speed_nan_objects.sum())}',
              f'stop signs {n_signs}, of them without a controlled lane {signs_no_lane}']
    if lane_bad:
        failed.append('ego lane')
    if sum_bad:
        failed.append('vehicle counts')
    with open(a.out_csv[:-4] + '_params.txt' if a.out_csv.endswith('.csv') else a.out_csv + '.params.txt', 'w',
              encoding='utf-8') as f:
        f.write(f'EXPLORATORY (user, 2026-11-05): new_cues.py run {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        for k_, v_ in (('segments', a.segments), ('targets', a.targets), ('frame_ssm', a.frame_ssm),
                       ('nearby counts', a.nearby)):
            f.write(f'{k_} = {os.path.abspath(v_)}\n')
        f.write(f'MAX_AHEAD {MAX_AHEAD} m, CROSS_TOL {CROSS_TOL} m, moving >= {MOVING} m/s, heading change over '
                f'{HEADING_DT} s (within {HEADING_TOL} s), mph {MPH} m/s, cameras {CAMERAS}\n')
        if a.min_cross_angle > 0 or a.with_angle:
            f.write(f'--min_cross_angle {a.min_cross_angle} deg (crossings at a smaller angle do not count), '
                    f'--with_angle {a.with_angle}\n')
        for c_ in checks:
            f.write(f'check: {c_}\n')
        f.write(f'checks not met: {failed if failed else "none"}\n')
    print(f'done in {time.time() - t0:.0f} s; {len(out)} frames; checks not met: {failed if failed else "none"}')
    for c_ in checks:
        print('  ' + c_)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
