"""EXPLORATORY, decided after the residuals of D's additive model were seen (user, 2026-11-06; CLAUDE.md "我定（2026-11-06…）"
item 1): the cues of round E per target frame, now, in the ego frame (x forward, y left, origin the rear axle), with
new_cues.py's conventions: objects in the five cameras' view (Segment.objects_at, as count_nearby.py), the frame's lead
(lead_map of the target list) left out; position the box centre; velocity the label's speed_x/y/z (global frame,
wod_common.SPEED_FRAME) turned into the vehicle frame; ground speed its horizontal size; map-frame centres and headings
by Segment.to_map; the ego's lane by LaneMap.match_sequence (as waymo_ssm_frames.py and new_cues.py).
 1 n_side_vehicle: vehicles with -5 <= x <= 15 m and 1.5 m < |y| <= 5.0 m; n_side_vehicle_toward: of them the ones
   whose lateral ground velocity points to the ego's lane side and is at least 0.3 m/s (vy <= -0.3 when y > 0,
   vy >= 0.3 when y < 0).
 2 dist_merge: along the ego's lane forward (new_cues.forward_paths: every branch through the exit lanes, up to 100 m),
   the arc length of the first point where another lane ends within MERGE_TOL = 1.0 m of the branch: the other lane's
   last point projected onto the branch (wod_common.project), the foot strictly inside the branch (not clamped to
   either end: a lane ending behind the ego, say the ego lane's own predecessor, or beyond the branch's end does not
   count); the branch's own lanes left out (the "merge" new_cues.py leaves out of the lane crossings). The nearest over
   the branches; none within 100 m: 100; the ego on no mapped lane: missing.
 3 large vehicles: not computed here (the user looks at the box lengths first).
 4 n_ped_cyc_in_crosswalk_50m: pedestrians and cyclists within 50 m whose box centre (map frame) is inside a crosswalk
   polygon of the map (matplotlib Path.contains_point).
 5 n_ped_cyc_road_<R>m, n_ped_cyc_offroad_<R>m: count_nearby.py's pedestrians + cyclists within R = 10, 20, 30, 50 m
   split by the distance of the box centre (map frame) to the nearest lane centre line (every lane of the map, as
   LaneMap keeps them: bike lanes and the interpolated lanes in intersections included): <= 2.0 m on the road. Map
   without a lane: missing.
 6 n_vehicle_stationary_inlane_<R>m, n_vehicle_stationary_other_<R>m: new_cues.py's stationary vehicles (ground speed
   < 1 m/s) within R split: in a lane = box centre within 1.5 m of the nearest lane centre line and the box heading
   within 30 deg of that line's direction there (the nearest segment of the nearest lane); otherwise elsewhere.
 7 heading_change_next3s_deg: the ego's yaw 3.0 s later minus now (the frame nearest t + 3.0 s, within 0.15 s; none:
   missing), wrapped, degrees, left positive.
Checks: the ego lane the same as frame_ssm's ego_lane; road + offroad = nearby_counts.csv's n_ped_cyc_<R>m and inlane +
other = n_vehicle_stationary_<R>m of D's new cues, on the frames those files have; toward <= side; the nearest-lane
distance recomputed another way (wod_common.project on every lane) for the objects of N_SECOND random frames (the
directions only reported: where the nearest point is a polyline vertex, the two methods may take either segment);
objects without a speed.
Output <out_csv> (must not exist): one row per target frame (split, in_main, closing_below_0_1 kept from the list);
<out_csv minus .csv>_params.txt beside it; exit code 1 when a check is not met.
usage: python -I new_cues_e.py <segments folder> <targets csv (vlm_frames/comparison_lead7_targets.csv)> <frame_ssm.csv>
       <nearby_counts.csv> <D's new cues csv (new_cues_min30.csv)> <out_csv>
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import new_cues as nc  # noqa: E402
import wod_common as wc  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
CAMERAS, TYPES, RADII = nc.CAMERAS, nc.TYPES, nc.RADII
MOVING = nc.MOVING
SIDE_X, SIDE_Y, SIDE_VY = (-5.0, 15.0), (1.5, 5.0), 0.3
MERGE_TOL = 1.0
CW_RADIUS = 50.0
ROAD_TOL = 2.0
INLANE_D, INLANE_DH = 1.5, 30.0
FUTURE_DT, FUTURE_TOL = 3.0, 0.15
N_SECOND, SEED_SECOND = 10, 20261106
SPLIT_COLS = [f'n_ped_cyc_{w}_{r}m' for r in RADII for w in ('road', 'offroad')] + \
    [f'n_vehicle_stationary_{w}_{r}m' for r in RADII for w in ('inlane', 'other')]
NEW = ['n_side_vehicle', 'n_side_vehicle_toward', 'dist_merge', 'n_ped_cyc_in_crosswalk_50m'] + SPLIT_COLS + \
    ['heading_change_next3s_deg']


# ------------------------------------------------------------------ pieces (tested on their own)
def side_counts(types, x, y, vy):
    """Item 1 from objects in the ego frame: (vehicles beside, of them moving toward the ego's lane)."""
    box = (types == 'vehicle') & (x >= SIDE_X[0]) & (x <= SIDE_X[1]) & (np.abs(y) > SIDE_Y[0]) & \
        (np.abs(y) <= SIDE_Y[1])
    toward = box & (((y > 0) & (vy <= -SIDE_VY)) | ((y < 0) & (vy >= SIDE_VY)))
    return int(box.sum()), int(toward.sum())


def first_merge(xy, s, own, lanes, tol=MERGE_TOL):
    """Arc length along the branch (xy, s) of the first end point of another lane within tol of it, the foot of the
    projection strictly inside the branch; lanes in `own` left out. NaN none."""
    lo, hi = xy.min(0) - tol, xy.max(0) + tol
    best = np.nan
    for lane, ln in lanes.items():
        if lane in own:
            continue
        e = ln['xy'][-1]
        if (e < lo).any() or (e > hi).any():
            continue
        se, de, _ = wc.project(xy, s, e[None])
        if abs(de[0]) <= tol and 1e-9 < se[0] < s[-1] - 1e-9:
            best = float(se[0]) if np.isnan(best) else min(best, float(se[0]))
    return best


def merge_ahead(lanes, lane0, s0, max_len=nc.MAX_AHEAD):
    """Item 2: the nearest merge over the branches forward from (lane0, s0); none within max_len -> max_len. Also the
    number of branches."""
    paths = nc.forward_paths(lanes, lane0, s0, max_len)
    v = np.array([first_merge(xy, s, {p[0] for p in pieces}, lanes) for xy, s, pieces in paths], float)
    v = v[~np.isnan(v) & (v <= max_len)]
    return (float(v.min()) if len(v) else max_len), len(paths)


def nearest_lane(lmap, P, chunk=16):
    """Distance of each point P (M x 2, map frame) to the nearest lane centre line of the LaneMap (every segment of every
    lane) and that segment's direction; NaN without lanes."""
    d = np.full(len(P), np.nan)
    h = np.full(len(P), np.nan)
    if not lmap.ok or not len(P):
        return d, h
    for a in range(0, len(P), chunk):
        p = P[a:a + chunk]
        t = np.clip(((p[:, None, :] - lmap.A[None]) * lmap.AB[None]).sum(-1) / lmap.L2[None], 0, 1)
        q = lmap.A[None] + t[..., None] * lmap.AB[None]
        dd = np.hypot(p[:, None, 0] - q[..., 0], p[:, None, 1] - q[..., 1])
        j = dd.argmin(1)
        d[a:a + chunk] = dd[np.arange(len(p)), j]
        h[a:a + chunk] = lmap.hdg[j]
    return d, h


def split_counts(types, dist, speed, dlane, hdg_obj, hdg_lane):
    """Items 5 and 6 from the objects' types, distances to the ego, ground speeds, nearest-lane distances and the
    headings (object, nearest lane segment; map frame). Missing (NaN) when an object has no lane distance."""
    if np.isnan(dlane).any():
        return {c: np.nan for c in SPLIT_COLS}
    pc = np.isin(types, ['pedestrian', 'cyclist'])
    road = dlane <= ROAD_TOL
    stat = (types == 'vehicle') & (speed < MOVING)
    inl = stat & (dlane <= INLANE_D) & (np.abs(wc.wrap(hdg_obj - hdg_lane)) <= np.radians(INLANE_DH))
    out = {}
    for R in RADII:
        near = dist <= R
        out[f'n_ped_cyc_road_{R}m'] = int((pc & near & road).sum())
        out[f'n_ped_cyc_offroad_{R}m'] = int((pc & near & ~road).sum())
        out[f'n_vehicle_stationary_inlane_{R}m'] = int((inl & near).sum())
        out[f'n_vehicle_stationary_other_{R}m'] = int((stat & ~inl & near).sum())
    return out


def in_crosswalk(types, dist, P, polys, radius=CW_RADIUS):
    """Item 4: pedestrians and cyclists within radius whose map-frame centre P is inside one of the polygons."""
    pc = np.isin(types, ['pedestrian', 'cyclist']) & (dist <= radius)
    paths = [Path(poly) for poly in polys if len(poly) >= 3]
    n = 0
    for p in P[pc]:
        n += int(any(pa.contains_point(p) for pa in paths))
    return n


def heading_change_next(yaw, t, k, dt=FUTURE_DT, tol=FUTURE_TOL):
    """Item 7: yaw at the frame nearest t[k] + dt minus yaw now (degrees, wrapped, left positive); NaN when that frame
    is farther than tol from t[k] + dt."""
    j = int(np.argmin(np.abs(t - (t[k] + dt))))
    if abs((t[j] - t[k]) - dt) > tol:
        return np.nan
    return float(np.degrees(wc.wrap(yaw[j] - yaw[k])))


def nearest_lane_second_way(lmap, P):
    """nearest_lane another way: wod_common.project of the points onto every lane polyline, the smallest |offset|."""
    best_d = np.full(len(P), np.inf)
    best_h = np.full(len(P), np.nan)
    for ln in lmap.lanes.values():
        _, dd, hh = wc.project(ln['xy'], ln['s'], P)
        better = np.abs(dd) < best_d
        best_d[better] = np.abs(dd)[better]
        best_h[better] = hh[better]
    return best_d, best_h


def main():
    ap = argparse.ArgumentParser()
    for name in ('segments', 'targets', 'frame_ssm', 'nearby', 'd_cues', 'out_csv'):
        ap.add_argument(name)
    a = ap.parse_args()
    if os.path.exists(a.out_csv):
        sys.exit(f'{a.out_csv} exists already')
    t0 = time.time()
    tg = pd.read_csv(a.targets, comment='#')
    keep = [c for c in ('split', 'segment', 'frame_idx', 'lead_map', 'in_main', 'closing_below_0_1') if c in tg.columns]
    tg = tg[keep]
    fs = pd.read_csv(a.frame_ssm, usecols=['segment', 'frame_idx', 'ego_lane']).set_index(['segment', 'frame_idx'])
    nb = pd.read_csv(a.nearby).set_index(['segment', 'frame_idx'])
    dc = pd.read_csv(a.d_cues).set_index(['segment', 'frame_idx'])
    rng = np.random.default_rng(SEED_SECOND)
    second = set(map(tuple, tg[['segment', 'frame_idx']].to_numpy()[rng.choice(len(tg), N_SECOND, replace=False)]))
    rows = []
    lane_bad = sum_pc_bad = sum_st_bad = n_pc_cmp = n_st_cmp = 0
    second_dev_d = second_dev_h = 0.0
    second_n = second_h_diff = 0
    for sname, g in tg.groupby('segment', sort=True):
        seg = wc.Segment(a.segments, sname)
        lmap = wc.LaneMap(seg.map)
        if lmap.ok:
            ego_lanes, ego_s = lmap.match_sequence(seg.ego_map[:, :2], seg.yaw)
        else:
            ego_lanes, ego_s = [None] * len(seg.frames), np.full(len(seg.frames), np.nan)
        m = seg.map
        polys = [np.column_stack([r.xs, r.ys]).astype(float) for r in m[m['kind'] == 'crosswalk'].itertuples()]
        for r in g.itertuples():
            k = int(r.frame_idx)
            row = {c: getattr(r, c) for c in keep if c != 'lead_map'}
            row['ego_lane'] = ego_lanes[k]
            ref = fs.loc[(sname, k), 'ego_lane'] if (sname, k) in fs.index else np.nan
            same = (ego_lanes[k] is None and pd.isna(ref)) or (ego_lanes[k] is not None and not pd.isna(ref)
                                                                and int(ref) == int(ego_lanes[k]))
            lane_bad += int(not same)
            lane0 = ego_lanes[k]
            if lane0 is None or lane0 not in lmap.lanes:
                row['dist_merge'], row['n_branches'] = np.nan, 0
            else:
                row['dist_merge'], row['n_branches'] = merge_ahead(lmap.lanes, lane0, float(ego_s[k]))
            o = seg.objects_at(k, cameras=CAMERAS, types=TYPES)
            o = o[o['id'] != r.lead_map]
            v = o[['speed_x', 'speed_y', 'speed_z']].to_numpy(float)
            if wc.SPEED_FRAME == 'global':
                v = v @ seg.R[k]                      # global -> vehicle frame (R maps vehicle to global)
            speed = np.hypot(o['speed_x'].to_numpy(float), o['speed_y'].to_numpy(float))
            x, y = o['cx'].to_numpy(float), o['cy'].to_numpy(float)
            types = o['type'].to_numpy()
            dist = np.hypot(x, y)
            P, hdg_map, _, _ = seg.to_map(k, o)
            dl, hl = nearest_lane(lmap, P)
            row['n_side_vehicle'], row['n_side_vehicle_toward'] = side_counts(types, x, y, v[:, 1])
            row['n_ped_cyc_in_crosswalk_50m'] = in_crosswalk(types, dist, P, polys)
            row.update(split_counts(types, dist, speed, dl, hdg_map, hl))
            row['heading_change_next3s_deg'] = heading_change_next(seg.yaw, seg.t, k)
            row['objects_without_lane_distance'] = int(np.isnan(dl).sum())
            row['speed_nan_objects'] = int(np.isnan(speed).sum())
            pc = np.isin(types, ['pedestrian', 'cyclist'])
            stat = (types == 'vehicle') & (speed < MOVING)
            for R in RADII:
                row[f'n_ped_cyc_{R}m'] = int((pc & (dist <= R)).sum())
                row[f'n_vehicle_stationary_{R}m'] = int((stat & (dist <= R)).sum())
                if not np.isnan(row[f'n_ped_cyc_road_{R}m']):
                    if (sname, k) in nb.index:
                        n_pc_cmp += 1
                        sum_pc_bad += int(row[f'n_ped_cyc_road_{R}m'] + row[f'n_ped_cyc_offroad_{R}m'] !=
                                          int(nb.loc[(sname, k), f'n_ped_cyc_{R}m']))
                    if (sname, k) in dc.index:
                        n_st_cmp += 1
                        sum_st_bad += int(row[f'n_vehicle_stationary_inlane_{R}m'] +
                                          row[f'n_vehicle_stationary_other_{R}m'] !=
                                          int(dc.loc[(sname, k), f'n_vehicle_stationary_{R}m']))
            if (sname, k) in second and lmap.ok and len(P):
                d2, h2 = nearest_lane_second_way(lmap, P)
                second_dev_d = max(second_dev_d, float(np.abs(d2 - dl).max()))
                dh_ = np.abs(wc.wrap(h2 - hl))
                second_dev_h = max(second_dev_h, float(dh_.max()))
                second_h_diff += int((dh_ > 1e-9).sum())
                second_n += len(P)
            rows.append(row)
        print(f'{sname}: {len(g)} frames ({time.time() - t0:.0f} s)', flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(a.out_csv, index=False)
    failed = []
    toward_bad = int((out.n_side_vehicle_toward > out.n_side_vehicle).sum())
    checks = [f'ego lane the same as frame_ssm\'s ego_lane: {len(out) - lane_bad} of {len(out)} frames',
              f'road + offroad pedestrians and cyclists = nearby_counts.csv n_ped_cyc_<R>m: {n_pc_cmp - sum_pc_bad} of '
              f'{n_pc_cmp} (frame x radius on the {n_pc_cmp // len(RADII)} frames it has)',
              f'in a lane + elsewhere stationary vehicles = D\'s n_vehicle_stationary_<R>m: {n_st_cmp - sum_st_bad} of '
              f'{n_st_cmp} (frame x radius on the {n_st_cmp // len(RADII)} frames it has)',
              f'side vehicles toward the lane more than side vehicles: {toward_bad} frames',
              f'nearest lane recomputed another way for the {second_n} objects of {N_SECOND} random frames: largest '
              f'distance difference {second_dev_d:.3g} m; direction different for {second_h_diff} objects (largest '
              f'{np.degrees(second_dev_h):.3g} deg; where the nearest point is a polyline vertex either segment\'s '
              f'direction is that lane\'s, only reported)',
              f'objects without a speed: {int(out.speed_nan_objects.sum())}; objects without a lane distance: '
              f'{int(out.objects_without_lane_distance.sum())}']
    if lane_bad:
        failed.append('ego lane')
    if sum_pc_bad or sum_st_bad or not n_pc_cmp or not n_st_cmp:
        failed.append('totals')
    if toward_bad:
        failed.append('toward')
    if not second_dev_d <= 1e-9:
        failed.append('nearest lane')
    with open(a.out_csv[:-4] + '_params.txt' if a.out_csv.endswith('.csv') else a.out_csv + '.params.txt', 'w',
              encoding='utf-8') as f:
        f.write(f'EXPLORATORY, decided after the residuals of D\'s additive model were seen (user, 2026-11-06): '
                f'new_cues_e.py run {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        for k_ in ('segments', 'targets', 'frame_ssm', 'nearby', 'd_cues'):
            f.write(f'{k_} = {os.path.abspath(getattr(a, k_))}\n')
        f.write(f'side: x {SIDE_X} m, |y| in ({SIDE_Y[0]}, {SIDE_Y[1]}] m, toward the lane: lateral ground speed >= '
                f'{SIDE_VY} m/s; merge: another lane\'s end within {MERGE_TOL} m of the branch, up to '
                f'{nc.MAX_AHEAD} m; crosswalk: within {CW_RADIUS} m; on the road: within {ROAD_TOL} m of a lane centre '
                f'line; in a lane: within {INLANE_D} m and {INLANE_DH} deg; stationary: ground speed < {MOVING} m/s; '
                f'heading change over the next {FUTURE_DT} s (within {FUTURE_TOL} s); radii {RADII}; cameras {CAMERAS}; '
                f'large vehicles (item 3) not computed\n')
        for c_ in checks:
            f.write(f'check: {c_}\n')
        f.write(f'checks not met: {failed if failed else "none"}\n')
    print(f'done in {time.time() - t0:.0f} s; {len(out)} frames; checks not met: {failed if failed else "none"}')
    for c_ in checks:
        print('  ' + c_)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
