"""EXPLORATORY (user, 2026-11-03, after seeing the risk and quantities results; CLAUDE.md "找模型自己的规则"): how many
road users are around the ego at every target frame, a cue for code/analyze_rules.py.

Main version: in the cameras' view, occluded objects included. At every target frame k (the "now" of the VLM input)
the labelled pedestrians, cyclists and vehicles in at least one of the five cameras are counted, as
wod_common.Segment.objects_at(k, cameras=CAMERAS) keeps them: an object is kept when Waymo's projected_lidar_labels
has a box for it in one of these cameras at frame k (projected_boxes.parquet, ids "<3D label id>_<camera>": the 3D
box projected into the image, whether or not it is hidden behind something; Waymo's 2D camera labels are not used).
Sensitivity version (columns *_lidar1; user, 2026-11-03): the same, but only objects with at least 1 lidar point in
their 3D box (num_lidar_points >= 1). The frame's lead (lead_map of the target list, the lead of the SSMs) is not
counted; Waymo has no label for the ego itself. Distance = horizontal distance from the ego origin (the vehicle frame's
origin, the centre of the rear axle) to the box centre, hypot(cx, cy); counted within 10, 20, 30 and 50 m (<= R).

Checks (the run stops when one fails, before anything is written): in both versions the counts never decrease with R;
the lidar version never counts more than the main one; the lead is a labelled object of its frame; --check_frames
random frames (--seed) counted a second way (the parquet files read directly, the camera ids built from the 3D label
ids, plain Python loops) give the same counts, both versions.

Inputs: <segments> with one folder per segment (vlm_inputs/segments/: links to tables/<split>/<segment>); <targets>, the
target list (vlm_frames/main_all9_targets.csv: segment, frame_idx, lead_map, closing_below_0_1).
Output <out_csv> (nearby_counts.csv, must not exist yet): segment, frame_idx, closing_below_0_1, lead_id, lead_type,
lead_in_view, lead_distance, lead_lidar_points, and for R in 10, 20, 30, 50: n_pedestrian_<R>m, n_cyclist_<R>m,
n_vehicle_<R>m, n_ped_cyc_<R>m (pedestrians + cyclists), and the same with _lidar1. On stdout: the distribution of every
count (median, frames with 0, maximum) in both versions for the 1,084 frames of the main result and the 1,123 of the
sensitivity run, the share of the objects counted that the lidar version leaves out, and how many labelled road users
within 50 m no camera has (ahead of the rear axle / behind it).
usage: python -I count_nearby.py <segments> <targets.csv> <out_csv> [--check_frames 10] [--seed 20261103]
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wod_common as wc  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
CAMERAS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
TYPES = ['pedestrian', 'cyclist', 'vehicle']
RADII = [10, 20, 30, 50]
KINDS = TYPES + ['ped_cyc']
COUNT_COLS = [f'n_{t}_{r}m' for r in RADII for t in KINDS]
LIDAR_COLS = [c + '_lidar1' for c in COUNT_COLS]


def counts_from(types, dist, suffix=''):
    """Counts per type and radius from the types and distances of the objects counted."""
    row = {}
    for r in RADII:
        for t in TYPES:
            row[f'n_{t}_{r}m{suffix}'] = int(((types == t) & (dist <= r)).sum())
        row[f'n_ped_cyc_{r}m{suffix}'] = row[f'n_pedestrian_{r}m{suffix}'] + row[f'n_cyclist_{r}m{suffix}']
    return row


def count_frame(seg, k, lead):
    """The counts of frame k through objects_at (both versions), the lead's row among the objects in view, and the
    labelled road users within 50 m that no camera has (ahead of / behind the rear axle)."""
    o = seg.objects_at(k, cameras=CAMERAS, types=TYPES)
    lead_rows = o[o['id'] == lead]
    o = o[o['id'] != lead]
    types = o['type'].to_numpy()
    dist = np.hypot(o['cx'].to_numpy(float), o['cy'].to_numpy(float))
    pts = o['num_lidar_points'].to_numpy(float)
    lid = pts >= 1
    row = {**counts_from(types, dist), **counts_from(types[lid], dist[lid], '_lidar1')}
    every = seg.objects_at(k, types=TYPES)                  # every labelled road user, seen by a camera or not
    every = every[every['id'] != lead]
    unseen = every[~every['id'].isin(o['id'])]
    near = np.hypot(unseen['cx'].to_numpy(float), unseen['cy'].to_numpy(float)) <= 50
    ahead = unseen['cx'].to_numpy(float) >= 0
    return row, lead_rows, int((near & ahead).sum()), int((near & ~ahead).sum()), int(np.isnan(pts).sum())


def count_second_way(seg_dir, k, lead):
    """The same counts read straight from the parquet files, without objects_at: an object is in view when
    "<its id>_<camera>" is a projected box of frame k for one of CAMERAS; plain loops."""
    obj = pd.read_parquet(os.path.join(seg_dir, 'objects.parquet'),
                          columns=['frame_idx', 'id', 'type', 'cx', 'cy', 'num_lidar_points'])
    pb = pd.read_parquet(os.path.join(seg_dir, 'projected_boxes.parquet'), columns=['frame_idx', 'camera', 'id'])
    boxes = set()
    for f, c, i in zip(pb['frame_idx'], pb['camera'], pb['id']):
        if f == k and c in CAMERAS:
            boxes.add(i)
    row = {c: 0 for c in COUNT_COLS + LIDAR_COLS}
    for f, i, t, x, y, p in zip(obj['frame_idx'], obj['id'], obj['type'], obj['cx'], obj['cy'],
                                obj['num_lidar_points']):
        if f != k or t not in TYPES or i == lead:
            continue
        if not any(f'{i}_{c}' in boxes for c in CAMERAS):
            continue
        d = math.hypot(x, y)
        for r in RADII:
            if d <= r:
                for sfx in ([''] + (['_lidar1'] if p >= 1 else [])):
                    row[f'n_{t}_{r}m{sfx}'] += 1
                    if t != 'vehicle':
                        row[f'n_ped_cyc_{r}m{sfx}'] += 1
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('segments')
    ap.add_argument('targets')
    ap.add_argument('out_csv')
    ap.add_argument('--check_frames', type=int, default=10)
    ap.add_argument('--seed', type=int, default=20261103)
    a = ap.parse_args()
    if os.path.exists(a.out_csv):
        sys.exit(f'{a.out_csv} exists already')
    tg = pd.read_csv(a.targets, comment='#')[['segment', 'frame_idx', 'lead_map', 'closing_below_0_1']]
    if tg.duplicated(['segment', 'frame_idx']).any() or tg['lead_map'].isna().any():
        sys.exit(f'{a.targets}: a frame twice, or a frame without a lead')
    rows, unseen_ahead, unseen_behind, no_pts = [], 0, 0, 0
    for s, g in tg.groupby('segment', sort=True):
        seg = wc.Segment(a.segments, s)
        for r in g.sort_values('frame_idx').itertuples(index=False):
            k = int(r.frame_idx)
            row, lead_rows, ua, ub, nan_pts = count_frame(seg, k, r.lead_map)
            labelled = seg.objects[(seg.objects['frame_idx'] == k) & (seg.objects['id'] == r.lead_map)]
            if len(labelled) != 1:
                sys.exit(f'{s} frame {k}: the lead {r.lead_map} is not one labelled object of the frame '
                         f'({len(labelled)} rows)')
            lr = labelled.iloc[0]
            rows.append(dict(segment=s, frame_idx=k, closing_below_0_1=bool(r.closing_below_0_1), lead_id=r.lead_map,
                             lead_type=lr['type'], lead_in_view=len(lead_rows) == 1,
                             lead_distance=float(np.hypot(lr['cx'], lr['cy'])),
                             lead_lidar_points=float(lr['num_lidar_points']), **row))
            unseen_ahead += ua
            unseen_behind += ub
            no_pts += nan_pts
    out = pd.DataFrame(rows)

    # checks
    for t in KINDS:
        for sfx in ('', '_lidar1'):
            v = out[[f'n_{t}_{r}m{sfx}' for r in RADII]].to_numpy()
            if (np.diff(v, axis=1) < 0).any():
                sys.exit(f'{t}{sfx}: a count decreases with the radius')
        for r in RADII:
            if (out[f'n_{t}_{r}m_lidar1'] > out[f'n_{t}_{r}m']).any():
                sys.exit(f'{t} within {r} m: the lidar version counts more than the main one')
    rng = np.random.default_rng(a.seed)
    pick = sorted(rng.choice(len(out), min(a.check_frames, len(out)), replace=False))
    for i in pick:
        r = out.iloc[i]
        other = count_second_way(os.path.join(a.segments, r.segment), int(r.frame_idx), r.lead_id)
        mine = {c: int(r[c]) for c in COUNT_COLS + LIDAR_COLS}
        if other != mine:
            diff = {c: (mine[c], other[c]) for c in mine if mine[c] != other[c]}
            sys.exit(f'{r.segment} frame {r.frame_idx}: the two ways differ {diff}')
    out.to_csv(a.out_csv, index=False)

    print(f'{len(out)} frames in {out.segment.nunique()} segments; cameras {CAMERAS}; types {TYPES}; '
          f'distance = hypot(cx, cy) of the box centre from the rear axle, counted when <= R')
    print('main version: in the cameras\' view (a projected 3D box), occluded objects included; lidar version '
          '(_lidar1): only objects with at least 1 lidar point in their 3D box')
    print(f'checks: counts never decrease with the radius (both versions); the lidar version never above the main one; '
          f'every lead is one labelled object of its frame; {len(pick)} random frames (seed {a.seed}) counted a second '
          f'way: the same (both versions); objects in view without a lidar point count: {no_pts}')
    print(f'leads: {out.lead_type.value_counts().to_dict()}; in view of a camera {int(out.lead_in_view.sum())} of '
          f'{len(out)}; distance (rear axle to box centre) median {out.lead_distance.median():.2f} m; with no lidar '
          f'point {int((out.lead_lidar_points < 1).sum())}')
    print(f'labelled road users within 50 m (lead not counted) that no camera has, summed over the frames: '
          f'{unseen_ahead} ahead of the rear axle (x >= 0), {unseen_behind} behind it')
    for name, sel in (('main result (closing_below_0_1 False)', ~out.closing_below_0_1),
                      ('sensitivity run (all)', np.ones(len(out), bool))):
        o = out[sel]
        print(f'== {name}: {len(o)} frames')
        for sfx, ver in (('', 'main version (in view, occluded included)'), ('_lidar1', 'at least 1 lidar point')):
            print(f'  {ver}:')
            for r in RADII:
                parts = []
                for t in KINDS:
                    v = o[f'n_{t}_{r}m{sfx}']
                    parts.append(f'{t} median {v.median():g}, 0 in {int((v == 0).sum())} frames, max {int(v.max())}')
                print(f'    within {r} m: ' + '; '.join(parts))
        print('  left out by the lidar version (no lidar point in the box), summed over the frames:')
        for r in RADII:
            parts = []
            for t in KINDS:
                m_, l_ = int(o[f'n_{t}_{r}m'].sum()), int(o[f'n_{t}_{r}m_lidar1'].sum())
                parts.append(f'{t} {m_ - l_} of {m_} ({(m_ - l_) / m_:.1%})' if m_ else f'{t} 0 of 0')
            print(f'    within {r} m: ' + '; '.join(parts))
    print(f'written {os.path.abspath(a.out_csv)}')


if __name__ == '__main__':
    main()
