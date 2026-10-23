"""Why the ego does not match a mapped lane in some frames (user's question, 2026-10-21). Per segment: map features by
kind (and lane types), then for every frame the nearest lane centre line (any direction, and within 45 deg of the ego
heading), whether the ego origin is inside a driveway / crosswalk polygon, the nearest road edge, speed, direction of
travel, and whether waymo_ssm_frames.py found an ego lane (frame_ssm.csv, rule: < 2.5 m and < 45 deg).

usage: python -I ego_lane_check.py <extracted root> <frame_ssm.csv> <segment> [...]
"""
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.path import Path

sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
from wod_common import LaneMap, Segment, wrap  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
LANE_TYPE = {0: 'undefined', 1: 'freeway', 2: 'surface street', 3: 'bike lane'}


def seg_dist(p, A, B):
    AB = B - A
    t = np.clip(((p - A) * AB).sum(1) / np.maximum((AB ** 2).sum(1), 1e-12), 0, 1)
    return np.hypot(*(p - (A + t[:, None] * AB)).T)


def runs(idx):
    if not len(idx):
        return []
    out, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if b != a + 1:
            out.append((start, a)); start = b
    out.append((start, idx[-1]))
    return out


def main():
    root, fcsv, segs = sys.argv[1], sys.argv[2], sys.argv[3:]
    fr_all = pd.read_csv(fcsv, usecols=['segment', 'frame_idx', 'ego_lane', 'map_fallback', 'lead_map'])
    for name in segs:
        seg = Segment(root, name)
        m = seg.map
        print(f'\n===== {name}  ({seg.context["location"]}, {seg.context["time_of_day"]}, {len(seg.frames)} frames)')
        print('map features: ' + ', '.join(f'{k} {v}' for k, v in m['kind'].value_counts().items()))
        lanes = m[m['kind'] == 'lane']
        print('lane types: ' + ', '.join(f'{LANE_TYPE.get(k, k)} {v}' for k, v in lanes['subtype'].value_counts().items()))
        lm = LaneMap(m)
        polys = {k: [Path(np.column_stack([r.xs, r.ys])) for r in m[m['kind'] == k].itertuples() if len(r.xs) >= 3]
                 for k in ('driveway', 'crosswalk', 'speed_bump')}
        edges = [np.column_stack([r.xs, r.ys]) for r in m[m['kind'] == 'road_edge'].itertuples() if len(r.xs) >= 2]
        EA = np.vstack([e[:-1] for e in edges]) if edges else None
        EB = np.vstack([e[1:] for e in edges]) if edges else None
        lane_type_of = dict(zip(lanes['feature_id'], lanes['subtype']))
        f = fr_all[fr_all['segment'] == name].set_index('frame_idx')
        rows = []
        for k in range(len(seg.frames)):
            p, h = seg.ego_map[k, :2], seg.yaw[k]
            d = seg_dist(p, lm.A, lm.B)
            j = int(d.argmin())
            dh = np.abs(wrap(lm.hdg - h))
            ok = dh < np.deg2rad(45)
            v = seg.ego_v[k, :2]
            sp = float(np.hypot(*v))
            rows.append(dict(
                k=k, matched=pd.notna(f.at[k, 'ego_lane']), speed=sp,
                travel_vs_heading=np.degrees(abs(wrap(np.arctan2(v[1], v[0]) - h))) if sp > 0.5 else np.nan,
                near_lane_m=d[j], near_lane_dh=np.degrees(dh[j]), near_lane_type=LANE_TYPE.get(lane_type_of[lm.lane_of[j]]),
                near_lane_45_m=d[ok].min() if ok.any() else np.nan,
                in_driveway=any(P.contains_point(p) for P in polys['driveway']),
                in_crosswalk=any(P.contains_point(p) for P in polys['crosswalk']),
                road_edge_m=seg_dist(p, EA, EB).min() if EA is not None else np.nan))
        r = pd.DataFrame(rows)
        path_len = float(np.hypot(*np.diff(seg.ego_map[:, :2], axis=0).T).sum())
        print(f'ego: drives {path_len:.1f} m in {seg.t[-1]:.1f} s, mean speed {r["speed"].mean():.3f} m/s, '
              f'max {r["speed"].max():.3f} m/s; frames slower than 0.5 m/s: {int((r["speed"] < 0.5).sum())}; '
              f'moving backwards (travel > 90 deg from heading): {int((r["travel_vs_heading"] > 90).sum())}')
        un = r[~r['matched']]
        print(f'frames without an ego lane in frame_ssm.csv: {len(un)} / {len(r)}; runs: {runs(un["k"].tolist())}')
        for lab, g in (('without a lane', un), ('with a lane', r[r['matched']])):
            if not len(g):
                continue
            print(f'  {lab} ({len(g)} frames): nearest lane centre {g["near_lane_m"].min():.3f}-{g["near_lane_m"].max():.3f} m '
                  f'(median {g["near_lane_m"].median():.3f}), its direction differs by {g["near_lane_dh"].median():.1f} deg '
                  f'(median; {g["near_lane_dh"].min():.1f}-{g["near_lane_dh"].max():.1f}); nearest lane within 45 deg: '
                  f'median {g["near_lane_45_m"].median():.3f} m; nearest lane types {g["near_lane_type"].value_counts().to_dict()}; '
                  f'in a driveway polygon {int(g["in_driveway"].sum())}, in a crosswalk {int(g["in_crosswalk"].sum())}; '
                  f'road edge median {g["road_edge_m"].median():.3f} m; speed median {g["speed"].median():.3f} m/s')
        for k in sorted(set([0, len(r) // 2, len(r) - 1] + [a for a, _ in runs(un['k'].tolist())])):
            x = r.iloc[k]
            print(f'    frame {k:3d}: lane {"yes" if x.matched else "no "}  speed {x.speed:6.3f}  nearest lane '
                  f'{x.near_lane_m:7.3f} m at {x.near_lane_dh:5.1f} deg ({x.near_lane_type}), within 45 deg '
                  f'{x.near_lane_45_m:7.3f} m, driveway {x.in_driveway}, road edge {x.road_edge_m:6.3f} m')


if __name__ == '__main__':
    main()
