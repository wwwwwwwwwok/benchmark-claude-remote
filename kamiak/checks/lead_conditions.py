"""Why a given vehicle is or is not the lead at a frame (2026-10-22, B18 and A08 of the acceptance check).
For each pairing method (map lane route, ego-path corridor, straight corridor) the lead conditions for that vehicle:
ahead along the route (s_rel > 0), |d| < LANE_HALF from the route centre line, heading within 30 deg, gap > 0, and the
lateral overlap with the ego (candidate: four corners; ego: four corners [2026-10-21 version] and box centre +- half
width [2026-10-22 version]); which condition fails. Also: the vehicle's speed now and its largest speed in the segment;
the lane width from the map's boundary lines where it projects onto the lane route; and the ego's later frames relative
to it (stops behind it or passes it, nearest distance and lateral clearance).

usage: python -I lead_conditions.py <extracted split root> <segment> <frame> <object id> [--follow]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
import wod_common as wc  # noqa: E402
import waymo_ssm_frames as wsf  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
CAMERAS = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']


def ego_corners_map(seg, k):
    c = np.array([[wc.EGO_X_FRONT, wc.EGO_HALF_W, 0.0], [wc.EGO_X_FRONT, -wc.EGO_HALF_W, 0.0],
                  [wc.EGO_X_REAR, -wc.EGO_HALF_W, 0.0], [wc.EGO_X_REAR, wc.EGO_HALF_W, 0.0]])
    return (c @ seg.R[k].T + seg.pose[k, :3, 3] + seg.offset[k])[:, :2]


def conditions(seg, k, xy, s_ego, row):
    """Lead conditions of one object (a one-row DataFrame of seg.objects) against a route polyline."""
    s_cum = wc.cumlen(xy)
    P, hdg, _, _ = seg.to_map(k, row)
    s, d, ph = wc.project(xy, s_cum, P)
    s_rel = s[0] - s_ego
    gap = s_rel - row['length'].iloc[0] / 2 - wc.EGO_X_FRONT
    dh = abs(np.degrees(wc.wrap(hdg[0] - ph[0])))
    ext = wc.extend_back(xy, 10.0)
    se = wc.cumlen(ext)
    _, dc, _ = wc.project(ext, se, wc.box_corners(P[:, 0], P[:, 1], row['length'], row['width'], hdg)[0])
    _, de, _ = wc.project(ext, se, ego_corners_map(seg, k))
    _, dce, _ = wc.project(ext, se, wc.ego_centre_map(seg, k))
    lo, hi = dc.min(), dc.max()
    ov_corner = min(hi, de.max()) - max(lo, de.min())
    ov_centre = min(hi, dce[0] + wc.EGO_HALF_W) - max(lo, dce[0] - wc.EGO_HALF_W)
    c = dict(s_rel=s_rel, d=d[0], dh_deg=dh, gap=gap, lead_lo=lo, lead_hi=hi, ego_corner=(de.min(), de.max()),
             ego_centre=(dce[0] - wc.EGO_HALF_W, dce[0] + wc.EGO_HALF_W), overlap_corner=ov_corner,
             overlap_centre=ov_centre)
    fails = [n for n, ok in (('ahead (s_rel > 0)', s_rel > 0), (f'|d| < {wc.LANE_HALF} m', abs(d[0]) < wc.LANE_HALF),
                             ('heading < 30 deg', dh < 30), ('gap > 0', gap > 0)) if not ok]
    c['fails_old'] = fails
    c['fails_corner'] = fails + (['lateral overlap (corners)'] if ov_corner <= 0 else [])
    c['fails_centre'] = fails + (['lateral overlap (centre +- half width)'] if ov_centre <= 0 else [])
    return c


def lane_width(seg, lmap, chain, P):
    """Width between the map boundary lines of the route lane nearest to point P (left + right distance)."""
    m = seg.map.set_index('feature_id')
    best = None
    for lane in chain:
        xy = lmap.lanes[lane]['xy']
        s, d, ph = wc.project(xy, wc.cumlen(xy), P)
        if best is None or abs(d[0]) < abs(best[1]):
            best = (lane, d[0], s[0], xy)
    lane, d, s, xy = best
    i = np.searchsorted(wc.cumlen(xy), s)
    i = min(max(i, 1), len(xy) - 1)
    t = (s - wc.cumlen(xy)[i - 1]) / max(np.hypot(*(xy[i] - xy[i - 1])), 1e-9)
    q = xy[i - 1] + t * (xy[i] - xy[i - 1])
    out = dict(lane=int(lane), lead_offset_from_lane_centre=d)
    for side in ('left_boundaries', 'right_boundaries'):
        ids = list(m.at[lane, side]) if lane in m.index else []
        dists = []
        for b in ids:
            if b in m.index and len(m.at[b, 'xs']) >= 2:
                bxy = np.column_stack([m.at[b, 'xs'], m.at[b, 'ys']])
                dists.append((float(np.min(np.hypot(*(bxy - q).T))), m.at[b, 'kind'], int(m.at[b, 'subtype'])))
        out[side] = min(dists) if dists else None
    if out['left_boundaries'] and out['right_boundaries']:
        out['width'] = out['left_boundaries'][0] + out['right_boundaries'][0]
    return out


def main():
    root, name, k, oid = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    seg = wc.Segment(root, name)
    wc.set_ego_geometry(*wc.footprint_from_lidars(seg.context))
    wsf.use_ego(*wc.footprint_from_lidars(seg.context))
    lmap = wc.LaneMap(seg.map)
    ego_lanes, ego_s = lmap.match_sequence(seg.ego_map[:, :2], seg.yaw)
    row = seg.objects[(seg.objects['frame_idx'] == k) & (seg.objects['id'] == oid)]
    if row.empty:
        sys.exit(f'{oid} is not labelled in frame {k}')
    tr = seg.objects[seg.objects['id'] == oid]
    sp = np.hypot(tr['speed_x'], tr['speed_y'])
    seen = seg.objects_at(k, cameras=CAMERAS)
    print(f'{name} frame {k}, vehicle {oid} ({row["type"].iloc[0]}, {row["length"].iloc[0]:.2f} x '
          f'{row["width"].iloc[0]:.2f} m): speed now {np.hypot(row["speed_x"].iloc[0], row["speed_y"].iloc[0]):.3f} m/s, '
          f'largest in the segment {sp.max():.3f} m/s (frame {int(tr["frame_idx"].iloc[int(np.argmax(sp.to_numpy()))])}); '
          f'labelled in frames {tr["frame_idx"].min()}-{tr["frame_idx"].max()}; seen by a camera now: '
          f'{oid in set(seen["id"])}; ego {np.hypot(*seg.ego_v[k, :2]):.3f} m/s')
    # map lane route (falls back to the ego path when the ego is not on a lane)
    if ego_lanes[k] is not None:
        chain, xy = lmap.route(ego_lanes[k], ego_s[k], [l for l in ego_lanes[k:] if l is not None])
        routes = [('map (lane route)', xy, ego_s[k], chain)]
    else:
        routes = [('map (falls back to the ego path)', wc.ego_future_path(seg, k), 0.0, None)]
    routes.append(('path (ego path corridor)', wc.ego_future_path(seg, k), 0.0, None))
    for lab, xy, s0, chain in routes:
        c = conditions(seg, k, xy, s0, row)
        print(f'\n{lab}: s_rel {c["s_rel"]:.3f} m, d {c["d"]:.3f} m, heading diff {c["dh_deg"]:.2f} deg, gap '
              f'{c["gap"]:.3f} m; across the route: vehicle [{c["lead_lo"]:.3f}, {c["lead_hi"]:.3f}] m, ego corners '
              f'[{c["ego_corner"][0]:.3f}, {c["ego_corner"][1]:.3f}] m (overlap {c["overlap_corner"]:.3f}), ego centre +- '
              f'half width [{c["ego_centre"][0]:.3f}, {c["ego_centre"][1]:.3f}] m (overlap {c["overlap_centre"]:.3f})')
        print(f'  fails: old rule {c["fails_old"] or "none"}; corner rule {c["fails_corner"] or "none"}; centre rule '
              f'{c["fails_centre"] or "none"}')
        if chain is not None:
            P, _, _, _ = seg.to_map(k, row)
            print(f'  lane width from the map boundary lines: {lane_width(seg, lmap, chain, P)}')
    # straight corridor in the vehicle frame
    r = row.iloc[0]
    gap = r.cx - r.length / 2 - wc.EGO_X_FRONT
    print(f'\nstraight (vehicle frame): cx {r.cx:.3f} m, cy {r.cy:.3f} m, heading {np.degrees(wc.wrap(r.heading)):.2f} deg,'
          f' gap {gap:.3f} m; fails: ' + str([n for n, ok in (('ahead', r.cx > 0), (f'|cy| < {wc.LANE_HALF} m',
                                                                                    abs(r.cy) < wc.LANE_HALF),
                                                               ('heading < 30 deg', abs(wc.wrap(r.heading)) < np.radians(30)),
                                                               ('gap > 0', gap > 0)) if not ok] or 'none'))
    if '--follow' in sys.argv:
        print('\nlater frames (vehicle frame of the ego at that frame): ego speed, the vehicle\'s centre cx / cy, its '
              'speed, nearest distance between the two boxes, gap ahead of the ego front, side clearance')
        rows = []
        for kk in range(k, len(seg.frames)):
            o = seg.objects[(seg.objects['frame_idx'] == kk) & (seg.objects['id'] == oid)]
            if o.empty:
                continue
            b = o.iloc[0]
            q = wsf.rect(b.cx, b.cy, b.length, b.width, b.heading)
            side = abs(b.cy) - wc.EGO_HALF_W - b.width / 2
            rows.append(dict(frame=kk, t=round(float(seg.t[kk] - seg.t[k]), 1), ego_speed=np.hypot(*seg.ego_v[kk, :2]),
                             cx=b.cx, cy=b.cy, speed=np.hypot(b.speed_x, b.speed_y), dist=wsf.poly_dist(wsf.EGO_BOX, q),
                             gap_ahead=b.cx - b.length / 2 - wc.EGO_X_FRONT, side_clearance=side,
                             front=b.cx + b.length / 2))
        f = pd.DataFrame(rows)
        print(f[f.frame % 5 == 0].drop(columns='front').round(3).to_string(index=False))
        i = f['dist'].idxmin()
        print(f'nearest: frame {f.at[i, "frame"]} (+{f.at[i, "t"]} s), {f.at[i, "dist"]:.3f} m apart, gap ahead '
              f'{f.at[i, "gap_ahead"]:.3f} m, side clearance {f.at[i, "side_clearance"]:.3f} m; ego slowest '
              f'{f["ego_speed"].min():.3f} m/s at +{f.at[f["ego_speed"].idxmin(), "t"]} s; the vehicle ends up wholly '
              f'behind the ego rear bumper: {bool((f["front"] < wc.EGO_X_REAR).any())}; last labelled frame '
              f'{f["frame"].max()} of {len(seg.frames) - 1}')


if __name__ == '__main__':
    main()
