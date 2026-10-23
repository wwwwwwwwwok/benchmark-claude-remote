"""Frames whose lead changed between two frame_ssm.csv runs (old rule vs the lateral-overlap rule, 2026-10-21), and why.
For every changed frame and method (map, path): the old and new lead, whether the old lead fails the new lateral-overlap
test (its lateral interval across the route vs the ego's, and the distance between the two intervals), the old lead's
centre offset d from the route, its speed, and the old / new TTC. Also checks that frames whose map lead did not change
keep exactly the same values in every other column, and counts the reversing flags.

usage: python -I lead_change_reasons.py <extracted root> <old frame_ssm.csv> <new frame_ssm.csv> [--out <csv>] [--quiet]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
import wod_common as wc  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
LEAD_COLS = ['gap', 'v_ego', 'v_lead', 'a_ego', 'a_lead', 'closing', 'TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD']


def intervals(seg, k, xy, o_rows):
    """Lateral intervals across the route of the ego and of the given objects (rows of seg.objects at frame k)."""
    ext = wc.extend_back(xy, 10.0)
    s_ext = wc.cumlen(ext)
    _, de, _ = wc.project(ext, s_ext, wc.ego_corners_map(seg, k))
    P, hdg, v, _ = seg.to_map(k, o_rows)
    _, dc, _ = wc.project(ext, s_ext, wc.box_corners(P[:, 0], P[:, 1], o_rows['length'], o_rows['width'],
                                                     hdg).reshape(-1, 2))
    dc = dc.reshape(-1, 4)
    _, dcen, _ = wc.project(xy, wc.cumlen(xy), P)
    return (de.min(), de.max()), list(zip(dc.min(1), dc.max(1))), dcen, np.hypot(v[:, 0], v[:, 1])


def main():
    args = [x for x in sys.argv[1:] if x != '--quiet']
    out = None
    if '--out' in args:
        i = args.index('--out')
        out = args[i + 1]
        args = args[:i] + args[i + 2:]
    root, old_csv, new_csv = args
    old = pd.read_csv(old_csv, dtype={'lead_map': str, 'lead_path': str, 'lead_straight': str})
    new = pd.read_csv(new_csv, dtype={'lead_map': str, 'lead_path': str, 'lead_straight': str})
    old = old[old['segment'].isin(set(new['segment']))].reset_index(drop=True)
    if not old[['segment', 'frame_idx']].equals(new[['segment', 'frame_idx']]):
        sys.exit('the two files do not have the same frames')
    print(f'{new["segment"].nunique()} segments, {len(new)} frames')
    for m in ('map', 'path', 'straight'):
        a, b = old[f'lead_{m}'], new[f'lead_{m}']
        removed = a.notna() & b.isna()
        switched = a.notna() & b.notna() & (a != b)
        added = a.isna() & b.notna()
        print(f'  lead_{m}: unchanged {int(((a == b) | (a.isna() & b.isna())).sum())}, removed {int(removed.sum())}, '
              f'switched to another vehicle {int(switched.sum())}, added {int(added.sum())}')
    same = (old['lead_map'] == new['lead_map']) | (old['lead_map'].isna() & new['lead_map'].isna())
    cols = [c for c in old.columns if c not in ('lead_path', 'gap_path', 'TTC_path', 'edge_lead', 'edge_frame')]
    bad = [c for c in cols if not old.loc[same, c].astype(str).equals(new.loc[same, c].astype(str))]
    print(f'  frames with an unchanged map lead: {int(same.sum())}; columns that differ there (path columns and edge '
          f'flags not compared): {bad if bad else "none"}')
    for c in ('edge_lead', 'edge_frame'):
        print(f'  {c}: {int(old[c].sum())} -> {int(new[c].sum())}')
    for c in ('reversing_ego', 'reversing_lead'):
        print(f'  {c}: {int(new[c].sum())} frames' + (f' ({", ".join(f"{s[:6]} {k}" for s, k in new.loc[new[c], ["segment", "frame_idx"]].values)})' if new[c].any() and new[c].sum() <= 20 else ''))

    rows = []
    for name, g in new.groupby('segment'):
        chg = g.index[(old.loc[g.index, 'lead_map'].fillna('') != g['lead_map'].fillna('')) |
                      (old.loc[g.index, 'lead_path'].fillna('') != g['lead_path'].fillna(''))]
        if not len(chg):
            continue
        seg = wc.Segment(root, name)
        wc.set_ego_geometry(*wc.footprint_from_lidars(seg.context))
        lmap = wc.LaneMap(seg.map)
        ego_lanes, ego_s = lmap.match_sequence(seg.ego_map[:, :2], seg.yaw) if lmap.ok else ([None] * len(g), None)
        for i in chg:
            k = int(new.at[i, 'frame_idx'])
            for m in ('map', 'path'):
                lo_, ln_ = old.at[i, f'lead_{m}'], new.at[i, f'lead_{m}']
                if (lo_ if isinstance(lo_, str) else '') == (ln_ if isinstance(ln_, str) else ''):
                    continue
                if m == 'map' and ego_lanes[k] is not None:
                    future = [l for l in ego_lanes[k:] if l is not None]
                    _, xy = lmap.route(ego_lanes[k], ego_s[k], future)
                else:
                    xy = wc.ego_future_path(seg, k)
                ids = [x for x in (lo_, ln_) if isinstance(x, str)]
                o = seg.objects[(seg.objects['frame_idx'] == k) & seg.objects['id'].isin(ids)].set_index('id')
                (e0, e1), ivs, dcen, sp = intervals(seg, k, xy, o.reset_index())
                info = {oid: (iv, dc, s) for oid, iv, dc, s in zip(o.index, ivs, dcen, sp)}
                r = dict(segment=name, frame_idx=k, method=m, old_lead=lo_, new_lead=ln_,
                         change='removed' if not isinstance(ln_, str) else 'switched',
                         old_TTC=old.at[i, 'TTC'] if m == 'map' else old.at[i, 'TTC_path'],
                         new_TTC=new.at[i, 'TTC'] if m == 'map' else new.at[i, 'TTC_path'],
                         route='lane' if (m == 'map' and ego_lanes[k] is not None) else 'ego path',
                         ego_lo=e0, ego_hi=e1)
                if isinstance(lo_, str):
                    (a0, a1), dc, s = info[lo_]
                    r.update(old_lo=a0, old_hi=a1, old_centre_d=dc, old_speed=s,
                             old_overlaps=bool(a0 < e1 and e0 < a1), old_sep=max(a0 - e1, e0 - a1))
                if isinstance(ln_, str):
                    (b0, b1), dc, s = info[ln_]
                    r.update(new_lo=b0, new_hi=b1, new_centre_d=dc, new_speed=s, new_overlaps=bool(b0 < e1 and e0 < b1))
                rows.append(r)
    d = pd.DataFrame(rows)
    if out:
        d.to_csv(out, index=False)
        print(f'changed frames written to {out}')
    if len(d) and '--quiet' not in sys.argv:
        pd.set_option('display.width', 250)
        pd.set_option('display.max_rows', 500)
        show = d.assign(seg=d['segment'].str[:6], old_lead=d['old_lead'].str[:6], new_lead=d['new_lead'].fillna('-').str[:6])
        print(show[['seg', 'frame_idx', 'method', 'route', 'change', 'old_lead', 'new_lead', 'old_TTC', 'new_TTC', 'ego_lo',
                    'ego_hi', 'old_lo', 'old_hi', 'old_sep', 'old_centre_d', 'old_speed', 'old_overlaps']
                   + (['new_overlaps'] if 'new_overlaps' in d else [])].to_string(index=False, float_format=lambda x: f'{x:.3f}'))


if __name__ == '__main__':
    main()
