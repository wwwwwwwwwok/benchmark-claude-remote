"""Lateral overlap of the remaining small-TTC leads under the lateral-overlap rule (2026-10-22): for every new-rule frame
with TTC < --max_ttc (not edge, not reversing, all 7 lead-vehicle SSMs with a value): the ego heading relative to the
route at the ego, the ego's and the lead's lateral intervals across the route (four corners, as in the rule), how much
they overlap, and whether they would still overlap if the ego box were aligned with the route (its centre offset +- half
width), and whether the ego-path pairing picks the same lead.

usage: python -I overlap_details.py <extracted root> <frame_ssm.csv> [<extracted root> <frame_ssm.csv>] [--max_ttc 3]
       [--out <csv>]
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser('~/benchmark-claude-remote/code'))
import wod_common as wc  # noqa: E402

sys.stdout.reconfigure(encoding='utf-8')
LEAD = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD']


def main():
    args = sys.argv[1:]
    opt = {}
    for key in ('--max_ttc', '--out'):
        if key in args:
            i = args.index(key)
            opt[key] = args[i + 1]
            args = args[:i] + args[i + 2:]
    max_ttc = float(opt.get('--max_ttc', 3))
    rows = []
    for root, fcsv in zip(args[0::2], args[1::2]):
        f = pd.read_csv(fcsv, dtype={'lead_map': str, 'lead_path': str})
        rev = f['reversing_ego'].astype(bool) | f['reversing_lead'].astype(bool)
        val = np.logical_and.reduce([(np.nan_to_num(f[s].to_numpy(float)) > 0) if s in ('DRAC', 'CI')
                                     else np.isfinite(f[s].to_numpy(float)) for s in LEAD])
        f = f[~f['edge_frame'].astype(bool) & ~rev & val & (f['TTC'] < max_ttc)]
        for name, g in f.groupby('segment'):
            seg = wc.Segment(root, name)
            wc.set_ego_geometry(*wc.footprint_from_lidars(seg.context))
            lmap = wc.LaneMap(seg.map)
            ego_lanes, ego_s = lmap.match_sequence(seg.ego_map[:, :2], seg.yaw) if lmap.ok else ([None] * len(seg.frames), None)
            for r in g.itertuples():
                k = int(r.frame_idx)
                if ego_lanes[k] is not None:
                    xy = lmap.route(ego_lanes[k], ego_s[k], [l for l in ego_lanes[k:] if l is not None])[1]
                else:
                    xy = wc.ego_future_path(seg, k)
                ext = wc.extend_back(xy, 10.0)
                s_ext = wc.cumlen(ext)
                _, de, _ = wc.project(ext, s_ext, wc.ego_corners_map(seg, k))
                _, d0, ph = wc.project(ext, s_ext, seg.ego_map[k:k + 1, :2])
                o = seg.objects[(seg.objects['frame_idx'] == k) & (seg.objects['id'] == r.lead_map)]
                P, hdg, _, _ = seg.to_map(k, o)
                _, dc, _ = wc.project(ext, s_ext, wc.box_corners(P[:, 0], P[:, 1], o['length'], o['width'], hdg)[0])
                # ego centre (between bumpers) offset across the route, for the "aligned ego" comparison
                ctr = np.array([[(wc.EGO_X_FRONT + wc.EGO_X_REAR) / 2, 0.0, 0.0]]) @ seg.R[k].T + seg.pose[k, :3, 3] + seg.offset[k]
                _, dctr, _ = wc.project(ext, s_ext, ctr[:, :2])
                al0, al1 = dctr[0] - wc.EGO_HALF_W, dctr[0] + wc.EGO_HALF_W
                rows.append(dict(segment=name, frame_idx=k, TTC=r.TTC, gap=r.gap, v_ego=r.v_ego, v_lead=r.v_lead,
                                 ego_dh_deg=float(np.degrees(abs(wc.wrap(seg.yaw[k] - ph[0])))), ego_d=float(d0[0]),
                                 ego_lo=de.min(), ego_hi=de.max(), lead_lo=dc.min(), lead_hi=dc.max(),
                                 overlap=min(de.max(), dc.max()) - max(de.min(), dc.min()),
                                 overlap_if_aligned=min(al1, dc.max()) - max(al0, dc.min()),
                                 same_as_path=r.lead_map == r.lead_path, map_fallback=bool(r.map_fallback)))
    d = pd.DataFrame(rows)
    if '--out' in opt:
        d.to_csv(opt['--out'], index=False)
    q = lambda c: ' / '.join(f'{v:.3f}' for v in d[c].quantile([0, .1, .5, .9, 1]))
    print(f'TTC < {max_ttc:g} s, not edge, not reversing, 7 lead SSMs with values: {len(d)} frames, '
          f'{d.segment.nunique()} segments')
    for c in ('ego_dh_deg', 'ego_d', 'overlap', 'overlap_if_aligned', 'v_lead', 'v_ego'):
        print(f'  {c:20s} min / p10 / median / p90 / max: {q(c)}')
    for t in (5, 10, 20):
        print(f'  ego heading more than {t} deg off the route: {int((d.ego_dh_deg > t).sum())}')
    print(f'  overlap < 0.1 m: {int((d.overlap < 0.1).sum())}; < 0.3 m: {int((d.overlap < 0.3).sum())}; '
          f'no overlap if the ego were aligned with the route: {int((d.overlap_if_aligned <= 0).sum())}; '
          f'same lead as the ego-path pairing: {int(d.same_as_path.sum())}; map fallback (no ego lane): '
          f'{int(d.map_fallback.sum())}')


if __name__ == '__main__':
    main()
