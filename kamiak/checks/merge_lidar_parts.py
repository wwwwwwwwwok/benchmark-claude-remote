"""Join the per-block outputs of code/check_lidar_points.py (kamiak/waymo_lidar_check.sbatch) into one folder per split
and print the summaries that check_lidar_points.py prints for one run, per split and for all splits together.
The share of vehicle tracks with a type needs every vehicle track, read from the segment tables.

usage: python -I merge_lidar_parts.py <workspace> <RUN> <tables root> <out root> validation [training]
       writes <out root>/<split>/lidar_check/{lead_rear_points_vs_box,vehicle_type_from_lidar_seg,vehicle_type_per_track}.csv
"""
import glob
import os
import sys

import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
FILES = ('lead_rear_points_vs_box.csv', 'vehicle_type_from_lidar_seg.csv', 'vehicle_type_per_track.csv')


def summary(title, rr, pt, veh, leads):
    print(f'\n===== {title}')
    print(f'lead rear face: {len(rr)} lead frames, {rr.groupby(["segment", "lead_id"]).ngroups} leads, '
          f'{rr.segment.nunique()} segments')
    for c in ('rear_nearest_minus_box', 'rear_dense_minus_box'):
        q = rr[c].abs().quantile([.5, .95, 1.0]).to_numpy()
        s = rr[c].quantile([.05, .5, .95]).to_numpy()
        print(f'  |{c}|: median {q[0]:.3f}, 95th pct {q[1]:.3f}, max {q[2]:.3f}  (signed: 5th {s[0]:.3f}, median '
              f'{s[1]:.3f}, 95th {s[2]:.3f})')
    n_tracks = veh.drop_duplicates().shape[0]
    typed = veh.set_index(['segment', 'id']).index.isin(pt.set_index(['segment', 'id']).index)
    print(f'vehicle type: {len(pt)} of {n_tracks} vehicle tracks ({len(pt) / n_tracks:.1%}), {typed.mean():.1%} of '
          f'vehicle boxes')
    print(pt.groupby('seg_type').agg(tracks=('length', 'size'), length_med=('length', 'median'),
                                     height_med=('height', 'median'),
                                     agreement_med=('agreement', 'median')).round(2).to_string())
    lt = pt.merge(leads, on=['segment', 'id'])
    print(f'lead vehicles (distinct segment + id): {len(leads)}, with a type: {len(lt)}; types: '
          f'{lt.seg_type.value_counts().to_dict()}')


def main():
    ws, run, tables, out = sys.argv[1:5]
    splits = sys.argv[5:]
    allr, allp, allv, alll = [], [], [], []
    for split in splits:
        parts = os.path.join(ws, f'ssm_parts_{run}', split)
        blocks = sorted(glob.glob(os.path.join(parts, '*', 'lidar_check')))
        n_blocks = len(glob.glob(os.path.join(parts, '*', 'ssm', 'frame_ssm.csv')))
        if len(blocks) != n_blocks:
            sys.exit(f'{split}: {len(blocks)} blocks have lidar_check, {n_blocks} blocks exist')
        got = {f: [] for f in FILES}
        for b in blocks:
            for f in FILES:
                p = os.path.join(b, f)
                if os.path.exists(p) and os.path.getsize(p) > 1:
                    got[f].append(pd.read_csv(p))
        o = os.path.join(out, split, 'lidar_check')
        os.makedirs(o, exist_ok=True)
        tabs = {f: pd.concat(v, ignore_index=True) for f, v in got.items()}
        for f, t in tabs.items():
            t.to_csv(os.path.join(o, f), index=False)
        fs = pd.read_csv(os.path.join(out, split, 'ssm', 'frame_ssm.csv'), usecols=['segment', 'lead_map'],
                         dtype={'lead_map': str})
        leads = fs.dropna().drop_duplicates().rename(columns={'lead_map': 'id'})
        veh = []
        for s in fs.segment.unique():
            ob = pd.read_parquet(os.path.join(tables, split, s, 'objects.parquet'), columns=['id', 'type'])
            veh.append(pd.DataFrame({'segment': s, 'id': ob.loc[ob['type'] == 'vehicle', 'id']}))
        veh = pd.concat(veh, ignore_index=True)
        summary(f'{split} ({len(blocks)} blocks)', tabs[FILES[0]], tabs[FILES[2]], veh, leads)
        allr.append(tabs[FILES[0]]), allp.append(tabs[FILES[2]]), allv.append(veh), alll.append(leads)
    if len(splits) > 1:
        summary('all splits', pd.concat(allr), pd.concat(allp), pd.concat(allv), pd.concat(alll))


if __name__ == '__main__':
    main()
