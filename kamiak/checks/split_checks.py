"""Checks after a split has been merged (user's step 7, 2026-10-21): segments and frames, the 77 m label range in
params.txt, segments whose ego never matches a lane, and columns that are empty for a whole segment (they trigger the
pandas FutureWarning in waymo_ssm_frames.py). Reads only the merged results.

usage: python -I split_checks.py <results/<split> folder> [...]
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')


def main():
    for res in sys.argv[1:]:
        split = os.path.basename(os.path.normpath(res))
        f = pd.read_csv(os.path.join(res, 'ssm', 'frame_ssm.csv'), dtype=str, keep_default_na=False)
        n = f.groupby('segment').size()
        edge = f['edge_frame'] == 'True'
        k = f['frame_idx'].astype(int)
        tgt = ~edge & (k % 5 == 0) & (k >= 20)
        print(f'\n===== {split}: {len(n)} segments, {len(f)} frames (per segment {n.min()}-{n.max()}, median '
              f'{int(n.median())}); edge frames {int(edge.sum())}; not edge {int((~edge).sum())}; target frames '
              f'{int(tgt.sum())}')
        with open(os.path.join(res, 'ssm', 'params.txt'), encoding='utf-8') as fh:
            lines = fh.read().splitlines()
        lab = {ln.split()[1]: float(ln.split()[-2]) for ln in lines if ln.startswith('labels ')}
        v = np.array(list(lab.values()))
        print(f'params.txt: {len(lab)} label lines; farthest box nearest point max {v.max():.3f} m, min {v.min():.3f} m; '
              f'> 77 m: {int((v > 77).sum())}; = 77.000 m: {int((np.round(v, 3) == 77.0).sum())}; nan: '
              f'{int(np.isnan(v).sum())}')
        egos = [re.findall(r'[-\d.]+ m', ln) for ln in lines if re.match(r'ego \d', ln)]
        print('ego footprints (front, rear, half width): ' +
              ', '.join(f'{" / ".join(e)} x{c}' for e, c in pd.Series([tuple(x) for x in egos]).value_counts().items()))
        chk = pd.read_csv(os.path.join(res, 'checks', 'extract_checks.csv'), index_col=0)
        unm = chk.loc['ego_unmatched_frames'].astype(float)
        frm = chk.loc['frames'].astype(float)
        nolane = f.groupby('segment')['ego_lane'].apply(lambda s: int((s == '').sum()))
        print(f'ego never on a lane within 10 m / 45 deg (checks): {int((unm == frm).sum())} segments '
              f'{sorted(unm.index[unm == frm])}; some frames: {int(((unm > 0) & (unm < frm)).sum())} segments')
        print(f'ego_lane empty in every frame (SSM rule 2.5 m / 45 deg): {int((nolane == n).sum())} segments '
              f'{sorted(nolane.index[nolane == n])}; in some frames: {int(((nolane > 0) & (nolane < n)).sum())} segments, '
              f'{int(nolane.sum())} frames in all')
        empty = {}
        for c in f.columns:
            e = f.groupby('segment')[c].apply(lambda s: bool((s == '').all()))
            for s in e.index[e]:
                empty.setdefault(s, []).append(c)
        print(f'segments with a column empty in every frame: {len(empty)}')
        for s, cols in sorted(empty.items()):
            print(f'  {s}: {cols}')
        lanes = chk.loc['lanes'].astype(float)
        print(f'segments without any lane in the map: {int((lanes == 0).sum())} {sorted(lanes.index[lanes == 0])}')


if __name__ == '__main__':
    main()
