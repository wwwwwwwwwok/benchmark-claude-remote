"""Compare the frame_ssm.csv rows of the same segments computed on two machines (first used 2026-10-21: the 3 demo
segments of the Kamiak validation run against the office PC's results/waymo_demo/frame_ssm.csv).

Standard (user, 2026-10-21; the analysis only uses rankings):
  1. columns that are not measurements (ids of the lead / closest objects, lane, edge flags, counts, timestamps) are
     exactly the same;
  2. measurement columns: relative difference |x - y| / max(|x|, |y|) <= 1e-6, with the same NaN / inf pattern;
  3. in every measurement column, every pair of frames has the same order and the same ties on both machines.
Absolute differences are printed too, for information.

usage: python -I compare_demo_office.py <frame_ssm.csv of machine A> <frame_ssm.csv of machine B>
       (B may hold more segments; the segments of A are compared)
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
EXACT = ['segment', 'frame_idx', 'timestamp_micros', 'ego_lane', 'n_objects', 'lead_straight', 'lead_path', 'lead_map',
         'edge_ego', 'edge_lead', 'edge_frame', 'map_fallback', 'TTC2D_obj', 'ACT_obj', 'min_dist_obj']
REL_TOL = 1e-6


def text_rows(path, segs):
    """Data lines of the given segments, keyed by (segment, frame_idx); line ends dropped (a file written on Windows
    may have \\r\\n)."""
    out = {}
    with open(path, encoding='utf-8') as f:
        next(f)
        for line in f:
            line = line.rstrip('\r\n')
            seg, k = line.split(',', 2)[:2]
            if seg in segs:
                out[(seg, int(k))] = line
    return out


def main():
    path_a, path_b = sys.argv[1], sys.argv[2]
    a_df = pd.read_csv(path_a)
    segs = list(dict.fromkeys(a_df['segment']))
    b_all = pd.read_csv(path_b)
    b_df = b_all[b_all['segment'].isin(segs)].reset_index(drop=True)
    print(f'A: {len(a_df)} rows, segments {segs}')
    print(f'B: {len(b_df)} rows of these segments (file has {len(b_all)} rows, {b_all["segment"].nunique()} segments)')
    if list(a_df.columns) != list(b_df.columns):
        sys.exit(f'columns differ:\n A {list(a_df.columns)}\n B {list(b_df.columns)}')
    unknown = [c for c in EXACT if c not in a_df.columns]
    if unknown:
        sys.exit(f'columns not in the files: {unknown}')
    key = ['segment', 'frame_idx']
    if not a_df[key].equals(b_df[key]):
        x, y = set(map(tuple, a_df[key].values)), set(map(tuple, b_df[key].values))
        sys.exit(f'rows differ: {len(x - y)} only in A, {len(y - x)} only in B (or a different order)')

    t_a, t_b = text_rows(path_a, set(segs)), text_rows(path_b, set(segs))
    print(f'rows identical as text: {sum(t_a[k] == t_b.get(k) for k in t_a)} / {len(t_a)}')

    print('\n1. non-measurement columns (must be exactly the same):')
    bad_exact = []
    for c in EXACT:
        a, b = a_df[c], b_df[c]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x, y = a.to_numpy(float), b.to_numpy(float)
            n_diff = int((~((x == y) | (np.isnan(x) & np.isnan(y)))).sum())
        else:
            x = a.astype(object).where(a.notna(), '').astype(str)
            y = b.astype(object).where(b.notna(), '').astype(str)
            n_diff = int((x != y).sum())
        print(f'  {c:16s} {"same" if n_diff == 0 else f"DIFFERENT in {n_diff} rows"}')
        if n_diff:
            bad_exact.append(c)

    meas = [c for c in a_df.columns if c not in EXACT]
    print(f'\n2. measurement columns (relative difference <= {REL_TOL:g}; NaN and inf must match):')
    print(f'  {"column":16s} {"rows differ":>11s} {"rel > 1e-6":>10s} {"max rel":>10s} {"max |diff|":>11s}  NaN/inf mismatch')
    bad_num = []
    for c in meas:
        x, y = a_df[c].to_numpy(float), b_df[c].to_numpy(float)
        pattern = (np.isnan(x) != np.isnan(y)) | (np.isinf(x) != np.isinf(y)) | \
                  (np.isinf(x) & np.isinf(y) & (np.sign(x) != np.sign(y)))
        fin = np.isfinite(x) & np.isfinite(y)
        d = np.abs(x[fin] - y[fin])
        scale = np.maximum(np.abs(x[fin]), np.abs(y[fin]))
        rel = np.where(d > 0, d / np.where(scale > 0, scale, 1.0), 0.0)
        n_diff, n_big = int((d > 0).sum()), int((rel > REL_TOL).sum())
        mx, mr = (float(d.max()), float(rel.max())) if len(d) else (0.0, 0.0)
        print(f'  {c:16s} {n_diff:11d} {n_big:10d} {mr:10.3e} {mx:11.3e}  {int(pattern.sum())}')
        if n_big or pattern.any():
            bad_num.append(c)
            for i in np.flatnonzero(pattern)[:5]:
                print(f'      NaN/inf differs: {a_df.at[i, "segment"][:6]} frame {a_df.at[i, "frame_idx"]}: '
                      f'A {x[i]!r}, B {y[i]!r}')
            for i in np.flatnonzero(fin)[np.argsort(-rel)[:min(5, n_big)]]:
                print(f'      {a_df.at[i, "segment"][:6]} frame {a_df.at[i, "frame_idx"]}: A {x[i]!r}, B {y[i]!r}')

    # every pair of frames: same order and same ties (inf counts as larger than every number; NaN rows, the same in
    # both files when 2. passes, are left out); over all frames of the compared segments
    print('\n3. order of frame pairs, measurement columns:')
    print(f'  {"column":16s} {"pairs":>8s} {"order differs":>13s} {"tie in one only":>15s}')
    bad_order = []
    for c in meas:
        x, y = a_df[c].to_numpy(float), b_df[c].to_numpy(float)
        m = ~np.isnan(x) & ~np.isnan(y)
        x, y = x[m], y[m]
        i, j = np.triu_indices(len(x), 1)
        with np.errstate(invalid='ignore'):
            sx, sy = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
        sx[x[i] == x[j]], sy[y[i] == y[j]] = 0, 0            # inf - inf -> nan: equal infs are a tie
        flip, tie1 = int(((sx * sy) < 0).sum()), int(((sx == 0) != (sy == 0)).sum())
        print(f'  {c:16s} {len(i):8d} {flip:13d} {tie1:15d}')
        if flip or tie1:
            bad_order.append(c)

    ok = not (bad_exact or bad_num or bad_order)
    print(f'\nresult: {"PASS" if ok else "FAIL"}  1. non-measurement columns '
          f'{"all the same" if not bad_exact else "differ: " + str(bad_exact)}; 2. measurement columns '
          f'{"all within 1e-6 relative" if not bad_num else "differ: " + str(bad_num)}; 3. order and ties '
          f'{"all the same" if not bad_order else "differ: " + str(bad_order)}')


if __name__ == '__main__':
    main()
