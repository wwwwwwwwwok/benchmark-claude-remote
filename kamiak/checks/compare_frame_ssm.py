"""Compare two frame_ssm.csv files of the same frames (two machines, two Kamiak nodes, or two runs), with the user's
standard (2026-10-21, floor added 2026-10-22; the analysis only uses rankings):
  1. columns that are not measurements (ids, lane, flags, counts, timestamps) are exactly the same;
  2. measurement columns agree: relative difference |x - y| / max(|x|, |y|) <= 1e-6, or absolute difference < 1e-9;
     NaN and inf in the same places;
  3. in every measurement column every pair of frames has the same order, except pairs whose values differ by less than
     1e-9 in either file (not compared). Counted over all compared frames together (the analysis pools them).
Rows can be left out (--exclude: a csv with segment, frame_idx, e.g. frames whose lead changed on purpose).
Order check without comparing all pairs: one pass in A order with Fenwick trees over the ranks of B counts the discordant
pairs, minus those closer than 1e-9 in A or in B (flipped_pairs).

usage: python -I compare_frame_ssm.py <A.csv> <B.csv> [--exclude <csv>] [--segments-of-A]
"""
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')
EXACT = ['segment', 'frame_idx', 'timestamp_micros', 'ego_lane', 'n_objects', 'lead_straight', 'lead_path', 'lead_map',
         'edge_ego', 'edge_lead', 'edge_frame', 'map_fallback', 'reversing_ego', 'reversing_lead', 'TTC2D_obj',
         'ACT_obj', 'min_dist_obj']
REL, ABS = 1e-6, 1e-9


class Fenwick:
    def __init__(self, n):
        self.t = np.zeros(n + 1, np.int64)
        self.n = n

    def add(self, i, v):
        while i <= self.n:
            self.t[i] += v
            i += i & -i

    def prefix(self, i):                          # count of ranks 1..i
        s = 0
        while i > 0:
            s += self.t[i]
            i -= i & -i
        return s


def flipped_pairs(a, b):
    """Pairs ordered one way in a and the other way in b (a_i < a_j and b_i > b_j), leaving out pairs whose values
    differ by less than ABS in a or in b; a sorted ascending, equal a values are ties (never counted). One pass in a
    order with two Fenwick trees over the ranks of b: all earlier elements, and those within ABS below in a.
    For element j: discordant = earlier with b > b_j; near in b = earlier with b_j < b < b_j + ABS; near in a = window
    elements with b > b_j; near in both = window elements with b_j < b < b_j + ABS."""
    ub = np.unique(b)
    r = np.searchsorted(ub, b) + 1                 # rank of b_j
    rh = np.maximum(np.searchsorted(ub, b + ABS, side='left'), r)  # ranks of values < b_j + ABS (b_j itself at least;
    #                                                                  for b_j = inf, b_j + ABS is inf too)
    allt, win = Fenwick(len(ub)), Fenwick(len(ub))
    n_all = n_win = 0
    lo = flips = 0
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[j + 1] == a[i]:
            j += 1
        while lo < i and not (a[i] - a[lo] < ABS):  # window = earlier elements with a > a_i - ABS
            win.add(int(r[lo]), -1)
            n_win -= 1
            lo += 1
        for t in range(i, j + 1):
            pa, pw = allt.prefix(int(r[t])), win.prefix(int(r[t]))
            inv = n_all - pa
            near_b = allt.prefix(int(rh[t])) - pa
            near_a = n_win - pw
            near_ab = win.prefix(int(rh[t])) - pw
            flips += inv - (near_a + near_b - near_ab)
        for t in range(i, j + 1):
            allt.add(int(r[t]), 1)
            win.add(int(r[t]), 1)
        n_all += j - i + 1
        n_win += j - i + 1
        i = j + 1
    return flips


def main():
    args = sys.argv[1:]
    excl = None
    if '--exclude' in args:
        i = args.index('--exclude')
        excl = pd.read_csv(args[i + 1])
        args = args[:i] + args[i + 2:]
    only_a = '--segments-of-A' in args
    args = [x for x in args if x != '--segments-of-A']
    A, B = pd.read_csv(args[0]), pd.read_csv(args[1])
    if only_a:
        B = B[B['segment'].isin(set(A['segment']))]
    key = ['segment', 'frame_idx']
    m = A.merge(B[key], on=key, how='inner')
    if len(m) != len(A) or len(m) != len(B):
        sys.exit(f'frames differ: A {len(A)}, B {len(B)}, common {len(m)}')
    A, B = A.sort_values(key).reset_index(drop=True), B.sort_values(key).reset_index(drop=True)
    if excl is not None:
        drop = pd.MultiIndex.from_frame(A[key]).isin(pd.MultiIndex.from_frame(excl[key]))
        A, B = A[~drop].reset_index(drop=True), B[~drop].reset_index(drop=True)
        print(f'left out {int(drop.sum())} rows (--exclude)')
    print(f'{len(A)} frames, {A["segment"].nunique()} segments')
    exact = [c for c in EXACT if c in A.columns]
    bad1 = []
    for c in exact:
        x, y = A[c], B[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            same = (x.to_numpy(float) == y.to_numpy(float)) | (x.isna() & y.isna()).to_numpy()
        else:
            same = (x.astype(object).where(x.notna(), '').astype(str) == y.astype(object).where(y.notna(), '').astype(str))
        if not np.all(same):
            bad1.append(f'{c} ({int((~np.asarray(same)).sum())} rows)')
    print('1. non-measurement columns: ' + ('all the same' if not bad1 else 'DIFFER: ' + ', '.join(bad1)))
    meas = [c for c in A.columns if c not in exact]
    print(f'2./3. measurement columns: {"column":10s} {"rows differ":>11s} {"over tol":>8s} {"max rel":>9s} '
          f'{"max abs":>9s} {"NaN/inf":>7s} {"pairs flipped":>13s}')
    bad2, bad3 = [], []
    for c in meas:
        x, y = A[c].to_numpy(float), B[c].to_numpy(float)
        pat = (np.isnan(x) != np.isnan(y)) | (np.isinf(x) != np.isinf(y)) | \
              (np.isinf(x) & np.isinf(y) & (np.sign(x) != np.sign(y)))
        fin = np.isfinite(x) & np.isfinite(y)
        d = np.abs(x[fin] - y[fin])
        sc = np.maximum(np.abs(x[fin]), np.abs(y[fin]))
        rel = np.where(d > 0, d / np.where(sc > 0, sc, 1.0), 0.0)
        over = (rel > REL) & (d >= ABS)
        ok = ~np.isnan(x) & ~np.isnan(y)
        xa, ya = x[ok], y[ok]
        o = np.argsort(xa, kind='mergesort')
        xa, ya = xa[o], ya[o]
        flips = flipped_pairs(xa, ya)
        print(f'{"":30s} {c:10s} {int((d > 0).sum()):11d} {int(over.sum()):8d} {rel.max() if len(rel) else 0:9.2e} '
              f'{d.max() if len(d) else 0:9.2e} {int(pat.sum()):7d} {flips:13d}')
        if over.any() or pat.any():
            bad2.append(c)
        if flips:
            bad3.append(c)
    ok = not (bad1 or bad2 or bad3)
    print(f'result: {"PASS" if ok else "FAIL"}  1. {"ok" if not bad1 else "differ"}; 2. values '
          f'{"ok" if not bad2 else "differ: " + str(bad2)}; 3. order {"ok" if not bad3 else "differs: " + str(bad3)}')


if __name__ == '__main__':
    main()
