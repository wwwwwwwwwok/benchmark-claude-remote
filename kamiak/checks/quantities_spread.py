"""Spread of the repeats of the quantities task (user, 2026-10-29), per run x model x condition x quantity, on the frames
with a truth value for the quantity (and with a frames file, only those, e.g. the 1,084 of the main result):
 - dispersion: the standard deviation over a frame's readable repeats (ddof 0, as in quantities_pilot_report.py; frames
   with at least 2), its median over the frames, beside the standard deviation of the truth between the frames
   (ddof 0) and the ratio of the two;
 - the first k repeats: the median over the frames of |median of the readable answers among repeats 0 .. k-1 - median
   of all 5|, k = 1 .. 4; frames where either has no answer are left out (counted).
Each input is <label>,<jsonl[.gz] named <run>_quantities_<model>>,<truth csv of code/quantity_truth.py>[,<csv with
segment and frame_idx columns: the frames to use>].
usage: python -I quantities_spread.py <out prefix> <input> [<input> ...]
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'code'))
import analyze_quantities as aq  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402
from quantity_truth import TRUTH  # noqa: E402

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
pd.set_option('display.max_rows', 500)
prefix, inputs = sys.argv[1], sys.argv[2:]
disp, firstk = [], []
for spec in inputs:
    label, path, truth_csv, *rest = spec.split(',')
    model = re.sub(r'\.jsonl(\.gz)?$', '', os.path.basename(path)).split('_quantities_', 1)[1]
    r = aq.read_quantities(path, model)
    tr = pd.read_csv(truth_csv, float_precision='round_trip')
    if rest:
        keep = pd.read_csv(rest[0], usecols=['segment', 'frame_idx']).drop_duplicates()
        tr = tr.merge(keep, on=['segment', 'frame_idx'], validate='one_to_one')
    print(f'{label} {model}: {len(r)} lines, {len(r[["segment", "frame_idx"]].drop_duplicates())} frames asked; '
          f'{len(tr)} frames used ({path})', flush=True)
    for q in aq.QUANTS:
        t = tr[['segment', 'frame_idx', TRUTH[q]]].dropna()
        truth_sd = float(np.std(t[TRUTH[q]].to_numpy(float)))
        for c in av.CONDITIONS:
            s = r[(r.quantity == q) & (r.condition == c)].merge(t[['segment', 'frame_idx']], on=['segment', 'frame_idx'])
            g = s.dropna(subset=['value']).groupby(['segment', 'frame_idx']).value
            n = g.size()
            sd = g.std(ddof=0)[n >= 2]
            disp.append(dict(run=label, model=model, condition=c, quantity=q, frames=len(sd),
                             within_sd_median=float(sd.median()), truth_sd_between=truth_sd,
                             ratio=float(sd.median()) / truth_sd))
            m5 = s.groupby(['segment', 'frame_idx']).value.median()
            for k in (1, 2, 3, 4):
                mk = s[s.repeat < k].groupby(['segment', 'frame_idx']).value.median().reindex(m5.index)
                ok = mk.notna() & m5.notna()
                firstk.append(dict(run=label, model=model, condition=c, quantity=q, k=k, frames=int(ok.sum()),
                                   frames_missing=int((~ok).sum()),
                                   abs_diff_median=float((mk[ok] - m5[ok]).abs().median())))
disp, firstk = pd.DataFrame(disp), pd.DataFrame(firstk)
disp.to_csv(prefix + '_dispersion.csv', index=False)
firstk.to_csv(prefix + '_first_k_medians.csv', index=False)
runs = list(dict.fromkeys(zip(disp.run, disp.model)))
rows = pd.MultiIndex.from_tuples([(u, m, q) for u, m in runs for q in aq.QUANTS], names=['run', 'model', 'quantity'])
print('\n=== median over frames of the SD over the repeats (ddof 0) / SD of the truth between frames')
d = disp.assign(c=[f'{r.within_sd_median:.3f} / {r.truth_sd_between:.3f}' for r in disp.itertuples()])
print(d.pivot(index=['run', 'model', 'quantity'], columns='condition', values='c').reindex(index=rows,
                                                                                            columns=av.CONDITIONS)
      .to_string())
print('\n=== median over frames of |median of the first k - median of 5| (k = 1, 2, 3, 4)')
f = firstk.assign(c=firstk.groupby(['run', 'model', 'quantity', 'condition'], sort=False).abs_diff_median
                  .transform(lambda s: ' / '.join(f'{v:.3f}' for v in s)))
print(f[f.k == 1].pivot(index=['run', 'model', 'quantity'], columns='condition', values='c')
      .reindex(index=rows, columns=av.CONDITIONS).to_string())
print(f'frames left out (no answer among the first k): {int(firstk.frames_missing.sum())} in all, '
      f'largest in a cell {int(firstk.frames_missing.max())}')
