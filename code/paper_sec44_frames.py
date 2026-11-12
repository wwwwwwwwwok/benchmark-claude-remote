"""Paper section 4.4, two more items (user, 2026-11-12): Qwen3.6-27B with thinking on, sep_camera, quantities task, the
main result's 1,084 frames. Reads existing outputs only (no pictures); writes into <out_dir>.

1 sec44_frames.csv, one row per frame: segment, frame_idx and per quantity the truth, the frame estimate (median of the
  readable repeats) and the number of readable repeats, copied from the formal analysis's quantity_frames.csv
  (analysis/quantities_after_27b_think/main_1084). Units in the column names: m, m/s (mps), m/s2 (mps2).
  Checks: the truth equals quantities/truth_main.csv; the estimate and the count recomputed from the run's own replies
  (vlm_scores/qwen3.6-27b/main_quantities_qwen3.6-27b.jsonl.gz, read as analyze_quantities.read_quantities reads it:
  the number in `parsed`, a reply without one missing) equal the table.
2 the share of frames whose estimate is within 25 % of the truth: the analysis computed it for the distance only
  (exploratory_quantity_errors.csv share_within_25pct; analyze_quantities.py: ratio = larger / smaller of estimate and
  truth below 1.25, an estimate of 0 or below never within; all frames in the denominator). Copied with file and line,
  and recomputed here on the same resamples (must be the same text). EXPLORATORY: the same rule, unchanged, on the ego
  and lead speeds, which the analysis did not compute; with the frames whose truth is 0 or small counted (at the
  programs' own speed limits, 0.1 m/s = lead_ssms thw_v_min and 0.5 m/s = wod_common.speed_and_accel v_min), and the
  frames with an estimate of 0 or below.
usage: python -I paper_sec44_frames.py <waymo_v143 folder> <out_dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import csv  # noqa: E402
import hashlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_quantities import read_quantities  # noqa: E402

MODEL, COND = 'qwen3.6-27b', 'sep_camera'
N_BOOT, SEED = 2000, 20261021
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'
WITHIN = 1.25
QUANTS = [('distance', 'm'), ('ego_speed', 'mps'), ('lead_speed', 'mps'), ('ego_accel', 'mps2'), ('lead_accel', 'mps2')]
L, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
Q = os.path.join(L, 'analysis', 'quantities_after_27b_think', 'main_1084')
RUN = os.path.join(L, 'vlm_scores', MODEL, f'main_quantities_{MODEL}.jsonl.gz')
TRUTH = os.path.join(L, 'quantities', 'truth_main.csv')
TRUTH_COL = {'distance': 'distance', 'ego_speed': 'v_ego', 'lead_speed': 'v_lead', 'ego_accel': 'a_ego',
             'lead_accel': 'a_lead'}
failed = []


def check(ok, what):
    print(f'  check {"ok" if ok else "NOT MET"}: {what}')
    if not ok:
        failed.append(what)


def txt(x):
    return '' if x is None or (isinstance(x, float) and np.isnan(x)) else repr(float(x))


def ci(arr):
    ok = ~np.isnan(arr)
    if ok.sum() < 0.9 * len(arr):
        return np.nan, np.nan, int(ok.sum())
    lo, hi = np.percentile(arr[ok], [2.5, 97.5])
    return float(lo), float(hi), int(ok.sum())


print(f'paper section 4.4, per-frame table and the 25 % share (user, 2026-11-12), {MODEL} x {COND}; run '
      f'{time.strftime("%Y-%m-%d %H:%M:%S")}')
# ============================================================================================================ 1
print('\n1. sec44_frames.csv')
qf = pd.read_csv(os.path.join(Q, 'quantity_frames.csv'), float_precision='round_trip')
qf = qf[(qf.model == MODEL) & (qf.condition == COND)]
frames = qf[qf.quantity == 'distance'][['segment', 'frame_idx']].sort_values(['segment', 'frame_idx']) \
    .reset_index(drop=True)
n = len(frames)
mi = pd.MultiIndex.from_frame(frames)
tab = frames.copy()
est, tru = {}, {}
for q, u in QUANTS:
    s = qf[qf.quantity == q].set_index(['segment', 'frame_idx']).reindex(mi)
    check(len(qf[qf.quantity == q]) == n and not s['estimate'].isna().any(),
          f'quantity_frames.csv {q}: {n} frames, each with an estimate')
    tab[f'{q}_truth_{u}'] = s['truth'].to_numpy(float)
    tab[f'{q}_estimate_{u}'] = s['estimate'].to_numpy(float)
    tab[f'{q}_n_read'] = s['n_read'].to_numpy(int)
    est[q], tru[q] = s['estimate'].to_numpy(float), s['truth'].to_numpy(float)
    check(set(s['n_requests'].astype(int)) == {5}, f'{q}: 5 requests on every frame')
truth = frames.merge(pd.read_csv(TRUTH, float_precision='round_trip'), on=['segment', 'frame_idx'], how='left',
                     validate='one_to_one')
check(all(np.array_equal(truth[TRUTH_COL[q]].to_numpy(float), tru[q]) for q, _ in QUANTS),
      f'the truth of every quantity equals {os.path.basename(TRUTH)} (distance, v_ego, v_lead, a_ego, a_lead)')
raw = read_quantities(RUN, MODEL)
raw = raw[raw.condition == COND]
g = raw.groupby(['quantity', 'segment', 'frame_idx'])['value']
med = g.median()
cnt = g.count()
ok_all = True
for q, u in QUANTS:
    m_ = med.loc[q].reindex(mi).to_numpy(float)
    c_ = cnt.loc[q].reindex(mi).to_numpy(int)
    same = np.array_equal(m_, est[q]) and np.array_equal(c_, tab[f'{q}_n_read'].to_numpy(int))
    ok_all &= same
    print(f'    {q:10s}: readable repeats per frame {dict(sorted(pd.Series(c_).value_counts().items()))}; recomputed '
          f'median and count equal the table: {same}')
check(ok_all, f'the estimates and counts recomputed from {os.path.basename(RUN)} ({len(raw)} replies of {COND}) equal '
      'quantity_frames.csv')
tab.to_csv(os.path.join(OUT, 'sec44_frames.csv'), index=False)
print(f'  written: sec44_frames.csv ({len(tab)} rows, columns {list(tab.columns)})')

# ============================================================================================================ 2
print('\n2. estimate within 25 % of the truth (larger / smaller < 1.25; an estimate of 0 or below never within)')
p_exp = os.path.join(Q, 'exploratory_quantity_errors.csv')
with open(p_exp, newline='', encoding='utf-8') as f:
    rd = csv.DictReader(f)
    rows = [(rd.line_num, d) for d in rd if d['model'] == MODEL and d['condition'] == COND and d['grouping'] == 'all']
for ln, d in rows:
    print(f"    {os.path.basename(p_exp)} line {ln}: quantity={d['quantity']}, share_within_25pct="
          f"{d['share_within_25pct']}, share_within_25pct_lo={d['share_within_25pct_lo']}, share_within_25pct_hi="
          f"{d['share_within_25pct_hi']}, share_within_25pct_nboot={d['share_within_25pct_nboot']}")
dist = {d['quantity']: (ln, d) for ln, d in rows}
seg_names = frames['segment'].to_numpy()
seg_list = np.array(sorted(set(seg_names)))
seg_frames = {s: np.flatnonzero(seg_names == s) for s in seg_list}
rng = np.random.default_rng(SEED)
draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
h = hashlib.sha256()
for d in draws:
    h.update(np.searchsorted(seg_list, d).astype(np.int32).tobytes())
check(h.hexdigest() == HASH, f"drawn segments' sha256 {h.hexdigest()}")
idxs = [np.concatenate([seg_frames[s] for s in d]) for d in draws]


def within(xs, ys):
    """analyze_quantities.py's rule, as written there for the distance."""
    pos = xs > 0
    ratio = np.full(len(xs), np.inf)
    ratio[pos] = np.maximum(xs[pos], ys[pos]) / np.minimum(xs[pos], ys[pos])
    return ratio < WITHIN


out = []
for q in ('distance', 'ego_speed', 'lead_speed'):
    w = within(est[q], tru[q])
    share = float(w.mean())
    lo, hi, nv = ci(np.array([within(est[q][i], tru[q][i]).mean() for i in idxs]))
    t, e = tru[q], est[q]
    r = dict(quantity=q, frames=n, within=int(w.sum()), share=share, share_lo=lo, share_hi=hi, nboot=nv,
             truth_min=float(t.min()), truth_zero=int((t == 0).sum()), truth_le_0_1=int((t <= 0.1).sum()),
             truth_lt_0_5=int((t < 0.5).sum()), estimate_le_0=int((e <= 0).sum()),
             truth_zero_and_estimate_zero=int(((t == 0) & (e == 0)).sum()),
             within_among_truth_le_0_1=int((w & (t <= 0.1)).sum()), within_among_truth_lt_0_5=int((w & (t < 0.5)).sum()))
    out.append(r)
    if q == 'distance':
        ln, d = dist[q]
        same = (txt(share), txt(lo), txt(hi)) == (d['share_within_25pct'], d['share_within_25pct_lo'],
                                                  d['share_within_25pct_hi']) and float(d['share_within_25pct_nboot']) == nv
        check(same, f'distance: recomputed {share!r} [{lo!r}, {hi!r}] (nboot {nv}) is the same text as line {ln}')
    tag = '' if q == 'distance' else '  EXPLORATORY (the analysis did not compute it; the same rule, unchanged)'
    print(f"  {q:10s}: {r['within']} of {n} frames within, share {share!r} [{lo!r}, {hi!r}] (nboot {nv}){tag}")
    print(f"  {'':10s}  truth: smallest {r['truth_min']!r}; exactly 0 on {r['truth_zero']} frames, <= 0.1 on "
          f"{r['truth_le_0_1']}, < 0.5 on {r['truth_lt_0_5']}; estimate 0 or below on {r['estimate_le_0']} (never within); "
          f"truth 0 and estimate 0 on {r['truth_zero_and_estimate_zero']}; frames within among truth <= 0.1: "
          f"{r['within_among_truth_le_0_1']}, among truth < 0.5: {r['within_among_truth_lt_0_5']}")
pd.DataFrame(out).to_csv(os.path.join(OUT, 'exploratory_sec44_within25.csv'), index=False)
print(f'  written: exploratory_sec44_within25.csv in {OUT}')
print(f'\nchecks not met: {len(failed)}')
for w_ in failed:
    print('  ', w_)
sys.exit(1 if failed else 0)
