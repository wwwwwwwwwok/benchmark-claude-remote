"""The four items the user asked for after the 8-model analysis (user, 2026-11-13), main result (main_1084), grouping
"all", read from the analysis's own outputs; every number with its file and line (line 1 = the header), full precision.
1. agreement.csv: every model x condition x SSM, tau-b and its 95% interval (tau_b_conflict, _lo, _hi; on these frames
   every SSM has a value, so tau_b_all is the same number: checked, also for the interval), SSMs in the order TTC, THW,
   DRAC, MTTC, CI, PICUD, PSD, 2D-TTC, ACT;
2. sep_camera score distribution per model, from vlm_requests_tidy.csv (every answer the analysis uses, forced answers
   included, as the frame scores): mean, SD (ddof 0 and ddof 1), number of distinct scores (checked against
   condition_summary.csv distinct_scores), the 5 most common values with counts; from vlm_frame_scores.csv the number
   of distinct 5-answer means (score_mean);
3. sep_camera repeat stability per model: condition_summary.csv retest_tau_b_mean, icc_mean_of_k (ICC(1,k), one-way
   random effects, mean of k = 5 answers, frames with all 5 answers), sd_repeats_mean (ddof 0), with lines; the range
   of the 10 test-retest tau-b, recomputed as analyze_vlm_ssm.repeat_stability does (pairs of repeats, frames with both
   answers, at least 3), whose mean must equal retest_tau_b_mean;
4. one model's thinking (default gemma-4-31b-it), from vlm_requests_tidy.csv and condition_summary.csv: reasoning
   tokens median and max per condition and in all, replies that hit max_tokens, forced answers used, no answer.
usage: python -I report_after_gemma.py <analysis folder>/main_1084 [--model gemma-4-31b-it]
"""
import argparse
import collections
import itertools
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ap = argparse.ArgumentParser()
ap.add_argument('folder')
ap.add_argument('--model', default='gemma-4-31b-it')
a = ap.parse_args()
SSMS = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD', 'TTC2D', 'ACT']      # 2D-TTC is TTC2D in the tables
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
pd.set_option('display.width', 250)


def with_lines(name):
    """The table with its file line, and the same table as text (the numbers printed exactly as written)."""
    path = os.path.join(a.folder, name)
    t, raw = pd.read_csv(path), pd.read_csv(path, dtype=str, keep_default_na=False)
    t['line'] = np.arange(2, len(t) + 2)
    return path, t, raw


def full(x):
    return repr(float(x))


ag_path, ag, ag_raw = with_lines('agreement.csv')
models = list(dict.fromkeys(ag.model))
ag = ag[ag.grouping == 'all']
bad = ag[(ag.tau_b_all != ag.tau_b_conflict) | (ag.tau_b_all_lo != ag.tau_b_conflict_lo)
         | (ag.tau_b_all_hi != ag.tau_b_conflict_hi)]
print(f'=== 1. {ag_path}, grouping all: {len(ag)} rows, models in file order {models}; rows where tau_b_all or its '
      f'interval differs from tau_b_conflict: {len(bad)}; n_conflict values {sorted(ag.n_conflict.unique())}, '
      f'n_no_conflict {sorted(ag.n_no_conflict.unique())}, nboot {sorted(ag.tau_b_conflict_nboot.unique())}')
print('model\tcondition\tssm\tline\ttau_b\tlo\thi')
for m, c, s in itertools.product(models, CONDS, SSMS):
    r = ag[(ag.model == m) & (ag.condition == c) & (ag.ssm == s)]
    assert len(r) == 1, (m, c, s, len(r))
    t = ag_raw.loc[r.index[0]]
    print(f'{m}\t{c}\t{s}\t{r.line.iloc[0]}\t{t.tau_b_conflict}\t{t.tau_b_conflict_lo}\t{t.tau_b_conflict_hi}')

td_path = os.path.join(a.folder, 'vlm_requests_tidy.csv')
td = pd.read_csv(td_path, usecols=['model', 'segment', 'frame_idx', 'condition', 'repeat', 'finish_reason', 'forced',
                                   'risk', 'reasoning_tokens', 'completion_tokens'])
fs_path = os.path.join(a.folder, 'vlm_frame_scores.csv')
fs = pd.read_csv(fs_path, usecols=['model', 'condition', 'segment', 'frame_idx', 'n_answers', 'score_mean'])
cs_path, cs, cs_raw = with_lines('condition_summary.csv')

print(f'\n=== 2. sep_camera score distribution ({td_path}; {fs_path}; distinct_scores from {cs_path})')
for m in models:
    q = td[(td.model == m) & (td.condition == 'sep_camera')]
    s = q.risk.dropna()
    top = collections.Counter(s.tolist()).most_common()
    top5 = top[:5]
    tie = [v for v, n in top[5:] if n == top5[-1][1]]
    f = fs[(fs.model == m) & (fs.condition == 'sep_camera')]
    c = cs[(cs.model == m) & (cs.condition == 'sep_camera')].iloc[0]
    print(f'{m}: requests {len(q)}, answers {len(s)} (forced among them {int(q.forced[q.risk.notna()].sum())}); mean '
          f'{full(s.mean())}; SD ddof0 {full(s.std(ddof=0))}, ddof1 {full(s.std(ddof=1))}; distinct scores '
          f'{s.nunique()} (condition_summary line {c.line}: {int(c.distinct_scores)}); most common '
          f'{", ".join(f"{v:g} x{n}" for v, n in top5)}' + (f' (also {n} times: {tie})' if tie else '')
          + f'; frames {len(f)}, distinct 5-answer means (score_mean) {f.score_mean.nunique()}, frames with fewer '
          f'than 5 answers {int((f.n_answers < 5).sum())}')

print(f'\n=== 3. sep_camera repeat stability ({cs_path}; the range recomputed from {td_path})')
for m in models:
    q = td[(td.model == m) & (td.condition == 'sep_camera')]
    w = q.pivot_table(index=['segment', 'frame_idx'], columns='repeat', values='risk')
    taus = []
    for x, y in itertools.combinations(w.columns, 2):
        mm = w[[x, y]].dropna()
        if len(mm) >= 3:
            taus.append(kendalltau(mm[x], mm[y], variant='b').statistic)
    sel = (cs.model == m) & (cs.condition == 'sep_camera')
    c, t = cs[sel].iloc[0], cs_raw[sel].iloc[0]
    d = float(np.nanmean(taus)) - float(c.retest_tau_b_mean)      # the csv keeps 16 significant digits
    print(f'{m} (line {c.line}): retest_tau_b_mean {t.retest_tau_b_mean} (recomputed mean of {len(taus)} pairs '
          f'{full(np.nanmean(taus))}, difference {d:.3g}, below 1e-12: {abs(d) < 1e-12}); range {full(min(taus))} to '
          f'{full(max(taus))}; ICC(1,5) icc_mean_of_k {t.icc_mean_of_k} (k_repeats {t.k_repeats}, frames_all_k '
          f'{t.frames_all_k}); icc_single {t.icc_single}; sd_repeats_mean {t.sd_repeats_mean}')

print(f'\n=== 4. {a.model}: thinking ({td_path}; {cs_path})')
q = td[td.model == a.model]
if q.empty:
    print(f'  {a.model} is not in this analysis')
else:
    for c in CONDS + ['all']:
        g = q if c == 'all' else q[q.condition == c]
        rt = g.reasoning_tokens.dropna()
        line = '' if c == 'all' else f" (condition_summary line {cs[(cs.model == a.model) & (cs.condition == c)].iloc[0].line})"
        print(f'  {c}{line}: requests {len(g)}, reasoning tokens given {len(rt)}, median {full(rt.median())}, max '
              f'{full(rt.max())}; hit max_tokens {int((g.finish_reason == "length").sum())}; forced answers used '
              f'{int(g.forced.sum())}; no answer {int(g.risk.isna().sum())}')
    print('  condition_summary rows:')
    print(cs[cs.model == a.model][['line', 'condition', 'requests', 'answered', 'forced_answers', 'no_answer',
                                   'hit_max_tokens', 'reasoning_tokens_median', 'completion_tokens_median']]
          .to_string(index=False))
