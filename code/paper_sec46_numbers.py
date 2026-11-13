"""Paper section 4.6 (input formats), numbers for the user (2026-11-13). Reads existing outputs only; CPU only.
Qwen3.6-27B with thinking on, the main result's 1,084 frames, grouping "all", from analysis/after_qwen3.5-4b/main_1084;
every number taken from a table is printed as written there, with its file and line (line 1 = the header).
1 agreement.csv: the 5 conditions x 9 SSMs, tau-b and its 95% interval (tau_b_conflict, _lo, _hi; every SSM has a value
  on these frames, so tau_b_all and its interval are the same numbers: checked); SSMs in the order TTC, THW, DRAC, MTTC,
  CI, PICUD, PSD, 2D-TTC (TTC2D in the tables), ACT.
2 per condition the SSM with the highest and the lowest tau-b; ssm_differences.csv: the rows of ACT against each of the
  other 8 (the table has ssm_1 - ssm_2 with ACT as ssm_2, stat tau_b_all; ACT - other = -diff, interval [-hi, -lo]),
  those whose interval leaves out 0 (the table's clear column), the difference checked against the two tau-b of 1.
3 condition_summary.csv per condition: retest_tau_b_mean, icc_mean_of_k (ICC(1,k), one-way random effects, mean of
  k = 5 answers, frames with all 5 answers), sd_repeats_mean (ddof 0); the range of the 10 test-retest tau-b recomputed
  from vlm_requests_tidy.csv as analyze_vlm_ssm.repeat_stability does (pairs of repeats, frames with both answers, at
  least 3; mean checked against retest_tau_b_mean).
4 between conditions: Kendall tau-b (scipy, variant b) of the frame scores (vlm_frame_scores.csv score_mean, the mean of
  the answers) of every pair of conditions on the 1,084 frames, 10 pairs (point estimates; not asked with intervals);
  and, as written, the Spearman correlations and disattenuated ones computed before (score_agreement.py, 2026-11-05,
  analysis/rule_explore/tree_27b/after_ceiling/agreement/main_1084/exploratory_agreement.csv) with their params.
usage: python -I paper_sec46_numbers.py <waymo_v143 folder> <out_dir>
"""
import csv
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

L, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
A = os.path.join(L, 'analysis', 'after_qwen3.5-4b', 'main_1084')
AGR = os.path.join(L, 'analysis', 'rule_explore', 'tree_27b', 'after_ceiling', 'agreement', 'main_1084')
MODEL = 'qwen3.6-27b'
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
SSMS = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD', 'TTC2D', 'ACT']
failed = []


def check(ok, what):
    print(f'  check {"ok" if ok else "NOT MET"}: {what}')
    if not ok:
        failed.append(what)


def rows_with_lines(path):
    """The csv as text rows (dicts), each with its file line."""
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        out = []
        for x in rd:
            x['_line'] = rd.line_num
            out.append(x)
    return out


def neg(s):
    """-x of a number written as text, as text (sign flipped, digits kept)."""
    return s[1:] if s.startswith('-') else '-' + s


print(f'paper section 4.6: input formats, {MODEL} (user, 2026-11-13); run {time.strftime("%Y-%m-%d %H:%M:%S")}\n'
      f'risk analysis {A}')

p_ag = os.path.join(A, 'agreement.csv')
ag = [x for x in rows_with_lines(p_ag) if x['model'] == MODEL and x['grouping'] == 'all']
same = all((x['tau_b_all'], x['tau_b_all_lo'], x['tau_b_all_hi']) ==
           (x['tau_b_conflict'], x['tau_b_conflict_lo'], x['tau_b_conflict_hi']) for x in ag)
check(len(ag) == 45 and same and {x['n_conflict'] for x in ag} == {'1084'} and
      {x['tau_b_conflict_nboot'] for x in ag} == {'2000'},
      f'{p_ag}: {len(ag)} rows of {MODEL}, grouping all; tau_b_all and its interval = tau_b_conflict and its interval in '
      f'every row; 1,084 frames; 2,000 resamples')
tau = {(x['condition'], x['ssm']): x for x in ag}
print(f'\n1. tau-b and 95% interval ({p_ag}, grouping all; columns tau_b_conflict, _lo, _hi)')
print('condition\tssm\tline\ttau_b\tlo\thi')
for c, s in itertools.product(CONDS, SSMS):
    x = tau[(c, s)]
    print(f"{c}\t{'2D-TTC' if s == 'TTC2D' else s}\t{x['_line']}\t{x['tau_b_conflict']}\t{x['tau_b_conflict_lo']}\t"
          f"{x['tau_b_conflict_hi']}")

print('\n2. highest and lowest tau-b per condition; ACT - each other SSM (ssm_differences.csv)')
for c in CONDS:
    v = {s: float(tau[(c, s)]['tau_b_conflict']) for s in SSMS}
    hi_, lo_ = max(v.values()), min(v.values())
    def show(names):
        return ', '.join(f"{'2D-TTC' if s == 'TTC2D' else s} {tau[(c, s)]['tau_b_conflict']} [{tau[(c, s)]['tau_b_conflict_lo']}, "
                         f"{tau[(c, s)]['tau_b_conflict_hi']}] (line {tau[(c, s)]['_line']})" for s in names)
    print(f'  {c}: highest {show([s for s in SSMS if v[s] == hi_])}; lowest {show([s for s in SSMS if v[s] == lo_])}')
p_sd = os.path.join(A, 'ssm_differences.csv')
sd = [x for x in rows_with_lines(p_sd) if x['model'] == MODEL and x['grouping'] == 'all'
      and 'ACT' in (x['ssm_1'], x['ssm_2'])]
check(len(sd) == 40 and all(x['ssm_2'] == 'ACT' and x['stat'] == 'tau_b_all' for x in sd),
      f'{p_sd}: {len(sd)} rows of {MODEL} with ACT, grouping all, all with ACT as ssm_2 and stat tau_b_all')
n_bad = 0
for x in sd:
    d = float(tau[(x['condition'], x['ssm_1'])]['tau_b_all']) - float(tau[(x['condition'], 'ACT')]['tau_b_all'])
    n_bad += abs(d - float(x['diff'])) > 1e-12
    clear = float(x['diff_lo']) > 0 or float(x['diff_hi']) < 0
    n_bad += clear != (x['clear'] == 'True')
check(n_bad == 0, 'every diff = tau_b_all(ssm_1) - tau_b_all(ACT) within 1e-12, and clear = interval leaves out 0')
print(f'  ACT - other = -(diff of the table); interval [-diff_hi, -diff_lo]; * = interval leaves out 0')
for c in CONDS:
    print(f'  {c}:')
    for x in sorted((x for x in sd if x['condition'] == c), key=lambda x: SSMS.index(x['ssm_1'])):
        s = '2D-TTC' if x['ssm_1'] == 'TTC2D' else x['ssm_1']
        print(f"    {'*' if x['clear'] == 'True' else ' '} ACT - {s:6s} line {x['_line']}: {neg(x['diff'])} "
              f"[{neg(x['diff_hi'])}, {neg(x['diff_lo'])}] (table: {x['ssm_1']} - ACT {x['diff']} [{x['diff_lo']}, "
              f"{x['diff_hi']}], nboot {x['nboot']})")
    k = [('2D-TTC' if x['ssm_1'] == 'TTC2D' else x['ssm_1']) for x in sd if x['condition'] == c and x['clear'] == 'True']
    print(f'    interval leaves out 0: {len(k)} of 8 {k}')

p_cs = os.path.join(A, 'condition_summary.csv')
cs = {x['condition']: x for x in rows_with_lines(p_cs) if x['model'] == MODEL}
p_td = os.path.join(A, 'vlm_requests_tidy.csv')
td = pd.read_csv(p_td, usecols=['model', 'segment', 'frame_idx', 'condition', 'repeat', 'risk'])
td = td[td.model == MODEL]
print(f'\n3. repeat stability per condition ({p_cs}; the range recomputed from {p_td})')
for c in CONDS:
    w = td[td.condition == c].pivot_table(index=['segment', 'frame_idx'], columns='repeat', values='risk')
    taus = []
    for x_, y_ in itertools.combinations(w.columns, 2):
        mm = w[[x_, y_]].dropna()
        if len(mm) >= 3:
            taus.append(kendalltau(mm[x_], mm[y_], variant='b').statistic)
    x = cs[c]
    dd = float(np.mean(taus)) - float(x['retest_tau_b_mean'])
    check(len(taus) == 10 and abs(dd) < 1e-12, f'{c}: mean of the {len(taus)} recomputed test-retest tau-b - '
          f'retest_tau_b_mean = {dd:.3g}')
    print(f"  {c} (line {x['_line']}): retest_tau_b_mean {x['retest_tau_b_mean']}, range {repr(float(min(taus)))} to "
          f"{repr(float(max(taus)))}; ICC(1,5) icc_mean_of_k {x['icc_mean_of_k']} (k_repeats {x['k_repeats']}, "
          f"frames_all_k {x['frames_all_k']}); sd_repeats_mean {x['sd_repeats_mean']}")

p_fs = os.path.join(A, 'vlm_frame_scores.csv')
fs = pd.read_csv(p_fs, usecols=['model', 'condition', 'segment', 'frame_idx', 'score_mean'], float_precision='round_trip')
fs = fs[fs.model == MODEL].pivot_table(index=['segment', 'frame_idx'], columns='condition', values='score_mean')
check(len(fs) == 1084 and fs[CONDS].notna().all().all(), f'{p_fs}: {len(fs)} frames with a score_mean in all 5 '
      f'conditions')
print(f'\n4. between conditions: Kendall tau-b of score_mean ({p_fs}), 1,084 frames')
rows4 = []
for c1, c2 in itertools.combinations(CONDS, 2):
    t = kendalltau(fs[c1], fs[c2], variant='b').statistic
    rows4.append(dict(condition_1=c1, condition_2=c2, frames=len(fs), tau_b=float(t)))
    print(f'  {c1:10s} x {c2:10s}: {repr(float(t))}')
pd.DataFrame(rows4).to_csv(os.path.join(OUT, 'exploratory_sec46_condition_tau_b.csv'), index=False)
p_ea = os.path.join(AGR, 'exploratory_agreement.csv')
print(f'  as written in {p_ea} (score_agreement.py, 2026-11-05):')
with open(p_ea, encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        print(f'    {i}: {line.rstrip()}')
print(f'  {os.path.join(AGR, "exploratory_agreement_params.txt")}:')
with open(os.path.join(AGR, 'exploratory_agreement_params.txt'), encoding='utf-8') as f:
    for line in f:
        print(f'    {line.rstrip()}')
print(f'\nwritten: exploratory_sec46_condition_tau_b.csv in {OUT}')
print(f'checks not met: {len(failed)}')
for w_ in failed:
    print('  ', w_)
sys.exit(1 if failed else 0)
