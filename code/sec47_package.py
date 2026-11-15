"""Results package for Sec. 4.7 (user, 2026-11-15): the 11 model settings of the risk analysis with InternVL3.5-8B added
(analysis/after_internvl3.5-8b/main_1084: 1,084 frames, grouping all = all frame pairs). Values are copied as written in
the analysis tables (text, full precision); each row carries its line in the source file. Nothing is recomputed except
the mean score (from vlm_requests_tidy.csv and vlm_frame_scores.csv) and the min / max summaries.
1. exploratory_sec47_tau_b_sep_camera.csv: per setting x SSM (Table 1 order), sep_camera, grouping all:
   tau_b_conflict, its 95% interval and nboot, whether the interval excludes 0 (above / below / contains), agreement.csv line.
2. exploratory_sec47_settings_sep_camera.csv: per setting, sep_camera, from condition_summary.csv: retest_tau_b_mean (mean
   tau-b between runs), icc_mean_of_k (ICC of the mean of k = 5), icc_single, requests, answered (forced answers
   included), forced_answers, no_answer, hit_max_tokens; unreadable in the main run = no_answer + forced_answers (a forced
   answer is asked only where the main reply had none); mean score over the answers (tidy table, sep_camera, answers with
   a score) and over the 1,084 frame means (vlm_frame_scores.csv score_mean).
3. exploratory_sec47_excluding_zero.csv: per setting, the SSMs whose interval excludes 0, above and below, and PSD's.
4. exploratory_sec47_min_max.csv: min and max of tau_b_conflict (sep_camera, all) per SSM and over all 9 SSMs, over the 10
   general-purpose settings (all but qwen-drive-1.0-4b) and over all 11, with the setting(s) that reach them.
5. the all-settings x 5-formats heatmap and its CSV (code/supplement_20261114.heatmap) with the 11 settings, in the order
   below (the user's order of 2026-11-15).
usage: python -I sec47_package.py <waymo_v143> <out_dir>
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import supplement_20261114 as sup  # noqa: E402

FOLDER = 'after_internvl3.5-8b'
ORDER = ['qwen3.6-27b', 'gemma-4-31b-it', 'gemma-4-31b-it-nothink', 'internvl3.5-38b-thinking', 'internvl3.5-38b',
         'internvl3.5-8b-thinking', 'internvl3.5-8b', 'qwen3-vl-32b-thinking', 'qwen3-vl-32b', 'qwen3.5-4b',
         'qwen-drive-1.0-4b']
GENERAL = [m for m in ORDER if m != 'qwen-drive-1.0-4b']
SSMS = sup.SSMS
COND = 'sep_camera'


def main():
    L, O = sys.argv[1:3]
    os.makedirs(O, exist_ok=True)
    A = os.path.join(L, 'analysis', FOLDER, 'main_1084')
    ag = pd.read_csv(os.path.join(A, 'agreement.csv'), dtype=str, keep_default_na=False)
    ag['line'] = np.arange(len(ag)) + 2
    a = ag[(ag.condition == COND) & (ag.grouping == 'all')]
    sup.check(sorted(a.model.unique()) == sorted(ORDER), f'agreement.csv sep_camera / all: the 11 settings {sorted(a.model.unique())}')
    rows1 = []
    for m in ORDER:
        for s in SSMS:
            r = a[(a.model == m) & (a.ssm == s)]
            sup.check(len(r) == 1, f'{m} {s}: one row')
            r = r.iloc[0]
            lo, hi = float(r.tau_b_conflict_lo), float(r.tau_b_conflict_hi)
            rows1.append(dict(setting=m, ssm=s, ssm_label=sup.LABEL.get(s, s), tau_b=r.tau_b_conflict,
                              lo95=r.tau_b_conflict_lo, hi95=r.tau_b_conflict_hi, nboot=r.tau_b_conflict_nboot,
                              n_frames=r.n_conflict, side='above 0' if lo > 0 else 'below 0' if hi < 0 else 'contains 0',
                              agreement_csv_line=int(r.line)))
    t1 = pd.DataFrame(rows1)
    t1.to_csv(os.path.join(O, 'exploratory_sec47_tau_b_sep_camera.csv'), index=False)
    sup.check((t1.n_frames == '1084').all(), 'every row: 1,084 frames')

    cs = pd.read_csv(os.path.join(A, 'condition_summary.csv'), dtype=str, keep_default_na=False)
    cs['line'] = np.arange(len(cs)) + 2
    td = pd.read_csv(os.path.join(A, 'vlm_requests_tidy.csv'), low_memory=False)
    td = td[td.condition == COND]
    fs = pd.read_csv(os.path.join(A, 'vlm_frame_scores.csv'), float_precision='round_trip')
    fs = fs[fs.condition == COND]
    rows2 = []
    for m in ORDER:
        r = cs[(cs.model == m) & (cs.condition == COND)]
        sup.check(len(r) == 1, f'condition_summary {m}: one row')
        r = r.iloc[0]
        sm = td[td.model == m].risk.dropna()
        fm = fs[fs.model == m].score_mean
        rows2.append(dict(setting=m, retest_tau_b_mean=r.retest_tau_b_mean, icc_mean_of_k=r.icc_mean_of_k, k=r.k_repeats,
                          icc_single=r.icc_single, requests=r.requests, answered=r.answered, forced_answers=r.forced_answers,
                          no_answer=r.no_answer, hit_max_tokens=r.hit_max_tokens,
                          unreadable_in_main_run=int(r.no_answer) + int(r.forced_answers),
                          mean_score_answers=float(sm.mean()), answers_with_score=int(len(sm)),
                          mean_score_frame_means=float(fm.mean()), frames=int(len(fm)),
                          condition_summary_line=int(r.line)))
        sup.check(int(len(sm)) == int(r.answered) and len(fm) == 1084, f'{m}: answers with a score = answered; 1,084 frames')
    pd.DataFrame(rows2).to_csv(os.path.join(O, 'exploratory_sec47_settings_sep_camera.csv'), index=False)

    rows3 = []
    for m in ORDER:
        g = t1[t1.setting == m]
        rows3.append(dict(setting=m, excluding_zero=';'.join(g[g.side != 'contains 0'].ssm_label),
                          above_0=';'.join(g[g.side == 'above 0'].ssm_label), below_0=';'.join(g[g.side == 'below 0'].ssm_label),
                          n_excluding=int((g.side != 'contains 0').sum()), PSD_side=g[g.ssm == 'PSD'].side.iloc[0]))
    pd.DataFrame(rows3).to_csv(os.path.join(O, 'exploratory_sec47_excluding_zero.csv'), index=False)

    rows4 = []
    for name, sets in (('10 general-purpose (without qwen-drive-1.0-4b)', GENERAL), ('all 11', ORDER)):
        for s in SSMS + ['all 9 SSMs']:
            g = t1[t1.setting.isin(sets) & ((t1.ssm == s) if s != 'all 9 SSMs' else True)].copy()
            g['v'] = g.tau_b.astype(float)
            mn, mx = g.v.min(), g.v.max()
            rows4.append(dict(settings=name, n_settings=len(sets), ssm=sup.LABEL.get(s, s), min_tau_b=g.loc[g.v == mn, 'tau_b'].iloc[0],
                              min_at='; '.join(f'{x.setting} {x.ssm_label}' for x in g[g.v == mn].itertuples()),
                              max_tau_b=g.loc[g.v == mx, 'tau_b'].iloc[0],
                              max_at='; '.join(f'{x.setting} {x.ssm_label}' for x in g[g.v == mx].itertuples())))
    pd.DataFrame(rows4).to_csv(os.path.join(O, 'exploratory_sec47_min_max.csv'), index=False)

    sup.heatmap(L, O, folder=FOLDER, order=ORDER)
    print(f'checks not met: {len(sup.failed)}')
    for w in sup.failed:
        print('  ', w)
    sys.exit(1 if sup.failed else 0)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main()
