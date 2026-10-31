"""Test (2) of the analysis program with a model that has one condition only (user, 2026-10-30): the 27B without
thinking's sep_camera replies renamed qwen3.6-27b and added as that model (--only_conditions qwen3.6-27b=sep_camera).
For main_1084/ and sensitivity_1123/ of the run with it (new) and of the run without it (old, the same program):
 - its own tables, but for the model name, character for character those of qwen3.6-27b-nothink on sep_camera:
   quantity_frames, quantity_answers, quantity_tau_b, quantity_errors, accel_classes, perceived_ssm_frames,
   perceived_ssm_missing, exploratory_e1_frames, exploratory_quantity_errors, and the model-SSM-vs-true-SSM rows of
   perceived_ssm_tau_b and of the E1 table (its risk-score rows are its own: 7 SSMs x the groupings, sep_camera only);
 - exploratory_thinking_minus_no_thinking.csv: sep_camera only, 5 quantities x (groupings + 1) rows, every diff 0 and
   every interval [0, 0];
 - no line of it in quantity_differences.csv (no TIME_PAIRS, no MODEL_PAIRS) or risk_check.csv (no grid);
 - E2: its rows of the old run (it is a risk-analysis model: true distance and ego speed, 5 conditions) unchanged and
   in order, plus its own estimates on sep_camera only.
The other models' lines: compare_without_model.py. usage: python -I check_single_condition_test.py <old> <new>
"""
import os
import sys

old, new = sys.argv[1:3]
M, N, C = 'qwen3.6-27b', 'qwen3.6-27b-nothink', 'sep_camera'
SAME_ALL = ['quantity_frames.csv', 'quantity_answers.csv', 'quantity_tau_b.csv', 'quantity_errors.csv',
            'accel_classes.csv', 'perceived_ssm_frames.csv', 'perceived_ssm_missing.csv', 'exploratory_e1_frames.csv',
            'exploratory_quantity_errors.csv']
SAME_TRUE = ['perceived_ssm_tau_b.csv', 'exploratory_e1_perceived_ssm_tau_b.csv']
bad = 0


def report(ok, text):
    global bad
    bad += not ok
    print(f'  [{"ok" if ok else "NOT OK"}] {text}')


def lines(path):
    t = open(path).read().splitlines()
    return t[0].split(','), [l.split(',') for l in t[1:]]


for sub in ('main_1084', 'sensitivity_1123'):
    print(f'== {sub}')
    for name in SAME_ALL + SAME_TRUE:
        head, rows = lines(os.path.join(new, sub, name))
        ci = head.index('condition')
        keep = (lambda r: True) if name in SAME_ALL else (lambda r: r[head.index('comparison')] == 'model_ssm_vs_true_ssm')
        mine = [r[1:] for r in rows if r[0] == M and keep(r)]
        theirs = [r[1:] for r in rows if r[0] == N and r[ci] == C and keep(r)]
        other = [r for r in rows if r[0] == M and not keep(r)]
        conds = sorted({r[ci] for r in rows if r[0] == M})
        report(len(mine) > 0 and mine == theirs and conds == [C],
               f'{name}: {M} {len(mine)} lines, {N} on {C} {len(theirs)}; the same but for the model name: '
               f'{mine == theirs}; its conditions {conds}' + (f'; besides, {len(other)} risk-score rows' if other else ''))
        if name in SAME_TRUE:
            ng = len({r[head.index('grouping')] for r in rows})
            report(len(other) == 7 * ng and {r[ci] for r in other} == {C},
                   f'{name}: its risk_score_vs_model_ssm rows {len(other)} (expected 7 x {ng}), on {sorted({r[ci] for r in other})}')
    head, rows = lines(os.path.join(new, sub, 'exploratory_thinking_minus_no_thinking.csv'))
    h = {k: i for i, k in enumerate(head)}
    ng = len({r[h['grouping']] for r in rows if r[h['stat']] == 'tau_b'})
    zero = all(float(r[h[k]]) == 0 for r in rows for k in ('diff', 'diff_lo', 'diff_hi'))
    report(len(rows) == 5 * (ng + 1) and {(r[h['model_1']], r[h['model_2']]) for r in rows} == {(M, N)}
           and {(r[h['condition_1']], r[h['condition_2']]) for r in rows} == {(C, C)} and zero
           and all(r[h['value_1']] == r[h['value_2']] and r[h['clear']] == 'False' for r in rows),
           f'with thinking - without: {len(rows)} rows (expected 5 x {ng + 1}), {M} - {N} on {C} only; every diff 0 '
           f'and interval [0, 0]: {zero}; nboot {sorted({r[h["nboot"]] for r in rows})}')
    for name in ('quantity_differences.csv', 'risk_check.csv'):
        _, rows = lines(os.path.join(new, sub, name))
        n = sum(M in r for r in rows)
        report(n == 0, f'{name}: lines naming {M}: {n}')
    head, rows_new = lines(os.path.join(new, sub, 'exploratory_e2_risk_vs_distance_speed.csv'))
    _, rows_old = lines(os.path.join(old, sub, 'exploratory_e2_risk_vs_distance_speed.csv'))
    h = {k: i for i, k in enumerate(head)}
    mn = [r for r in rows_new if r[0] == M]
    mo = [r for r in rows_old if r[0] == M]
    kept = [r for r in mn if r[h['variable']].startswith('true_')]
    own = [r for r in mn if not r[h['variable']].startswith('true_')]
    ng = len({r[h['grouping']] for r in rows_new})
    report(kept == mo and len(mo) == 5 * 2 * ng and len(own) == 2 * ng and {r[h['condition']] for r in own} == {C},
           f'E2: its rows of the old run {len(mo)}, the same and in order in the new: {kept == mo}; its own estimates '
           f'{len(own)} rows (expected 2 x {ng}) on {sorted({r[h["condition"]] for r in own})}')
print(f'\nchecks not passed: {bad}')
