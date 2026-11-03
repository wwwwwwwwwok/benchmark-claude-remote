"""Test (2) of the analysis with a second file for a model (user, 2026-11-03): the 27B without thinking's grid replies
renamed qwen3.6-27b and added with --extra_quantities (as the run with thinking on grid will be). In the run with it,
for main_1084/ and sensitivity_1123/:
 - the rows of qwen3.6-27b x grid, but for the model name, character for character those of qwen3.6-27b-nothink x grid:
   quantity_frames, quantity_answers, quantity_tau_b, quantity_errors, accel_classes, perceived_ssm_frames,
   perceived_ssm_missing, exploratory_e1_frames, exploratory_quantity_errors, and the model-SSM-vs-true-SSM rows of
   perceived_ssm_tau_b and of the E1 table (its risk-score rows on grid have no counterpart: 7 SSMs x the groupings);
 - E2: its own estimates on grid, 2 x the groupings (no counterpart: the 27B without thinking has no risk scores);
 - risk_check.csv: its 9 SSMs x the groupings on grid, every one identical to the risk analysis;
 - exploratory_thinking_minus_no_thinking.csv: its grid rows, 5 quantities x (groupings + 1), every diff 0 and every
   interval [0, 0];
 - quantity_differences.csv: no line of it (no ctrl_grid, so no TIME_PAIRS).
That the old run's lines are all there, unchanged and in order, and nothing else more: compare_superset.py.
usage: python -I check_extra_condition_test.py <analysis folder with the second file>
"""
import os
import sys

new = sys.argv[1]
M, N, C = 'qwen3.6-27b', 'qwen3.6-27b-nothink', 'grid'
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
        true_only = name in SAME_TRUE
        keep = (lambda r: r[head.index('comparison')] == 'model_ssm_vs_true_ssm') if true_only else (lambda r: True)
        mine = [r[1:] for r in rows if r[0] == M and r[ci] == C and keep(r)]
        theirs = [r[1:] for r in rows if r[0] == N and r[ci] == C and keep(r)]
        other = [r for r in rows if r[0] == M and r[ci] == C and not keep(r)]
        report(len(mine) > 0 and mine == theirs,
               f'{name}: {M} x {C} {len(mine)} lines, {N} x {C} {len(theirs)}; the same but for the model name: '
               f'{mine == theirs}' + (f'; besides, {len(other)} risk-score rows' if other else ''))
        if true_only:
            ng = len({r[head.index('grouping')] for r in rows})
            report(len(other) == 7 * ng, f'{name}: its risk_score_vs_model_ssm rows on {C}: {len(other)} (expected 7 x {ng})')
    head, rows = lines(os.path.join(new, sub, 'exploratory_e2_risk_vs_distance_speed.csv'))
    h = {k: i for i, k in enumerate(head)}
    ng = len({r[h['grouping']] for r in rows})
    own = [r for r in rows if r[0] == M and r[h['condition']] == C and not r[h['variable']].startswith('true_')]
    report(len(own) == 2 * ng and {r[h['variable']] for r in own} == {'model_distance', 'model_ego_speed'},
           f'E2: its own estimates on {C} {len(own)} rows (expected 2 x {ng})')
    head, rows = lines(os.path.join(new, sub, 'risk_check.csv'))
    h = {k: i for i, k in enumerate(head)}
    rc = [r for r in rows if r[0] == M]
    report(len(rc) == 9 * ng and {r[h['condition']] for r in rc} == {C} and all(r[h['identical']] == 'True' for r in rc),
           f'risk_check: its rows {len(rc)} (expected 9 x {ng}) on {sorted({r[h["condition"]] for r in rc})}, all identical '
           f'to the risk analysis: {all(r[h["identical"]] == "True" for r in rc)}')
    head, rows = lines(os.path.join(new, sub, 'exploratory_thinking_minus_no_thinking.csv'))
    h = {k: i for i, k in enumerate(head)}
    g = [r for r in rows if r[h['condition_1']] == C]
    zero = all(float(r[h[k]]) == 0 for r in g for k in ('diff', 'diff_lo', 'diff_hi'))
    report(len(g) == 5 * (ng + 1) and {(r[h['model_1']], r[h['model_2']], r[h['condition_2']]) for r in g} == {(M, N, C)}
           and zero and all(r[h['value_1']] == r[h['value_2']] and r[h['clear']] == 'False' for r in g),
           f'with thinking - without on {C}: {len(g)} rows (expected 5 x {ng + 1}); every diff 0 and interval [0, 0]: '
           f'{zero}; nboot {sorted({r[h["nboot"]] for r in g})}; rows in all {len(rows)}')
    _, rows = lines(os.path.join(new, sub, 'quantity_differences.csv'))
    n = sum(M in r for r in rows)
    report(n == 0, f'quantity_differences.csv: lines naming {M}: {n}')
print(f'\nchecks not passed: {bad}')
