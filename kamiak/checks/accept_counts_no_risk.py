"""Two counts of the quantities acceptance, per model (Claude, 2026-10-30). quantities_accept.py (perceived_ssm_tau_b
rows) and quantities_explore_accept.py (E2 rows) expect the risk-related rows for every model as soon as one model has
them; code/analyze_quantities.py leaves them out by design for a model that is not in the risk analysis (Qwen3.6-27B
without thinking). Here, per model of the quantities: perceived_ssm_tau_b.csv rows = conditions x 7 SSMs x groupings
x (2 comparisons with risk scores, else 1: model SSM vs true SSM only); exploratory_e2_risk_vs_distance_speed.csv:
risk score vs the true distance and ego speed for every model of the risk analysis, vs the model's own estimates for
every model with both; and that no cell is missing a value.
usage: python -I accept_counts_no_risk.py <analysis subfolder> <risk analysis subfolder> [--intervals]
"""
import os
import sys

import pandas as pd

d, risk = sys.argv[1:3]
intervals = '--intervals' in sys.argv[3:]
pss = pd.read_csv(os.path.join(d, 'perceived_ssm_tau_b.csv'))
e2 = pd.read_csv(os.path.join(d, 'exploratory_e2_risk_vs_distance_speed.csv'))
qm = sorted(pd.read_csv(os.path.join(d, 'quantity_tau_b.csv')).model.unique())
rm = sorted(pd.read_csv(os.path.join(risk, 'vlm_frame_scores.csv'), usecols=['model']).model.unique())
nc, ng = pss.condition.nunique(), pss.grouping.nunique()
bad = 0


def report(ok, text):
    global bad
    bad += not ok
    print(f'  [{"ok" if ok else "NOT MET"}] {text}')


print(f'{d}: quantity models {qm}; risk models {rm}; conditions {nc}, groupings {ng}')
cols = ['tau_b'] + (['tau_b_lo', 'tau_b_hi'] if intervals else [])
for m in qm:
    s = pss[pss.model == m]
    exp = nc * 7 * ng * (2 if m in rm else 1)
    report(len(s) == exp and not s[cols].isna().any().any(),
           f'perceived_ssm_tau_b {m}: {len(s)} rows (expected {exp}; in the risk analysis: {m in rm}); comparisons '
           f'{s.comparison.value_counts().to_dict()}; missing values {int(s[cols].isna().sum().sum())}')
for m in sorted(set(qm) | set(rm)):
    s = e2[e2.model == m]
    exp = nc * ng * ((2 if m in rm else 0) + (2 if m in rm and m in qm else 0))
    report(len(s) == exp and not s[cols].isna().any().any(),
           f'E2 {m}: {len(s)} rows (expected {exp}); variables {s.variable.value_counts().to_dict()}; missing values '
           f'{int(s[cols].isna().sum().sum())}')
report(set(pss.model) <= set(qm) and set(e2.model) <= set(qm) | set(rm), 'no other models in the two tables')
print(f'\nchecks not passed: {bad}')
