"""TTC with scalar speeds (|v|, the common textbook form) vs ours (velocity projected on the lane direction)."""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import kendalltau
ex, csv = sys.argv[1:3]
fs = pd.read_csv(csv)
rows = []
for seg in fs.segment.unique():
    o = pd.read_parquet(os.path.join(ex, seg, 'objects.parquet')).set_index(['frame_idx', 'id'])
    d = fs[(fs.segment == seg) & fs.lead_map.notna()]
    for r in d.itertuples():
        lv = o.loc[(r.frame_idx, r.lead_map)]
        v_lead_abs = float(np.hypot(lv.speed_x, lv.speed_y))
        rows.append(dict(segment=seg[:8], frame=r.frame_idx, gap=r.gap_map, ttc_ours=r.TTC,
                         v_ego_proj=r.v_ego_long, v_lead_proj=r.v_lead_long, v_ego_abs=r.ego_speed, v_lead_abs=v_lead_abs))
t = pd.DataFrame(rows)
t['clos_ours'] = t.v_ego_proj - t.v_lead_proj
t['clos_abs'] = t.v_ego_abs - t.v_lead_abs
t['ttc_abs'] = np.where(t.clos_abs > 0, t.gap / t.clos_abs, np.inf)
t['ttc_ours_chk'] = np.where(t.clos_ours > 0, t.gap / t.clos_ours, np.inf)
print('lead frames:', len(t), '| our TTC column reproduced:', np.allclose(np.where(np.isinf(t.ttc_ours), 1e9, t.ttc_ours), np.where(np.isinf(t.ttc_ours_chk), 1e9, t.ttc_ours_chk), rtol=1e-6))
print('speed: |v| minus projection (m/s)  ego: median %.4f max %.4f | lead: median %.4f max %.4f' % (
    (t.v_ego_abs - t.v_ego_proj).median(), (t.v_ego_abs - t.v_ego_proj).abs().max(),
    (t.v_lead_abs - t.v_lead_proj).median(), (t.v_lead_abs - t.v_lead_proj).abs().max()))
print('closing speed difference (m/s): median %.4f, 95th pct %.4f, max %.4f' % tuple((t.clos_abs - t.clos_ours).abs().quantile([.5, .95, 1.0])))
fin = np.isfinite(t.ttc_ours) | np.isfinite(t.ttc_abs)
print('frames with finite TTC: ours %d, scalar %d, either %d, both %d' % (np.isfinite(t.ttc_ours).sum(), np.isfinite(t.ttc_abs).sum(), fin.sum(), (np.isfinite(t.ttc_ours) & np.isfinite(t.ttc_abs)).sum()))
cap = 1e6
a, b = np.minimum(t.ttc_ours.replace(np.inf, cap), cap), np.minimum(t.ttc_abs.replace(np.inf, cap), cap)
print('Kendall tau-b over all lead frames (inf = tie at the top): %.4f' % kendalltau(a, b).statistic)
both = np.isfinite(t.ttc_ours) & np.isfinite(t.ttc_abs)
if both.any():
    rel = (t.ttc_abs[both] - t.ttc_ours[both]).abs() / t.ttc_ours[both]
    print('frames with both finite: min TTC ours %.2f s vs scalar %.2f s; relative TTC difference median %.3f, max %.3f' % (t.ttc_ours[both].min(), t.ttc_abs[both].min(), rel.median(), rel.max()))
    print('frames with TTC < 10 s: ours %d, scalar %d' % ((t.ttc_ours < 10).sum(), (t.ttc_abs < 10).sum()))
