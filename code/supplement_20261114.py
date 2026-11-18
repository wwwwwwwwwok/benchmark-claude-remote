"""EXPLORATORY / supplementary material (user, 2026-11-14). CPU only; no image is opened (the candidate images are copied
as files, byte for byte, by name).
1. Heatmap of all model settings x input formats: tau_b_conflict, grouping all, from
   analysis/after_gemma-4-31b-it-nothink/main_1084/agreement.csv (9 settings, 5 input formats, 9 SSMs in Table 1 order:
   TTC THW DRAC MTTC CI PICUD PSD 2D-TTC ACT); one panel per input format, one row per setting (in the order the settings
   first appear in agreement.csv), colour scale -0.5..0.5 (RdBu_r), values written in the cells (2 decimals, normal
   weight). Writes the full-precision CSV it is drawn from (with each value's line in agreement.csv), PNG and PDF. The candidate images: the last 5 paths of images_by_time in separate/targets_main.csv and the last path in grid/targets_main.csv (file names are zero-padded, e.g. step_025.jpg).
2. Weather (labels/scene_labels.csv column weather: dry / wet / rain, the user's labels) and time of day (Waymo's
   time_of_day): segments and frames per class, and per weather x time of day, for the 1,084-frame main set
   (ssm_target_frames.csv of the risk analysis).
3. Candidate frames for qualitative examples (main set, sep_camera, Qwen3.6-27B), by numbers only:
   frame score = score_mean (vlm_frame_scores.csv), single scores and hazard types = the risk analysis's tidy table
   (round 1's answers_base.csv), reasons = round 1's reasons.jsonl, k-means k 7 clusters = round 1's
   exploratory_reason_answers.csv (matched by segment, frame_idx, repeat); SSMs = ssm_target_frames.csv (the values the
   analysis uses); gap and ego speed (v_ego) = results_lead_final_excl/frame_ssm_validation_training.csv; percentiles of
   score_mean over the 1,084 frames (numpy, linear); medians of TTC, MTTC, ACT over the 1,084 frames.
   (a) score_mean <= P10 and >= 3 of 5 reasons in cluster 0; (b) score_mean >= P90 and >= 3 in cluster 5;
   (c) (TTC < 4 or MTTC < 4) and >= 3 in clusters 3 or 4 and P40 <= score_mean <= P60;
   (d) score_mean >= P90 and TTC, MTTC and ACT all > their medians.
   Selection per type: numpy default_rng(20261114), a new generator per type; if 5 or more segments qualify, 5 segments
   by rng.choice(sorted segments, 5, replace=False), then in each one frame by rng.choice(its sorted frame_idx); if fewer,
   every qualifying segment (one frame each, chosen the same way) and the shortfall is reported.
   With --copy-to: for each candidate a folder <type>_<rank>_<segment>_<frame_idx> with the 5 camera images at time t
   that the model was sent (vlm_inputs/separate/<segment>/<CAMERA>_<frame_idx>.jpg, the last time step of each camera in
   targets_main.csv) and the tiled grid image at t (vlm_inputs/grid/<segment>/step_<frame_idx>.jpg), copied with
   shutil.copyfile (bytes only), plus candidates.csv and sha256.txt at the top.
Checks (exit 1 when one is not met): 1,084 frames, 238 segments; 9 settings x 5 formats x 9 SSMs with a value; the 5
answers' mean = score_mean; every frame has a weather and a time of day; the copied files' sha256 = the sources'.
usage: python -I supplement_20261114.py <waymo_v143> <round-1 work folder> <repository> <out_dir> [--copy-to <folder>]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MODEL, COND = 'qwen3.6-27b', 'sep_camera'
SSMS = ['TTC', 'THW', 'DRAC', 'MTTC', 'CI', 'PICUD', 'PSD', 'TTC2D', 'ACT']
LABEL = {'TTC2D': '2D-TTC'}
CONDS = ['grid', 'sep_time', 'sep_camera', 'ctrl_grid', 'ctrl_sep']
CAMS = ['FRONT_LEFT', 'FRONT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT']
SEED = 20261114
failed = []


def check(ok, what):
    print(('  check ok: ' if ok else '  CHECK NOT MET: ') + what, flush=True)
    if not ok:
        failed.append(what)


def sha(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def heatmap(L, O, folder='after_gemma-4-31b-it-nothink', order=None, drop=None):
    """order: the settings in this order (default: as they first appear in agreement.csv); added 2026-11-15 for the
    11-setting figure, the defaults give the 2026-11-14 output unchanged. drop: settings left out before the order check
    (added 2026-11-17 for the supplement's 9-setting figure; None = unchanged)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p_ag = os.path.join(L, 'analysis', folder, 'main_1084', 'agreement.csv')
    ag = pd.read_csv(p_ag, dtype=str, keep_default_na=False)
    ag['line'] = np.arange(len(ag)) + 2
    a = ag[ag.grouping == 'all']
    models = list(dict.fromkeys(a.model))
    if drop is not None:
        check(set(drop) <= set(models), f'heatmap: settings to drop {drop} are in agreement.csv')
        models = [m for m in models if m not in set(drop)]
    if order is not None:
        check(sorted(order) == sorted(models), f'heatmap: the given order names the {len(models)} settings in agreement.csv')
        models = list(order)
    rows, bad = [], []
    for c in CONDS:
        for m in models:
            for s in SSMS:
                r = a[(a.model == m) & (a.condition == c) & (a.ssm == s)]
                if not (len(r) == 1 and r.tau_b_conflict.iloc[0] != ''):
                    bad.append((m, c, s))
                    continue
                r = r.iloc[0]
                rows.append(dict(condition=c, model=m, ssm=s, ssm_label=LABEL.get(s, s), tau_b_conflict=r.tau_b_conflict,
                                 lo95=r.tau_b_conflict_lo, hi95=r.tau_b_conflict_hi, nboot=r.tau_b_conflict_nboot,
                                 agreement_csv_line=int(r.line)))
    check(not bad and len(models) == (9 if order is None else len(order)) and len(rows) == len(models) * 5 * 9,
          f'heatmap: {len(models)} settings x 5 formats x 9 SSMs, each with one value (missing {bad})')
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(O, 'exploratory_supp_heatmap_models_formats.csv'), index=False)
    fig, axes = plt.subplots(1, 5, figsize=(5 * 3.6 + 1.2, 0.42 * len(models) + 1.6), sharey=True)
    for ax, c in zip(axes, CONDS):
        M = np.array([[float(df[(df.condition == c) & (df.model == m) & (df.ssm == s)].tau_b_conflict.iloc[0])
                       for s in SSMS] for m in models])
        im = ax.imshow(M, cmap='RdBu_r', vmin=-0.5, vmax=0.5, aspect='auto')
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=6.5, fontweight='normal')
        ax.set_xticks(range(len(SSMS)), [LABEL.get(s, s) for s in SSMS], rotation=90, fontsize=7)
        ax.set_yticks(range(len(models)), models, fontsize=7)
        ax.set_title(c, fontsize=9, fontweight='normal')
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label('Kendall tau-b (grouping all)', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(O, f'exploratory_supp_heatmap_models_formats.{ext}'), dpi=200, bbox_inches='tight',
                    metadata=({'CreationDate': None} if ext == 'pdf' else {'Software': None}))
    plt.close(fig)
    return p_ag, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('L')
    ap.add_argument('W1')
    ap.add_argument('repo')
    ap.add_argument('out')
    ap.add_argument('--copy-to')
    a = ap.parse_args()
    L, O = a.L, a.out
    os.makedirs(O, exist_ok=True)
    A = os.path.join(L, 'analysis', 'after_gemma-4-31b-it', 'main_1084')
    print(f'EXPLORATORY supplementary material; program sha256 {sha(os.path.abspath(__file__))}')

    # ---- 1. heatmap
    p_ag, models = heatmap(L, O)

    # ---- frames, scene labels
    st = pd.read_csv(os.path.join(A, 'ssm_target_frames.csv'), float_precision='round_trip') \
        .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    check(len(st) == 1084 and st.segment.nunique() == 238, f'{len(st)} frames, {st.segment.nunique()} segments')
    sl = pd.read_csv(os.path.join(a.repo, 'labels', 'scene_labels.csv'), keep_default_na=False)
    fr = st[['segment', 'frame_idx']].merge(sl[['segment', 'time_of_day', 'weather']], on='segment', how='left')
    check(fr.weather.ne('').all() and fr.time_of_day.ne('').all() and not fr.isna().any().any(),
          'every main-set frame has a weather and a time of day')

    # ---- 2. weather / time-of-day table
    rows = []
    for col in ('weather', 'time_of_day'):
        for v, g in fr.groupby(col):
            rows.append(dict(table=col, weather=v if col == 'weather' else '', time_of_day=v if col == 'time_of_day' else '',
                             segments=g.segment.nunique(), frames=len(g)))
    for (w, t), g in fr.groupby(['weather', 'time_of_day']):
        rows.append(dict(table='weather x time_of_day', weather=w, time_of_day=t, segments=g.segment.nunique(),
                         frames=len(g)))
    rows.append(dict(table='all', weather='', time_of_day='', segments=fr.segment.nunique(), frames=len(fr)))
    pd.DataFrame(rows).to_csv(os.path.join(O, 'exploratory_supp_weather_time.csv'), index=False)

    # ---- 3. candidates
    td = pd.read_csv(os.path.join(a.W1, 'answers_base.csv'), keep_default_na=False, na_values=[''],
                     float_precision='round_trip')
    recs = [json.loads(x) for x in open(os.path.join(a.W1, 'reasons.jsonl'), encoding='utf-8')]
    check([r['row'] for r in recs] == list(range(len(td))), 'reasons in answers_base row order')
    td['reason'] = [r['reason'] for r in recs]
    ca = pd.read_csv(os.path.join(L, 'analysis', 'reason_27b', 'main_1084', 'exploratory_reason_answers.csv'))
    td = td.merge(ca[['segment', 'frame_idx', 'repeat', 'cluster', 'risk']].rename(columns={'risk': 'risk_c'}),
                  on=['segment', 'frame_idx', 'repeat'], how='left', validate='one_to_one')
    check(not td.cluster.isna().any() and np.array_equal(td.risk.to_numpy(float), td.risk_c.to_numpy(float)),
          'clusters matched by segment, frame_idx, repeat; same risk')
    fs = pd.read_csv(os.path.join(A, 'vlm_frame_scores.csv'), float_precision='round_trip')
    fs = fs[(fs.model == MODEL) & (fs.condition == COND)][['segment', 'frame_idx', 'score_mean']]
    w5 = td.pivot(index=['segment', 'frame_idx'], columns='repeat')
    f = st[['segment', 'frame_idx'] + SSMS].merge(fs, on=['segment', 'frame_idx'], how='left') \
        .merge(fr, on=['segment', 'frame_idx'])
    raw = pd.read_csv(os.path.join(L, 'results_lead_final_excl', 'frame_ssm_validation_training.csv'), low_memory=False,
                      float_precision='round_trip')[['segment', 'frame_idx', 'gap', 'v_ego']]
    f = f.merge(raw, on=['segment', 'frame_idx'], how='left', validate='one_to_one')
    key = pd.MultiIndex.from_frame(f[['segment', 'frame_idx']])
    for r in range(5):
        f[f'score_r{r}'] = w5['risk'][r].reindex(key).to_numpy(float)
        f[f'hazard_type_r{r}'] = w5['hazard_type'][r].reindex(key).to_numpy()
        f[f'cluster_r{r}'] = w5['cluster'][r].reindex(key).to_numpy(int)
        f[f'reason_r{r}'] = w5['reason'][r].reindex(key).to_numpy()
    sc5 = f[[f'score_r{r}' for r in range(5)]].to_numpy(float)
    check(np.array_equal(sc5.mean(1), f.score_mean.to_numpy(float)) and not f.gap.isna().any(),
          'the 5 answers\' mean = score_mean on every frame; every frame has gap and ego speed')
    cl5 = f[[f'cluster_r{r}' for r in range(5)]].to_numpy(int)
    s = f.score_mean.to_numpy(float)
    p10, p40, p60, p90 = np.percentile(s, [10, 40, 60, 90])
    med = {c: float(np.median(f[c])) for c in ('TTC', 'MTTC', 'ACT')}
    rules = {
        'a': (s <= p10) & ((cl5 == 0).sum(1) >= 3),
        'b': (s >= p90) & ((cl5 == 5).sum(1) >= 3),
        'c': ((f.TTC < 4) | (f.MTTC < 4)).to_numpy() & (np.isin(cl5, [3, 4]).sum(1) >= 3) & (s >= p40) & (s <= p60),
        'd': (s >= p90) & (f.TTC > med['TTC']).to_numpy() & (f.MTTC > med['MTTC']).to_numpy() & (f.ACT > med['ACT']).to_numpy(),
    }
    names = {'a': 'quiet residential street, low score', 'b': 'pedestrian crossing, high score',
             'c': 'rear-end rated dangerous by SSMs, mid score', 'd': 'high score, SSMs rate safe'}
    out, rows_q = [], []
    for t, m in rules.items():
        q = f[m]
        segs = sorted(q.segment.unique())
        rng = np.random.default_rng(SEED)
        pick = list(rng.choice(segs, 5, replace=False)) if len(segs) >= 5 else segs
        rows_q.append(dict(type=t, name=names[t], qualifying_frames=int(m.sum()), qualifying_segments=len(segs),
                           chosen=len(pick), shortfall=max(0, 5 - len(segs))))
        for k, sg in enumerate(pick, 1):
            fis = sorted(q[q.segment == sg].frame_idx.astype(int))
            fi = int(rng.choice(fis))
            row = q[(q.segment == sg) & (q.frame_idx == fi)].iloc[0].to_dict()
            out.append(dict(type=t, type_name=names[t], rank=k, frames_qualifying_in_segment=len(fis), **row))
    cols = (['type', 'type_name', 'rank', 'segment', 'frame_idx', 'frames_qualifying_in_segment', 'time_of_day', 'weather',
             'score_mean'] + [f'score_r{r}' for r in range(5)] + [f'hazard_type_r{r}' for r in range(5)]
            + [f'cluster_r{r}' for r in range(5)] + SSMS + ['v_ego', 'gap'] + [f'reason_r{r}' for r in range(5)])
    cand = pd.DataFrame(out)[cols].rename(columns={'TTC2D': '2D-TTC', 'v_ego': 'ego_speed_mps', 'gap': 'gap_m'})
    cand.to_csv(os.path.join(O, 'exploratory_supp_candidates.csv'), index=False)
    pd.DataFrame(rows_q).to_csv(os.path.join(O, 'exploratory_supp_candidates_counts.csv'), index=False)
    with open(os.path.join(O, 'exploratory_supp_params.txt'), 'w', encoding='utf-8') as fh:
        fh.write(f'EXPLORATORY supplementary material (user, 2026-11-14)\nrun {time.strftime("%Y-%m-%d %H:%M:%S")}; out_dir '
                 f'{os.path.abspath(O)}\nprogram {os.path.abspath(__file__)} sha256 {sha(os.path.abspath(__file__))}\n'
                 f'heatmap: {p_ag}, grouping all, tau_b_conflict; settings in this order: {models}\n'
                 f'score_mean percentiles over 1,084 frames (numpy linear): P10 {p10!r}, P40 {p40!r}, P60 {p60!r}, P90 '
                 f'{p90!r}; medians TTC {med["TTC"]!r}, MTTC {med["MTTC"]!r}, ACT {med["ACT"]!r}\n'
                 f'selection: default_rng({SEED}) per type; 5 segments by rng.choice(sorted, 5, replace=False), then one '
                 f'frame per segment by rng.choice(sorted frame_idx)\n')
    print(f'P10 {p10!r}, P40 {p40!r}, P60 {p60!r}, P90 {p90!r}; medians {med}')
    print(pd.DataFrame(rows_q).to_string(index=False))

    if a.copy_to:
        D = a.copy_to
        os.makedirs(D, exist_ok=False)
        tg = pd.read_csv(os.path.join(L, 'vlm_inputs', 'separate', 'targets_main.csv'))
        tgg = pd.read_csv(os.path.join(L, 'vlm_inputs', 'grid', 'targets_main.csv'))
        shas = []
        for r in cand.itertuples():
            sub = os.path.join(D, f'{r.type}_{r.rank}_{r.segment}_{r.frame_idx}')
            os.makedirs(sub)
            t_row = tg[(tg.segment == r.segment) & (tg.frame_idx == r.frame_idx)]
            check(len(t_row) == 1, f'{r.segment} {r.frame_idx} in separate/targets_main.csv')
            # the time step t: the last 5 paths of images_by_time (one per camera, in CAMS order); the grid: the last path
            last5 = t_row.images_by_time.iloc[0].split(';')[-5:]
            g_row = tgg[(tgg.segment == r.segment) & (tgg.frame_idx == r.frame_idx)]
            g_last = g_row.images.iloc[0].split(';')[-1]
            ok = ([p.split('/')[-1].rsplit('_', 1)[0] for p in last5] == CAMS
                  and len({p.rsplit('_', 1)[1] for p in last5}) == 1 and int(last5[0].rsplit('_', 1)[1][:-4]) == r.frame_idx
                  and int(g_last.rsplit('_', 1)[1][:-4]) == r.frame_idx and len(g_row) == 1)
            check(ok, f'{r.segment} {r.frame_idx}: 5 camera images at t and the grid image at t found in targets_main.csv')
            src = [os.path.join(L, 'vlm_inputs', 'separate', p) for p in last5] + \
                [os.path.join(L, 'vlm_inputs', 'grid', g_last)]
            for p in src:
                dst = os.path.join(sub, os.path.basename(p))
                shutil.copyfile(p, dst)
                h1, h2 = sha(p), sha(dst)
                if h1 != h2:
                    check(False, f"copied {p} -> {dst}: sha256 differs")
                shas.append(f'{h2}  {os.path.relpath(dst, D)}')
        cand.to_csv(os.path.join(D, 'candidates.csv'), index=False)
        with open(os.path.join(D, 'sha256.txt'), 'w') as fh:
            fh.write('\n'.join(shas) + '\n')
    print(f'checks not met: {len(failed)}')
    for w in failed:
        print('  ', w)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
