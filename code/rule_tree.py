"""EXPLORATORY: can Qwen3.6-27B's risk scores be written as one decision tree? (user, 2026-11-04, decided before any
tree was seen; CLAUDE.md "找模型自己的规则", the entry of 2026-11-04 on the decision tree). Every output is exploratory
(exploratory_tree_*); the inputs are only read and no existing program is changed (analyze_rules.py is imported for its
functions and constants only).

Frames, dependent variable, resamples: as code/analyze_rules.py. The frames of the risk analysis subfolder
(ssm_target_frames.csv: 1,084 main, 1,123 sensitivity), sorted by segment and frame; the dependent variable is the risk
score of qwen3.6-27b (<risk>/vlm_frame_scores.csv score_mean), one tree per condition, as its rank among these frames
divided by their number (ties: the mean rank; scipy rankdata); the risk analysis's resamples of whole segments (numpy
default_rng(--seed), one rng.choice over the sorted segments per resample), checked before they are used against the
risk analysis's params.txt (number, seed, frames, segment list) and the sha256 of the drawn segment indices in the
quantities analysis's params.txt.

Cues, in their own units (not ranked):
  A  the six true quantities of <truth.csv> (as the rule analysis): distance (m), ego speed, lead speed (m/s), closing
     speed = ego speed - lead speed (m/s), ego and lead acceleration (m/s2);
  B  A; the S of the rule analysis (the same 0/1 dummies: time of day and location with the category with the most
     frames as reference, weather with the class with the most segments as reference; a reference has no column of
     its own); the pedestrians + cyclists and the vehicles within 10, 20, 30 and 50 m (<nearby_counts.csv>, main
     version: in the cameras' view, occluded ones included), all eight counts in one tree;
  C  (sep_camera only) its own estimates of the same six (<quantities>/quantity_frames.csv estimate, the median of the
     repeats, as K_own; own closing speed = own ego speed - own lead speed).
  No H (the hazard type and camera written in the same replies).
Tree: sklearn.tree.DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=0), every other setting the
default (squared error, best split, every cue considered at every node). scikit-learn works on float32 copies of the
cues, so a cut is a midpoint between two float32 values; frames with a value <= the cut go left.

Outputs (in out_dir, every name exploratory_tree_*):
  nodes.csv   every node of every depth-3 tree grown on all the frames: the risk score's (5 conditions x A, B; C on
              sep_camera) and the seven lead-vehicle SSMs' on A (the SSM "larger = riskier" ranked like the risk score;
              its raw value in the SSM's own units): depth, parent, side, path, the cut (cue, threshold), frames,
              segments, mean rank, mean raw score (raw value);
  trees.txt   the same trees written out;
  oos.csv     out-of-sample R2 by leaving one segment out, as analyze_rules.py --oos: for every segment a tree is grown
              on the other segments' frames (the ranks once, on all the frames) and predicts that segment's frames;
              R2 = 1 - sum of squared prediction errors over all frames / sum of squares of the rank about its mean
              over all frames (may be negative). Interval: nothing is regrown; per resample the drawn segments' error and
              total sums of squares are added (a segment drawn twice counts twice) and R2 computed from them; 2.5 / 97.5 %
              percentiles, none when fewer than 90 % are defined; it leaves out the variability of the regrowing. The
              risk score with max_depth 1 to 5 (min_samples_leaf 50; 3 is the tree of nodes.csv, the others only
              describe), the SSMs with max_depth 3; the in-sample R2 of the tree grown on all the frames beside;
  vs_linear.csv  the depth-3 tree's out-of-sample R2 beside the same cell's linear fit in
              <rules>/exploratory_oos_r2.csv (A: K; B: K+S+N10, K+S+N20, K+S+N30, K+S+N50; C: K_own);
  stability.csv, stability_summary.csv  per condition the A tree regrown on every resample (the drawn frames, a
              segment drawn twice counted twice, the ranks taken anew among them): its first cut and the two below it;
              the share of resamples whose first cut is on each cue, and for the most frequent cue the 2.5 / 50 / 97.5 %
              percentiles of its cut;
  params.txt  inputs, versions, settings, checks.
Checks (exit code 1 when one is not met): the resamples, before they are used (stop); the linear fits rebuilt here from
the rule analysis's variables (K, K+S+N_R, K_own on ranks; analyze_rules.ols and loso_predict) give the in-sample R2,
the out-of-sample R2 and its interval of <rules>/exploratory_oos_r2.csv within LIN_TOL, with the same frames, segments
and resamples (so the frames, the dependent variable, the cues and the resamples are those of the rule analysis);
every node's frames and mean are scikit-learn's (n_node_samples, value); every frame is predicted once in each
leave-one-segment-out run.

usage: python -I rule_tree.py <risk analysis subfolder> <quantities analysis subfolder> <frame_ssm.csv> <truth.csv>
       <scene_labels.csv> <nearby_counts.csv> <rule analysis subfolder> <out_dir> [--n_boot 2000] [--seed 20261021]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import hashlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402

MODEL = 'qwen3.6-27b'
OWN_CONDITION = 'sep_camera'          # the only condition with its own estimates (thinking on)
MAX_DEPTH, MIN_LEAF, TREE_SEED = 3, 50, 0
DEPTHS = [1, 2, 3, 4, 5]              # out of sample only, descriptive (3 is the tree reported)
LIN_TOL = 1e-12                       # the linear fits rebuilt here against exploratory_oos_r2.csv
NODE_TOL = 1e-12                      # a node's mean here against scikit-learn's value
LINEAR = {'A': ['K'], 'B': [f'K+S+N{r}' for r in ar.RADII], 'C': ['K_own']}


def grow(X, y, depth=MAX_DEPTH):
    return DecisionTreeRegressor(max_depth=depth, min_samples_leaf=MIN_LEAF, random_state=TREE_SEED).fit(X, y)


def loso_tree(X, y, g, depth):
    """Leave-one-segment-out predictions of a tree: for every segment code k of g, the tree grown on the frames of the
    other segments, applied to the frames of k."""
    pred = np.full(len(y), np.nan)
    for k in range(int(g.max()) + 1):
        te = g == k
        pred[te] = grow(X[~te], y[~te], depth).predict(X[te])
    return pred


def r2_oos(y, pred, u, g, nseg, counts):
    """Out-of-sample R2 and its interval, as analyze_rules.py --oos: per segment the error and total sums of squares
    (about the mean over all frames); the resamples add those of their drawn segments (counts: resamples x segments)."""
    sse, sst = np.zeros(nseg), np.zeros(nseg)
    sse[u] = np.bincount(g, (y - pred) ** 2)
    sst[u] = np.bincount(g, (y - y.mean()) ** 2)
    rb = 1.0 - (counts @ sse) / (counts @ sst)
    lo, hi, nv = ar.ci(rb)
    return 1.0 - sse.sum() / sst.sum(), lo, hi, nv


def describe(tree, X, y, raw, names, seg_code):
    """Every node of a fitted tree: rows (depth, parent, side, path, cue, threshold, frames, segments, mean rank, mean
    raw) and text lines; the number of nodes whose frames or mean differ from scikit-learn's."""
    t = tree.tree_
    dp = tree.decision_path(X).tocsc()
    rows, lines, bad = [], [], 0

    def walk(j, parent, side, depth, path):
        nonlocal bad
        members = dp.indices[dp.indptr[j]:dp.indptr[j + 1]]
        leaf = t.children_left[j] == -1
        mr = float(y[members].mean())
        bad += int(len(members) != t.n_node_samples[j] or abs(mr - float(t.value[j].ravel()[0])) > NODE_TOL)
        r = dict(node=j, depth=depth, parent=parent, side=side, path=' & '.join(path) if path else '(all frames)',
                 leaf=bool(leaf), cue='' if leaf else names[t.feature[j]],
                 threshold=np.nan if leaf else float(t.threshold[j]), frames=int(len(members)),
                 segments=int(len(np.unique(seg_code[members]))), mean_rank=mr, mean_raw=float(raw[members].mean()))
        rows.append(r)
        pre = '    ' * depth + (f'[{side}] ' if side else '')
        stat = (f'frames {r["frames"]}, segments {r["segments"]}, mean rank {mr:.3f}, mean raw {r["mean_raw"]:.3f}')
        if leaf:
            lines.append(f'{pre}leaf {j}: {stat}')
        else:
            lines.append(f'{pre}node {j}: cut {r["cue"]} <= {r["threshold"]:.4g} | {stat}')
            c = f'{r["cue"]} <= {r["threshold"]:.4g}'
            walk(t.children_left[j], j, '<=', depth + 1, path + [c])
            walk(t.children_right[j], j, '>', depth + 1, path + [c.replace('<=', '>')])

    walk(0, -1, '', 0, [])
    return rows, lines, bad


def first_cuts(tree, names):
    """The first cut and the two below it: (cue, threshold) each; ('leaf', nan) for a leaf, ('none', nan) below one."""
    t = tree.tree_

    def split(j):
        if j < 0:
            return 'none', np.nan
        if t.children_left[j] == -1:
            return 'leaf', np.nan
        return names[t.feature[j]], float(t.threshold[j])

    return split(0), split(t.children_left[0]), split(t.children_right[0])


def main():
    ap = argparse.ArgumentParser()
    for name in ('risk', 'quantities', 'frame_ssm', 'truth', 'scene_labels', 'nearby', 'rules', 'out_dir'):
        ap.add_argument(name)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261021)
    a = ap.parse_args()
    if os.path.exists(os.path.join(a.out_dir, 'exploratory_tree_params.txt')):
        sys.exit(f'{a.out_dir} has exploratory_tree_params.txt already: give a new folder')
    t0 = time.time()
    failed, checks = [], []

    # --- frames and SSMs, as analyze_rules.py
    st = pd.read_csv(os.path.join(a.risk, 'ssm_target_frames.csv'))
    frames = st[['segment', 'frame_idx']].sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    n = len(frames)
    mi = pd.MultiIndex.from_frame(frames)
    ssm_df = av.ssm_table(a.frame_ssm, frames)
    for s in ar.LEAD7:
        if not (ssm_df[s + '_conflict'].all() and ssm_df[s + '_use'].all()):
            sys.exit(f'{s}: not every frame has a value (or some frames are left out of it); this program assumes so')
    seg_names = frames['segment'].to_numpy()
    seg_list = np.array(sorted(set(seg_names)))
    nseg = len(seg_list)
    seg_frames = {s: np.flatnonzero(seg_names == s) for s in seg_list}
    seg_code = np.searchsorted(seg_list, seg_names)
    u, g = np.unique(seg_code, return_inverse=True)       # as analyze_rules.py --oos (every frame used)

    # --- risk scores of the model
    fsc = pd.read_csv(os.path.join(a.risk, 'vlm_frame_scores.csv'), float_precision='round_trip')
    conds = [c for c in av.CONDITIONS if ((fsc.model == MODEL) & (fsc.condition == c)).any()]
    if conds != av.CONDITIONS:
        sys.exit(f'{MODEL}: conditions {conds} in vlm_frame_scores.csv, not {av.CONDITIONS}')
    risk = {}
    for c in conds:
        s = fsc[(fsc.model == MODEL) & (fsc.condition == c)].set_index(['segment', 'frame_idx'])['score_mean']
        if len(s) != n or not s.index.isin(mi).all():
            sys.exit(f'{MODEL} {c}: its frames in vlm_frame_scores.csv are not the {n} frames of ssm_target_frames.csv')
        risk[c] = s.reindex(mi).to_numpy(float)
        if np.isnan(risk[c]).any():
            sys.exit(f'{MODEL} {c}: {int(np.isnan(risk[c]).sum())} frames without a risk score')

    # --- cues, as analyze_rules.py
    tr = frames.merge(pd.read_csv(a.truth, float_precision='round_trip'), on=['segment', 'frame_idx'], how='left',
                      validate='one_to_one', indicator=True)
    if (tr['_merge'] != 'both').any():
        sys.exit(f'{int((tr["_merge"] != "both").sum())} frames are not in {a.truth}')
    cue = {'true_distance': tr['distance'].to_numpy(float), 'true_ego_speed': tr['v_ego'].to_numpy(float),
           'true_lead_speed': tr['v_lead'].to_numpy(float),
           'true_closing_speed': tr['v_ego'].to_numpy(float) - tr['v_lead'].to_numpy(float),
           'true_ego_accel': tr['a_ego'].to_numpy(float), 'true_lead_accel': tr['a_lead'].to_numpy(float)}
    nb = frames.merge(pd.read_csv(a.nearby), on=['segment', 'frame_idx'], how='left', validate='one_to_one',
                      indicator=True)
    if (nb['_merge'] != 'both').any():
        sys.exit(f'{int((nb["_merge"] != "both").sum())} frames are not in {a.nearby}')
    for r in ar.RADII:
        cue[f'ped_cyc_{r}m'] = nb[f'n_ped_cyc_{r}m'].to_numpy(float)
        cue[f'vehicle_{r}m'] = nb[f'n_vehicle_{r}m'].to_numpy(float)
    qf = pd.read_csv(os.path.join(a.quantities, 'quantity_frames.csv'), float_precision='round_trip')
    qg = qf[(qf.model == MODEL) & (qf.condition == OWN_CONDITION)]
    e = {q: qg[qg.quantity == q].set_index(['segment', 'frame_idx'])['estimate'].reindex(mi).to_numpy(float)
         for q in ('distance', 'ego_speed', 'lead_speed', 'ego_accel', 'lead_accel')}
    own = {'model_distance': e['distance'], 'model_ego_speed': e['ego_speed'], 'model_lead_speed': e['lead_speed'],
           'model_closing_speed': e['ego_speed'] - e['lead_speed'], 'model_ego_accel': e['ego_accel'],
           'model_lead_accel': e['lead_accel']}
    sl = pd.read_csv(a.scene_labels, dtype=str, keep_default_na=False).set_index('segment')
    missing = sorted(set(seg_list) - set(sl.index))
    if missing:
        sys.exit(f'{a.scene_labels}: no row for {len(missing)} segments, e.g. {missing[:3]}')
    s_terms, s_cols, s_info = [], [], {}
    for col in ('time_of_day', 'location', 'weather'):
        v = sl.loc[seg_names, col].str.strip().to_numpy()
        if (v == '').any():
            sys.exit(f'{a.scene_labels}: no {col} for some segments of these frames')
        seg_v = sl.loc[seg_list, col].str.strip()
        if col == 'weather':
            if not seg_v.isin(ar.WEATHER).all():
                sys.exit(f'{a.scene_labels}: weather not one of {ar.WEATHER}: {sorted(set(seg_v) - set(ar.WEATHER))}')
            cats = ar.WEATHER
            nseg_c = {k: int((seg_v == k).sum()) for k in cats}
            ref = max(cats, key=lambda k: nseg_c[k])
        else:
            cats = sorted(set(v))
            nfr = {k: int((v == k).sum()) for k in cats}
            ref = max(cats, key=lambda k: nfr[k])
        s_info[col] = dict(reference=ref, segments={k: int((seg_v == k).sum()) for k in cats},
                           frames={k: int((v == k).sum()) for k in cats})
        for k in cats:
            if k != ref:
                s_terms.append(f'{col}={k}')
                s_cols.append((v == k).astype(float))
    names = {'A': list(ar.TRUE_CUES), 'B': list(ar.TRUE_CUES) + s_terms + list(ar.NEAR_CUES), 'C': list(ar.OWN_CUES)}
    X = {'A': np.column_stack([cue[k] for k in ar.TRUE_CUES]),
         'B': np.column_stack([cue[k] for k in ar.TRUE_CUES] + s_cols + [cue[k] for k in ar.NEAR_CUES]),
         'C': np.column_stack([own[k] for k in ar.OWN_CUES])}
    for cs, x in X.items():
        if np.isnan(x).any():
            bad_cols = [names[cs][j] for j in range(x.shape[1]) if np.isnan(x[:, j]).any()]
            sys.exit(f'cues {cs}: values missing in {bad_cols}; this program needs every frame')

    # --- the resamples, checked before they are used (as analyze_rules.py)
    n_boot_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(a.risk, 'params.txt'))
    for what, mine, theirs in (('resamples', a.n_boot, n_boot_r), ('seed', a.seed, seed_r), ('frames', n, frames_r)):
        if mine != theirs:
            failed.append(f'{what}: {mine} here, {theirs} in the risk analysis\'s params.txt')
    if list(seg_list) != segs_r:
        failed.append(f'segment list: {nseg} here, {len(segs_r)} in the risk analysis\'s params.txt, not the same')
    rng = np.random.default_rng(a.seed)
    draws = [rng.choice(seg_list, nseg) for _ in range(a.n_boot)]
    h = hashlib.sha256()
    for d in draws:
        h.update(np.searchsorted(seg_list, d).astype(np.int32).tobytes())
    nb_q, seed_q, hash_q = ar.read_hash_quantities(os.path.join(a.quantities, 'params.txt'))
    if (nb_q, seed_q, hash_q) != (a.n_boot, a.seed, h.hexdigest()):
        failed.append(f'drawn segments: sha256 {h.hexdigest()} ({a.n_boot}, seed {a.seed}) here, {hash_q} ({nb_q}, '
                      f'seed {seed_q}) in the quantities analysis\'s params.txt')
    if failed:
        sys.exit('stopped before the resamples are used: ' + '; '.join(failed))
    counts = np.stack([np.bincount(np.searchsorted(seg_list, d), minlength=nseg) for d in draws])
    os.makedirs(a.out_dir, exist_ok=True)
    print(f'frames {n} in {nseg} segments; {MODEL} x {conds}; resamples as the risk and quantities analyses '
          f'({time.time() - t0:.0f} s)', flush=True)

    # --- the linear fits of the rule analysis rebuilt from these variables, against exploratory_oos_r2.csv
    oo = pd.read_csv(os.path.join(a.rules, 'exploratory_oos_r2.csv'), float_precision='round_trip')
    oo = oo[(oo.target == 'risk') & (oo.model == MODEL)].set_index(['condition', 'fit'])
    lin_dev, lin_n, lin_bad = 0.0, 0, []
    for c in conds:
        y = ar.rank01(risk[c])
        K = [ar.rank01(cue[k]) for k in ar.K_TRUE]
        designs = [('K', K)] + [(f'K+S+N{r}', K + s_cols + [ar.rank01(cue[f'ped_cyc_{r}m']),
                                                             ar.rank01(cue[f'vehicle_{r}m'])]) for r in ar.RADII]
        if c == OWN_CONDITION:
            designs.append(('K_own', [ar.rank01(own[k]) for k in ar.K_OWN]))
        for fname, cols in designs:
            lin_n += 1
            if (c, fname) not in oo.index:
                lin_bad.append(f'{c} {fname}: not in exploratory_oos_r2.csv')
                continue
            ref = oo.loc[(c, fname)]
            r2_in = ar.ols(y, cols)[2]
            r2o, lo, hi, nv = r2_oos(y, ar.loso_predict(y, cols, g), u, g, nseg, counts)
            dev = max(abs(r2_in - ref.r2_in), abs(r2o - ref.r2_oos), abs(lo - ref.r2_oos_lo), abs(hi - ref.r2_oos_hi))
            lin_dev = max(lin_dev, dev)
            if not (dev <= LIN_TOL and int(ref.nboot) == nv and int(ref.frames) == n and int(ref.segments) == nseg):
                lin_bad.append(f'{c} {fname}: largest difference {dev:.3g}, nboot {nv} vs {int(ref.nboot)}, frames '
                               f'{int(ref.frames)}, segments {int(ref.segments)}')
    checks.append(f'linear fits rebuilt here ({lin_n}: K and K+S+N10..N50 per condition, K_own on {OWN_CONDITION}) '
                  f'against {os.path.join(a.rules, "exploratory_oos_r2.csv")}: in-sample R2, out-of-sample R2 and '
                  f'interval at most {lin_dev:.3g} apart (tolerance {LIN_TOL:g}), nboot, frames and segments the same: '
                  f'{lin_n - len(lin_bad)} of {lin_n}')
    if lin_bad:
        failed.append('linear check: ' + '; '.join(lin_bad))
    print(f'linear check done ({time.time() - t0:.0f} s): {lin_n - len(lin_bad)} of {lin_n} the same', flush=True)

    # --- trees on all the frames, out of sample (risk: depths 1-5; SSMs: depth 3)
    node_rows, oos_rows, text = [], [], []
    node_bad, unpredicted, n_fits = 0, 0, 0

    def one_tree(target, c, s, cs, y, raw, depths):
        nonlocal node_bad, unpredicted, n_fits
        for d in depths:
            full = grow(X[cs], y, d)
            fit_in = full.predict(X[cs])
            pred = loso_tree(X[cs], y, g, d)
            n_fits += 1 + nseg
            unpredicted += int(np.isnan(pred).sum())
            r2o, lo, hi, nv = r2_oos(y, pred, u, g, nseg, counts)
            oos_rows.append(dict(target=target, model=MODEL if target == 'risk' else '', condition=c, ssm=s, cues=cs,
                                 max_depth=d, frames=n, segments=nseg, leaves=int(full.get_n_leaves()),
                                 r2_in=1.0 - float(((y - fit_in) ** 2).sum()) / float(((y - y.mean()) ** 2).sum()),
                                 r2_oos=r2o, r2_oos_lo=lo, r2_oos_hi=hi, nboot=nv))
            if d == MAX_DEPTH:
                rows, lines, bad = describe(full, X[cs], y, raw, names[cs], seg_code)
                node_bad += bad
                for r in rows:
                    node_rows.append(dict(target=target, model=MODEL if target == 'risk' else '', condition=c, ssm=s,
                                          cues=cs, **r))
                head = (f'{MODEL} {c}, cues {cs}' if target == 'risk' else f'SSM {s} ("larger = riskier"), cues {cs}')
                text.append(f'=== {head}: R2 in sample {oos_rows[-1]["r2_in"]:.3f}, out of sample {r2o:.3f} '
                            f'[{lo:.3f}, {hi:.3f}]; mean raw = {"risk score" if target == "risk" else s + " in its own units"}')
                text.extend(lines)

    for c in conds:
        y = ar.rank01(risk[c])
        for cs in ('A', 'B') + (('C',) if c == OWN_CONDITION else ()):
            one_tree('risk', c, '', cs, y, risk[c], DEPTHS)
        print(f'{c} done ({time.time() - t0:.0f} s)', flush=True)
    for s in ar.LEAD7:
        one_tree('ssm', '', s, 'A', ar.rank01(ssm_df[s + '_risk'].to_numpy(float)), ssm_df[s].to_numpy(float),
                 [MAX_DEPTH])
    print(f'SSM trees done ({time.time() - t0:.0f} s)', flush=True)
    checks.append(f'nodes whose frames or mean differ from scikit-learn\'s n_node_samples / value (within {NODE_TOL:g}): '
                  f'{node_bad} of {len(node_rows)}')
    checks.append(f'frames not predicted in the leave-one-segment-out runs: {unpredicted} ({n_fits} trees grown)')
    if node_bad:
        failed.append('nodes')
    if unpredicted:
        failed.append('unpredicted frames')

    # --- stability of the A tree over the resamples
    stab = []
    for c in conds:
        for b, draw in enumerate(draws):
            idx = np.concatenate([seg_frames[s] for s in draw])
            (f_, t_), (lf, lt), (rf, rt) = first_cuts(grow(X['A'][idx], ar.rank01(risk[c][idx])), names['A'])
            stab.append(dict(condition=c, resample=b, frames=len(idx), first_cue=f_, first_cut=t_, left_cue=lf,
                             left_cut=lt, right_cue=rf, right_cut=rt))
        print(f'stability {c} done ({time.time() - t0:.0f} s)', flush=True)
    stab = pd.DataFrame(stab)
    pt = {r['condition']: r for r in node_rows if r['target'] == 'risk' and r['cues'] == 'A' and r['node'] == 0}
    summ = []
    for c in conds:
        q = stab[stab.condition == c]
        cnt = q.first_cue.value_counts()
        top = sorted(cnt.index, key=lambda k: (-cnt[k], k))[0]
        cuts = q.first_cut[q.first_cue == top].to_numpy(float)
        p = np.percentile(cuts, [2.5, 50, 97.5])
        for k in sorted(cnt.index, key=lambda k: (-cnt[k], k)):
            summ.append(dict(condition=c, first_cue=k, resamples=int(cnt[k]), share=float(cnt[k]) / len(q),
                             most_frequent=bool(k == top), cut_p2_5=p[0] if k == top else np.nan,
                             cut_p50=p[1] if k == top else np.nan, cut_p97_5=p[2] if k == top else np.nan,
                             all_frames_first_cue=pt[c]['cue'], all_frames_first_cut=pt[c]['threshold']))

    # --- tree beside the linear fit
    vs = []
    od = pd.DataFrame(oos_rows)
    for c in conds:
        for cs, fits in LINEAR.items():
            if cs == 'C' and c != OWN_CONDITION:
                continue
            t_ = od[(od.target == 'risk') & (od.condition == c) & (od.cues == cs) & (od.max_depth == MAX_DEPTH)].iloc[0]
            for f in fits:
                ref = oo.loc[(c, f)]
                vs.append(dict(model=MODEL, condition=c, cues=cs, tree_r2_in=t_.r2_in, tree_r2_oos=t_.r2_oos,
                               tree_r2_oos_lo=t_.r2_oos_lo, tree_r2_oos_hi=t_.r2_oos_hi, linear_fit=f,
                               linear_r2_in=ref.r2_in, linear_r2_oos=ref.r2_oos, linear_r2_oos_lo=ref.r2_oos_lo,
                               linear_r2_oos_hi=ref.r2_oos_hi, nboot=int(ref.nboot)))

    def write(name, rows):
        pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, name), index=False)

    write('exploratory_tree_nodes.csv', node_rows)
    write('exploratory_tree_oos.csv', oos_rows)
    write('exploratory_tree_vs_linear.csv', vs)
    stab.to_csv(os.path.join(a.out_dir, 'exploratory_tree_stability.csv'), index=False)
    write('exploratory_tree_stability_summary.csv', summ)
    with open(os.path.join(a.out_dir, 'exploratory_tree_trees.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY decision trees (rule_tree.py, user 2026-11-04): DecisionTreeRegressor(max_depth={MAX_DEPTH}, '
                f'min_samples_leaf={MIN_LEAF}, random_state={TREE_SEED}); the dependent variable is the rank / frames '
                f'(the risk score of {MODEL}, or the SSM "larger = riskier"); cuts in the cues\' own units, frames with '
                f'a value <= the cut go left; {n} frames in {nseg} segments\n\n')
        f.write('\n'.join(text) + '\n')
    with open(os.path.join(a.out_dir, 'exploratory_tree_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY (user, 2026-11-04, decided before any tree was seen): rule_tree.py run '
                f'{time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        for k_, v_ in (('risk analysis', a.risk), ('quantities analysis', a.quantities), ('frame_ssm', a.frame_ssm),
                       ('truth', a.truth), ('scene labels', a.scene_labels), ('nearby counts', a.nearby),
                       ('rule analysis (exploratory_oos_r2.csv)', a.rules)):
            f.write(f'{k_} = {os.path.abspath(v_)}\n')
        f.write(f'versions: python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, scipy '
                f'{scipy.__version__}, scikit-learn {sklearn.__version__}\n')
        f.write(f'frames {n} in {nseg} segments; model {MODEL}, conditions {conds}; own estimates: {OWN_CONDITION}\n')
        f.write(f'tree: DecisionTreeRegressor(max_depth={MAX_DEPTH}, min_samples_leaf={MIN_LEAF}, '
                f'random_state={TREE_SEED}), the rest default: '
                f'{DecisionTreeRegressor(max_depth=MAX_DEPTH, min_samples_leaf=MIN_LEAF, random_state=TREE_SEED).get_params()}\n')
        f.write('dependent variable: rank among the frames / their number (scipy rankdata, ties the mean rank): the risk '
                'score (score_mean) of the model, or the SSM "larger = riskier" (raw values in the SSM\'s own units)\n')
        for cs in ('A', 'B', 'C'):
            f.write(f'cues {cs} (own units, not ranked): {names[cs]}\n')
        for col_, info in s_info.items():
            f.write(f'S: {col_}, reference {info["reference"]!r} (no column of its own); segments per category '
                    f'{info["segments"]}, frames {info["frames"]}\n')
        f.write('N (in B): pedestrians + cyclists and vehicles within 10, 20, 30, 50 m, main version (in the cameras\' '
                'view, occluded ones included), all eight in one tree; no H\n')
        f.write(f'out of sample: leave one segment out ({nseg} folds), as analyze_rules.py --oos; R2 = 1 - SSE / SST '
                f'about the mean over all frames; risk score max_depth {DEPTHS} (3 reported, the rest descriptive), SSMs '
                f'{MAX_DEPTH}; interval: the {a.n_boot} resamples (seed {a.seed}, sha256 of the drawn segment indices '
                f'{h.hexdigest()}), per resample the drawn segments\' SSE and SST added, nothing regrown; 2.5 / 97.5 % '
                'percentiles, none when fewer than 90 % are defined. These intervals leave out the variability of the '
                'regrowing\n')
        f.write(f'stability: the A tree regrown on each of the {a.n_boot} resamples (the drawn frames, a segment drawn '
                'twice counted twice, the ranks anew); share of the first cut per cue; on the most frequent cue the '
                '2.5 / 50 / 97.5 % percentiles of its cut (np.percentile)\n')
        f.write(f'beside the linear fits of exploratory_oos_r2.csv: {LINEAR}\n')
        for c_ in checks:
            f.write(f'check: {c_}\n')
        f.write(f'checks not met: {failed if failed else "none"}\n')
    print(f'done in {time.time() - t0:.0f} s; checks not met: {failed if failed else "none"}; outputs in '
          f'{os.path.abspath(a.out_dir)}')
    for c_ in checks:
        print('  ' + c_)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
