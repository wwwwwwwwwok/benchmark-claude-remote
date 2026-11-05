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

--resampled_trees OUT (EXPLORATORY, added after the trees were seen; user, 2026-11-04): instead of the outputs above
(none of them is written; out_dir is then the folder of an earlier run of this program, only read), the A and B trees
of every condition and the C tree of sep_camera regrown on each of the resamples (the drawn frames, a segment drawn
twice counted twice, the ranks taken anew among them), every cut recorded. OUT/exploratory_resampled_trees_*.csv:
cuts (every cut of every tree: its position, 'first' and then one letter per step down, L the <= side and R the > side;
cue, threshold, frames, the mean rank on either side), first (the share of resamples whose first cut is on each cue),
second (the same for the cut below the first on either side, the three most frequent ranked; 'leaf' when that side is
not cut), any (the share of resamples with a cut on the cue anywhere in the tree; for B also: any vehicle count, any
pedestrian + cyclist count, rain, time of day, location), direction (per cue and group, over every cut and over the first
cuts: the share of cuts whose > side has the higher mean rank); params.txt. Checks besides those above: the A trees'
first cut and the two below it equal the earlier run's exploratory_tree_stability.csv resample by resample, and the
shares of the first cut its exploratory_tree_stability_summary.csv.

--boosting OUT --ceiling FILE (EXPLORATORY, added after the trees and the ceiling were seen; user, 2026-11-05): instead
of the outputs above (none of them is written; out_dir is then the folder of an earlier run of this program, only
read), a more flexible model on the same cues and dependent variable, to tell whether little is reproduced because of
the form of the function or because of the cues: sklearn.ensemble.HistGradientBoostingRegressor, in two settings and
not tuned (BOOST): the default with random_state 0 and early_stopping False; and max_depth 3, learning_rate 0.05,
max_iter 200, min_samples_leaf 50, the rest as the first. Fitted on A and B for every condition and on C for sep_camera.
Out-of-sample R2 and interval as the trees above (leave one segment out; the resamples add the segments' sums, nothing
refitted); the in-sample R2 of the fit on all the frames beside. OUT/exploratory_boosting_oos.csv, and
exploratory_boosting_vs.csv: per condition and cues side by side the two settings, the depth-3 tree of the earlier run
(its exploratory_tree_oos.csv), the linear fits (A: K; B: K+S+N10..N50; C: K_own; <rules>/exploratory_oos_r2.csv) and
the ceiling of score_ceiling.py (FILE, its exploratory_ceiling.csv); exploratory_boosting_params.txt. OpenMP and BLAS
run single-threaded (unless the environment says otherwise), so the fits do not depend on the machine's thread count.

usage: python -I rule_tree.py <risk analysis subfolder> <quantities analysis subfolder> <frame_ssm.csv> <truth.csv>
       <scene_labels.csv> <nearby_counts.csv> <rule analysis subfolder> <out_dir> [--n_boot 2000] [--seed 20261021]
       [--resampled_trees OUT | --boosting OUT --ceiling FILE]
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
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
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
BOOST = {'default': dict(random_state=0, early_stopping=False),                        # --boosting (user, 2026-11-05)
         'conservative': dict(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50, random_state=0,
                              early_stopping=False)}


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


def all_cuts(tree, names):
    """Every cut of a fitted tree, top down: (position, depth, cue, threshold, frames, mean rank on the <= side, mean rank
    on the > side); position 'first' for the first cut, then one letter per step down (L the <= side, R the > side)."""
    t = tree.tree_
    out = []

    def walk(j, pos, depth):
        le, gt = t.children_left[j], t.children_right[j]
        if le == -1:
            return
        out.append((pos or 'first', depth, names[t.feature[j]], float(t.threshold[j]), int(t.n_node_samples[j]),
                    float(t.value[le].ravel()[0]), float(t.value[gt].ravel()[0])))
        walk(le, pos + 'L', depth + 1)
        walk(gt, pos + 'R', depth + 1)

    walk(0, '', 0)
    return out


def grow_boost(X, y, setting):
    return HistGradientBoostingRegressor(**BOOST[setting]).fit(X, y)


def loso_boost(X, y, g, setting):
    """Leave-one-segment-out predictions of the boosting model (BOOST[setting]), as loso_tree."""
    pred = np.full(len(y), np.nan)
    for k in range(int(g.max()) + 1):
        te = g == k
        pred[te] = grow_boost(X[~te], y[~te], setting).predict(X[te])
    return pred


def boosting(a, conds, X, names, risk, u, g, nseg, counts, failed, checks, t0, draws_sha):
    """--boosting (EXPLORATORY, added after the trees and the ceiling were seen; user, 2026-11-05): see the docstring."""
    import threadpoolctl
    out = a.boosting
    os.makedirs(out, exist_ok=True)
    n = len(g)
    cells = [(c, cs) for c in conds for cs in ('A', 'B')] + [(OWN_CONDITION, 'C')]
    rows, unpredicted = [], 0
    for c, cs in cells:
        y = ar.rank01(risk[c])
        sst = float(((y - y.mean()) ** 2).sum())
        for setting in BOOST:
            full = grow_boost(X[cs], y, setting)
            pred = loso_boost(X[cs], y, g, setting)
            unpredicted += int(np.isnan(pred).sum())
            r2o, lo, hi, nv = r2_oos(y, pred, u, g, nseg, counts)
            rows.append(dict(model=MODEL, condition=c, cues=cs, setting=setting, frames=n, segments=nseg,
                             iterations=int(full.n_iter_),
                             r2_in=1.0 - float(((y - full.predict(X[cs])) ** 2).sum()) / sst, r2_oos=r2o,
                             r2_oos_lo=lo, r2_oos_hi=hi, nboot=nv))
            print(f'boosting {c} {cs} {setting}: out of sample {r2o:.3f} [{lo:.3f}, {hi:.3f}] '
                  f'({time.time() - t0:.0f} s)', flush=True)
    checks.append(f'frames not predicted in the leave-one-segment-out runs of the boosting models: {unpredicted}')
    if unpredicted:
        failed.append('unpredicted frames (boosting)')
    to = pd.read_csv(os.path.join(a.out_dir, 'exploratory_tree_oos.csv'), float_precision='round_trip')
    to = to[(to.target == 'risk') & (to.model == MODEL) & (to.max_depth == MAX_DEPTH)]
    oo = pd.read_csv(os.path.join(a.rules, 'exploratory_oos_r2.csv'), float_precision='round_trip')
    oo = oo[(oo.target == 'risk') & (oo.model == MODEL)]
    ce = pd.read_csv(a.ceiling, float_precision='round_trip')
    ce = ce[ce.model == MODEL].set_index('condition')
    vs = []
    for c, cs in cells:
        for r in rows:
            if r['condition'] == c and r['cues'] == cs:
                vs.append(dict(condition=c, cues=cs, measure=f'boosting, {r["setting"]}', value=r['r2_oos'],
                               lo=r['r2_oos_lo'], hi=r['r2_oos_hi'], source='this run'))
        t = to[(to.condition == c) & (to.cues == cs)]
        if len(t) == 1:
            vs.append(dict(condition=c, cues=cs, measure=f'tree, depth {MAX_DEPTH}', value=t.r2_oos.iloc[0],
                           lo=t.r2_oos_lo.iloc[0], hi=t.r2_oos_hi.iloc[0], source='exploratory_tree_oos.csv'))
        else:
            failed.append(f'{c} {cs}: {len(t)} depth-{MAX_DEPTH} rows in exploratory_tree_oos.csv')
        for f_ in LINEAR[cs]:
            t = oo[(oo.condition == c) & (oo.fit == f_)]
            if len(t) == 1:
                vs.append(dict(condition=c, cues=cs, measure=f'{f_} (linear)', value=t.r2_oos.iloc[0],
                               lo=t.r2_oos_lo.iloc[0], hi=t.r2_oos_hi.iloc[0], source='exploratory_oos_r2.csv'))
            else:
                failed.append(f'{c} {f_}: {len(t)} rows in exploratory_oos_r2.csv')
        if c in ce.index:
            vs.append(dict(condition=c, cues=cs, measure='ceiling 5r/(1+4r)', value=ce.loc[c, 'ceiling'],
                           lo=ce.loc[c, 'ceiling_lo'], hi=ce.loc[c, 'ceiling_hi'], source=os.path.basename(a.ceiling)))
        else:
            failed.append(f'{c}: not in {a.ceiling}')
    pd.DataFrame(rows).to_csv(os.path.join(out, 'exploratory_boosting_oos.csv'), index=False)
    pd.DataFrame(vs).to_csv(os.path.join(out, 'exploratory_boosting_vs.csv'), index=False)
    pools = [dict(user_api=p.get('user_api'), internal_api=p.get('internal_api'), num_threads=p.get('num_threads'))
             for p in threadpoolctl.threadpool_info()]
    with open(os.path.join(out, 'exploratory_boosting_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY, added after the trees and the ceiling were seen (user, 2026-11-05): rule_tree.py --boosting '
                f'run {time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        for k_, v_ in (('risk analysis', a.risk), ('quantities analysis', a.quantities), ('frame_ssm', a.frame_ssm),
                       ('truth', a.truth), ('scene labels', a.scene_labels), ('nearby counts', a.nearby),
                       ('rule analysis (exploratory_oos_r2.csv)', a.rules), ('earlier run of rule_tree.py', a.out_dir),
                       ('ceiling', a.ceiling)):
            f.write(f'{k_} = {os.path.abspath(v_)}\n')
        f.write(f'versions: python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, scipy '
                f'{scipy.__version__}, scikit-learn {sklearn.__version__}; thread pools {pools}\n')
        for s_, kw in BOOST.items():
            f.write(f'boosting, {s_}: HistGradientBoostingRegressor({kw}); all parameters '
                    f'{HistGradientBoostingRegressor(**kw).get_params()}\n')
        for cs in ('A', 'B', 'C'):
            f.write(f'cues {cs} (own units, not ranked): {names[cs]}\n')
        f.write(f'out of sample: leave one segment out ({nseg} folds), as analyze_rules.py --oos; R2 = 1 - SSE / SST '
                f'about the mean over all frames; interval: the {a.n_boot} resamples (seed {a.seed}, sha256 of the '
                f'drawn segment indices {draws_sha}), per resample the drawn segments\' SSE and SST added, nothing '
                'refitted; 2.5 / 97.5 % percentiles, none when fewer than 90 % are defined\n')
        f.write(f'beside: the depth-{MAX_DEPTH} tree of the earlier run, the linear fits {LINEAR}, the ceiling\n')
        for c_ in checks:
            f.write(f'check: {c_}\n')
        f.write(f'checks not met: {failed if failed else "none"}\n')
    print(f'done in {time.time() - t0:.0f} s; checks not met: {failed if failed else "none"}; outputs in '
          f'{os.path.abspath(out)}')
    for c_ in checks:
        print('  ' + c_)
    sys.exit(1 if failed else 0)


def resampled_trees(a, conds, X, names, risk, draws, seg_frames, failed, checks, t0, draws_sha):
    """--resampled_trees (EXPLORATORY, added after the trees were seen; user, 2026-11-04): see the docstring."""
    out = a.resampled_trees
    os.makedirs(out, exist_ok=True)
    groups = {'any vehicle count': [k for k in names['B'] if k.startswith('vehicle_')],
              'any pedestrian + cyclist count': [k for k in names['B'] if k.startswith('ped_cyc_')],
              'rain': [k for k in names['B'] if k == 'weather=rain'],
              'time of day': [k for k in names['B'] if k.startswith('time_of_day=')],
              'location': [k for k in names['B'] if k.startswith('location=')]}
    n_b = len(draws)
    rows, first_a = [], []
    for c, cs in [(c, cs) for c in conds for cs in ('A', 'B')] + [(OWN_CONDITION, 'C')]:
        for b, draw in enumerate(draws):
            idx = np.concatenate([seg_frames[s] for s in draw])
            tree = grow(X[cs][idx], ar.rank01(risk[c][idx]))
            for pos, depth, cue_, thr, fr, m_le, m_gt in all_cuts(tree, names[cs]):
                rows.append(dict(condition=c, cues=cs, resample=b, frames=len(idx), position=pos, depth=depth, cue=cue_,
                                 threshold=thr, node_frames=fr, mean_rank_le=m_le, mean_rank_gt=m_gt,
                                 gt_higher=bool(m_gt > m_le)))
            if cs == 'A':
                (f_, t_), (lf, lt), (rf, rt) = first_cuts(tree, names['A'])
                first_a.append(dict(condition=c, resample=b, frames=len(idx), first_cue=f_, first_cut=t_, left_cue=lf,
                                    left_cut=lt, right_cue=rf, right_cut=rt))
        print(f'resampled trees {c} {cs} done ({time.time() - t0:.0f} s)', flush=True)
    cuts = pd.DataFrame(rows)

    def at(c, cs, pos):
        """The cue of the cut at pos in every resample: 'leaf' when that node is not cut, 'none' below a leaf."""
        q = cuts[(cuts.condition == c) & (cuts.cues == cs)]
        have = q[q.position == pos].set_index('resample').cue
        parent = pos[:-1] if len(pos) > 1 else 'first'
        up = set(q[q.position == parent]['resample']) if pos != 'first' else set(range(n_b))
        return pd.Series([have.get(b, 'leaf' if b in up else 'none') for b in range(n_b)])

    first, second, anyc, direc = [], [], [], []
    for c, cs in [(c, cs) for c in conds for cs in ('A', 'B')] + [(OWN_CONDITION, 'C')]:
        q = cuts[(cuts.condition == c) & (cuts.cues == cs)]
        f = at(c, cs, 'first').value_counts()
        for k in sorted(f.index, key=lambda k: (-f[k], k)):
            first.append(dict(condition=c, cues=cs, first_cue=k, resamples=int(f[k]), share=f[k] / n_b))
        for side, pos in (('<= side (L)', 'L'), ('> side (R)', 'R')):
            s = at(c, cs, pos).value_counts()
            for rank, k in enumerate(sorted(s.index, key=lambda k: (-s[k], k)), 1):
                second.append(dict(condition=c, cues=cs, side=side, cue=k, resamples=int(s[k]), share=s[k] / n_b,
                                   rank=rank, top3=rank <= 3))
        members = [(k, [k]) for k in names[cs]] + (list(groups.items()) if cs == 'B' else [])
        for k, ks in members:
            hit = q[q.cue.isin(ks)]
            anyc.append(dict(condition=c, cues=cs, cue_or_group=k, group=len(ks) > 1 or k in groups,
                             resamples_with_a_cut=int(hit['resample'].nunique()), share=hit['resample'].nunique() / n_b))
            for scope, h in (('every cut', hit), ('first cut', hit[hit.position == 'first'])):
                direc.append(dict(condition=c, cues=cs, cue_or_group=k, group=len(ks) > 1 or k in groups, scope=scope,
                                  cuts=len(h), gt_side_higher=int(h.gt_higher.sum()),
                                  share_gt_side_higher=float(h.gt_higher.mean()) if len(h) else np.nan))

    # --- checks against the earlier run (out_dir): the A trees' first cut and the two below it, the first-cut shares
    old = pd.read_csv(os.path.join(a.out_dir, 'exploratory_tree_stability.csv'), float_precision='round_trip',
                      keep_default_na=False, na_values=[''])
    new = pd.DataFrame(first_a)
    bad = 0
    if len(old) != len(new) or list(old.columns) != list(new.columns):
        bad = max(len(old), len(new))
    else:
        for col in new.columns:
            x, y = old[col].to_numpy(), new[col].to_numpy()
            if new[col].dtype.kind == 'f':
                bad += int(sum(1 for u_, v_ in zip(x, y) if not (u_ == v_ or (np.isnan(u_) and np.isnan(v_)))))
            else:
                bad += int(sum(1 for u_, v_ in zip(x, y) if str(u_) != str(v_)))
    checks.append(f'A trees: first cut and the two below it, resample by resample, against '
                  f'{os.path.join(a.out_dir, "exploratory_tree_stability.csv")} ({len(new)} rows): {bad} values differ')
    if bad:
        failed.append('A trees against exploratory_tree_stability.csv')
    sm = pd.read_csv(os.path.join(a.out_dir, 'exploratory_tree_stability_summary.csv'), float_precision='round_trip')
    fa = pd.DataFrame(first)
    fa = fa[fa.cues == 'A']
    j = sm[['condition', 'first_cue', 'resamples', 'share']].merge(
        fa[['condition', 'first_cue', 'resamples', 'share']], on=['condition', 'first_cue'], how='outer',
        suffixes=('_old', '_new'), indicator=True)
    bad_s = int((j['_merge'] != 'both').sum() + (j.resamples_old != j.resamples_new).sum()
                + (j.share_old != j.share_new).sum())
    checks.append(f'A trees: shares of the first cut against exploratory_tree_stability_summary.csv ({len(j)} rows): '
                  f'{bad_s} differ')
    if bad_s:
        failed.append('A first-cut shares against exploratory_tree_stability_summary.csv')

    cuts.to_csv(os.path.join(out, 'exploratory_resampled_trees_cuts.csv'), index=False)
    for name, rows_ in (('first', first), ('second', second), ('any', anyc), ('direction', direc)):
        pd.DataFrame(rows_).to_csv(os.path.join(out, f'exploratory_resampled_trees_{name}.csv'), index=False)
    with open(os.path.join(out, 'exploratory_resampled_trees_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY, added after the trees were seen (user, 2026-11-04): rule_tree.py --resampled_trees run '
                f'{time.strftime("%Y-%m-%d %H:%M:%S")}\n')
        for k_, v_ in (('risk analysis', a.risk), ('quantities analysis', a.quantities), ('frame_ssm', a.frame_ssm),
                       ('truth', a.truth), ('scene labels', a.scene_labels), ('nearby counts', a.nearby),
                       ('rule analysis (exploratory_oos_r2.csv)', a.rules), ('earlier run of rule_tree.py', a.out_dir)):
            f.write(f'{k_} = {os.path.abspath(v_)}\n')
        f.write(f'versions: python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, scipy '
                f'{scipy.__version__}, scikit-learn {sklearn.__version__}\n')
        f.write(f'trees regrown on each of the {n_b} resamples (seed {a.seed}, sha256 of the drawn segment indices '
                f'{draws_sha}; the drawn frames, a segment drawn twice counted twice, the ranks taken anew among them): '
                f'A and B for {conds}, C for {OWN_CONDITION}; DecisionTreeRegressor(max_depth={MAX_DEPTH}, '
                f'min_samples_leaf={MIN_LEAF}, random_state={TREE_SEED}), the rest default\n')
        for cs in ('A', 'B', 'C'):
            f.write(f'cues {cs} (own units, not ranked): {names[cs]}\n')
        f.write(f'groups of B cues: {groups}\n')
        f.write('position of a cut: first, then one letter per step down (L the <= side, R the > side); "leaf" when a '
                'node is not cut, "none" below a leaf; direction: the > side has the higher mean rank (gt_higher)\n')
        for c_ in checks:
            f.write(f'check: {c_}\n')
        f.write(f'checks not met: {failed if failed else "none"}\n')
    print(f'done in {time.time() - t0:.0f} s; checks not met: {failed if failed else "none"}; outputs in '
          f'{os.path.abspath(out)}')
    for c_ in checks:
        print('  ' + c_)
    sys.exit(1 if failed else 0)


def main():
    ap = argparse.ArgumentParser()
    for name in ('risk', 'quantities', 'frame_ssm', 'truth', 'scene_labels', 'nearby', 'rules', 'out_dir'):
        ap.add_argument(name)
    ap.add_argument('--n_boot', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=20261021)
    ap.add_argument('--resampled_trees', metavar='OUT', default=None,
                    help='EXPLORATORY (added after the trees were seen): every cut of the A, B, C trees regrown on the '
                         'resamples, into OUT; out_dir is then an earlier run of this program, only read')
    ap.add_argument('--boosting', metavar='OUT', default=None,
                    help='EXPLORATORY (added after the trees and the ceiling were seen): HistGradientBoostingRegressor '
                         'on the A, B, C cues, out of sample, into OUT; out_dir is then an earlier run, only read')
    ap.add_argument('--ceiling', metavar='FILE', default=None,
                    help='with --boosting: exploratory_ceiling.csv of score_ceiling.py, for the side-by-side table')
    a = ap.parse_args()
    if a.resampled_trees and a.boosting:
        sys.exit('--resampled_trees and --boosting: one at a time')
    if a.boosting:
        if not (a.ceiling and os.path.exists(a.ceiling)):
            sys.exit('--boosting needs --ceiling <exploratory_ceiling.csv of score_ceiling.py>')
        if not os.path.exists(os.path.join(a.out_dir, 'exploratory_tree_oos.csv')):
            sys.exit(f'--boosting: {a.out_dir} (an earlier run of this program) has no exploratory_tree_oos.csv')
        if os.path.exists(os.path.join(a.boosting, 'exploratory_boosting_params.txt')):
            sys.exit(f'{a.boosting} has exploratory_boosting_params.txt already: give a new folder')
    elif a.resampled_trees:
        for f_ in ('exploratory_tree_stability.csv', 'exploratory_tree_stability_summary.csv'):
            if not os.path.exists(os.path.join(a.out_dir, f_)):
                sys.exit(f'--resampled_trees: {a.out_dir} (an earlier run of this program) has no {f_}')
        if os.path.exists(os.path.join(a.resampled_trees, 'exploratory_resampled_trees_params.txt')):
            sys.exit(f'{a.resampled_trees} has exploratory_resampled_trees_params.txt already: give a new folder')
    elif os.path.exists(os.path.join(a.out_dir, 'exploratory_tree_params.txt')):
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
    if a.resampled_trees:                 # EXPLORATORY, added after the trees were seen (user, 2026-11-04)
        resampled_trees(a, conds, X, names, risk, draws, seg_frames, failed, checks, t0, h.hexdigest())
        return
    if a.boosting:                        # EXPLORATORY, added after the trees and the ceiling were seen (2026-11-05)
        boosting(a, conds, X, names, risk, u, g, nseg, counts, failed, checks, t0, h.hexdigest())
        return

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
