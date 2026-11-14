"""EXPLORATORY (user, 2026-11-14; the plan is in CLAUDE.md "理由分析" -> "逐条答案", recorded as given and committed
before this program was written; it is not changed after the results): how Qwen3.6-27B scores, answer by answer.
sep_camera, the main result's 1,084 frames (238 segments), 5,420 answers (26 forced). No SSM is used. The unit is the
single answer: its reason's theme paired with that same answer's own score. CPU only; no image is opened; only the result
tables of the earlier rounds are read (no reply text).
Inputs (not recomputed):
  clusters: the k = 7 labels of the answer-level clustering (reason_27b/main_1084/exploratory_reason_answers.csv,
    column cluster; run 31765529, recorded in 6add85e); not re-clustered.
  keyword classes: round 3's answer table (reason_27b/main_1084_keywords/exploratory_keywords_answers.csv, columns
    k01..k20, round 3's matching and negation rules as that program applied them); an answer mentions a class when its
    status is "mentioned" (round 3's answer-level reading); "negated" and "none" are not mentioned.
  each answer's own score: the risk column of those tables (round 3 checked it = the risk analysis's tidy risk).
Step 1 (per cluster, per keyword class): mean of the own scores of the answers in it minus the mean of all 5,420 answers'
  scores; answers, frames, segments.
Step 2 (within frame): the frames whose 5 answers are split (at least 1 in the item, at least 1 not); per frame, the mean
  score of its answers in the item minus the mean of the others; the average over those frames, each frame weight 1;
  frames, segments, and how many of those frames have the same score in all 5 answers. Also the frames whose 5 answers
  are all in one cluster.
95% intervals: 2,000 resamples of the 238 segments with replacement (default_rng(20261021), one choice over the sorted
  segments per resample, drawn segments' sha256 48bc1f41..., the main analysis's resamples); every answer / split frame
  of a drawn segment, as often as the segment is drawn; the items and the split frames are not redone; 2.5 / 97.5 %
  percentiles (numpy, linear) over the resamples where the value is defined (the item has an answer / a split frame),
  the others counted as missing. No minimum size is applied (the plan has none).
Checks (exit 1 when one is not met): 5,420 answers, 1,084 frames, 238 segments, 5 per frame; the two tables have the same
  answers (their row orders differ: the keyword table is matched to the cluster table by segment, frame_idx, repeat) with the same risk and forced; every answer has a cluster 0..6 and a status for every class;
  frame means = vlm_frame_scores.csv score_mean (qwen3.6-27b, sep_camera) exactly; per cluster answers, frames, segments,
  mean - all and its interval = exploratory_reason_clusters.csv; per class the answers mentioned =
  exploratory_keywords_categories.csv; frames with all 5 in one cluster = the clustering's params line. The run script
  compares a second run byte for byte.
Outputs (out_dir): exploratory_answer_themes.csv (step 1), _within.csv (step 2), _within_frames.csv (step 2, one row per
  item and split frame), _check.csv, _params.txt.
usage: python -I reason_answers.py <waymo_v143 folder> <out_dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import hashlib  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402

MODEL, COND = 'qwen3.6-27b', 'sep_camera'
N_BOOT, BOOT_SEED, K = 2000, 20261021, 7
HASH = '48bc1f413f9d5b99da57ff4edde7bbb4eb378c5a9343c44ccad4e0165466604c'
failed, rows_ck = [], []


def check(ok, what):
    rows_ck.append(dict(check=what, ok=bool(ok)))
    print(('  check ok: ' if ok else '  CHECK NOT MET: ') + what)
    if not ok:
        failed.append(what)


def pct(v):
    ok = ~np.isnan(v)
    if not ok.any():
        return np.nan, np.nan, 0, int((~ok).sum())
    lo, hi = np.percentile(v[ok], [2.5, 97.5])
    return float(lo), float(hi), int(ok.sum()), int((~ok).sum())


def side(lo, hi):
    return 'no interval' if np.isnan(lo) else 'above 0' if lo > 0 else 'below 0' if hi < 0 else 'contains 0'


def main():
    t0 = time.time()
    L, O = sys.argv[1], sys.argv[2]
    os.makedirs(O, exist_ok=True)
    A = os.path.join(L, 'analysis', 'after_gemma-4-31b-it', 'main_1084')
    R1 = os.path.join(L, 'analysis', 'reason_27b', 'main_1084')
    R3 = os.path.join(L, 'analysis', 'reason_27b', 'main_1084_keywords')
    p_cl, p_kw = os.path.join(R1, 'exploratory_reason_answers.csv'), os.path.join(R3, 'exploratory_keywords_answers.csv')
    prog_sha = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()
    print(f'EXPLORATORY: answer-level scores by theme, {MODEL} {COND}; program sha256 {prog_sha}')

    ca = pd.read_csv(p_cl, keep_default_na=False, na_values=[''], float_precision='round_trip')
    kw = pd.read_csv(p_kw, keep_default_na=False, na_values=[''], float_precision='round_trip')
    print('\n1. inputs')
    key = ['segment', 'frame_idx', 'repeat']
    check(len(ca) == 5420 and len(kw) == 5420, f'5,420 answers in both tables ({len(ca)}, {len(kw)})')
    ki = pd.MultiIndex.from_frame(ca[key])
    kk = pd.MultiIndex.from_frame(kw[key])
    check(ki.is_unique and kk.is_unique and set(ki) == set(kk),
          'the same answers in both tables, each once (segment, frame_idx, repeat; the row orders differ, so the keyword '
          'table is put in the cluster table\'s order by these keys)')
    if failed:
        sys.exit('input checks not met; nothing written')
    kw = kw.set_index(key).reindex(ki).reset_index()
    check(ca[key].equals(kw[key]) and np.array_equal(ca.risk.to_numpy(float), kw.risk.to_numpy(float)) and
          ca.forced.equals(kw.forced), 'after matching by keys: the same risk and forced in both tables, answer by answer')
    check(not ca.risk.isna().any(), 'every answer has a score')
    cl = ca.cluster.to_numpy()
    check(not ca.cluster.isna().any() and set(cl.tolist()) <= set(range(K)), f'every answer has a cluster 0..{K - 1}')
    cl = cl.astype(int)
    kcols = [f'k{c:02d}' for c in range(1, 21)]
    st = kw[kcols]
    check(st.isin(['mentioned', 'negated', 'none']).all().all(), 'every answer has a status for every class')
    risk = ca.risk.to_numpy(float)
    seg = ca.segment.to_numpy()
    fkey = ca.segment + '|' + ca.frame_idx.astype(str)
    fcodes, funiq = pd.factorize(fkey, sort=True)
    nper = np.bincount(fcodes)
    seg_list = np.array(sorted(set(seg)))
    check(len(funiq) == 1084 and len(seg_list) == 238 and (nper == 5).all(),
          f'{len(funiq)} frames, {len(seg_list)} segments, 5 answers per frame')
    fsm = ca.groupby(['segment', 'frame_idx'], sort=True)['risk'].mean()
    p_fs = os.path.join(A, 'vlm_frame_scores.csv')
    fs = pd.read_csv(p_fs, float_precision='round_trip')
    fs = fs[(fs.model == MODEL) & (fs.condition == COND)].set_index(['segment', 'frame_idx'])['score_mean'].reindex(fsm.index)
    check(not fs.isna().any() and np.array_equal(fsm.to_numpy(), fs.to_numpy()),
          f'frame means of the answers\' scores = {p_fs} score_mean on 1,084 frames, exactly')
    overall = float(risk.mean())
    if failed:
        sys.exit('input checks not met; nothing written')

    items =[('cluster', str(c), f'cluster {c}', cl == c) for c in range(K)]
    cats = pd.read_csv(os.path.join(R3, 'exploratory_keywords_categories.csv'))
    names = dict(zip(cats.category, cats.name))
    items += [('keyword', str(c), names[c], (st[f'k{c:02d}'] == 'mentioned').to_numpy()) for c in range(1, 21)]

    # frames: answer rows per frame (sorted by frame code), for step 2
    order = np.argsort(fcodes, kind='stable')
    frow = order.reshape(-1, 5)                                   # 1,084 x 5 answer rows
    fr_seg = seg[frow[:, 0]]
    same5 = (risk[frow] == risk[frow][:, :1]).all(1)
    one_cl = (cl[frow] == cl[frow][:, :1]).all(1)
    p_cpar = os.path.join(R1, 'exploratory_reason_params.txt')
    mline = re.search(r'frames with all 5 answers in one cluster: (\d+) of (\d+)', open(p_cpar, encoding='utf-8').read())
    check(mline and int(mline.group(1)) == int(one_cl.sum()) and int(mline.group(2)) == 1084,
          f'frames with all 5 answers in one cluster: {int(one_cl.sum())} = {p_cpar} ({mline.group(1) if mline else None})')

    # per item: split frames and their differences
    split = {}
    for kind, it, name, m in items:
        mf = m[frow]
        nin = mf.sum(1)
        sp = (nin >= 1) & (nin <= 4)
        rf = risk[frow]
        d = np.full(len(frow), np.nan)
        d[sp] = (np.where(mf, rf, 0).sum(1)[sp] / nin[sp]) - (np.where(~mf, rf, 0).sum(1)[sp] / (5 - nin[sp]))
        split[(kind, it)] = (sp, d, nin)

    # resamples
    nb_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(A, 'params.txt'))
    check((nb_r, seed_r, frames_r) == (N_BOOT, BOOT_SEED, 1084) and list(seg_list) == segs_r,
          f'{A}/params.txt: {nb_r} resamples, seed {seed_r}, {frames_r} frames, the same {len(seg_list)} segments')
    rng = np.random.default_rng(BOOT_SEED)
    draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
    hh = hashlib.sha256()
    for dr in draws:
        hh.update(np.searchsorted(seg_list, dr).astype(np.int32).tobytes())
    check(hh.hexdigest() == HASH, f"drawn segments' sha256 {hh.hexdigest()}")
    seg_rows = {s: np.flatnonzero(seg == s) for s in seg_list}
    seg_frames = {s: np.flatnonzero(fr_seg == s) for s in seg_list}
    NI = len(items)
    b1 = np.full((N_BOOT, NI), np.nan)
    b2 = np.full((N_BOOT, NI), np.nan)
    masks = np.stack([m for *_, m in items], axis=1)
    for b, dr in enumerate(draws):
        idx = np.concatenate([seg_rows[s] for s in dr])
        fidx = np.concatenate([seg_frames[s] for s in dr])
        r_ = risk[idx]
        ov = np.nanmean(r_)
        for j, (kind, it, name, m) in enumerate(items):
            sel = masks[idx, j]
            if sel.any():
                b1[b, j] = r_[sel].mean() - ov
            sp, d, _ = split[(kind, it)]
            fs_ = fidx[sp[fidx]]
            if len(fs_):
                b2[b, j] = d[fs_].mean()

    rows1, rows2, rowsf = [], [], []
    for j, (kind, it, name, m) in enumerate(items):
        lo, hi, nb, miss = pct(b1[:, j])
        rows1.append(dict(kind=kind, item=it, name=name, answers=int(m.sum()),
                          frames=int(np.unique(fcodes[m]).size), segments=int(np.unique(seg[m]).size),
                          mean_in=float(risk[m].mean()) if m.any() else np.nan, mean_all=overall,
                          diff=float(risk[m].mean() - overall) if m.any() else np.nan, lo95=lo, hi95=hi, nboot=nb,
                          missing=miss, side=side(lo, hi)))
        sp, d, nin = split[(kind, it)]
        lo, hi, nb, miss = pct(b2[:, j])
        rows2.append(dict(kind=kind, item=it, name=name, split_frames=int(sp.sum()),
                          segments=int(np.unique(fr_seg[sp]).size), mean_diff=float(d[sp].mean()) if sp.any() else np.nan,
                          lo95=lo, hi95=hi, nboot=nb, missing=miss, side=side(lo, hi),
                          split_frames_same_score_all_5=int((sp & same5).sum()),
                          frames_all_5_in=int((nin == 5).sum()), frames_none_in=int((nin == 0).sum())))
        for f in np.flatnonzero(sp):
            mf = m[frow[f]]
            rf = risk[frow[f]]
            seg_f, fi = funiq[f].split('|')
            rowsf.append(dict(kind=kind, item=it, segment=seg_f, frame_idx=int(fi), answers_in=int(nin[f]),
                              mean_in=float(rf[mf].mean()), mean_out=float(rf[~mf].mean()), diff=float(d[f]),
                              same_score_all_5=bool(same5[f])))

    print('\n2. against the earlier tables')
    old = pd.read_csv(os.path.join(R1, 'exploratory_reason_clusters.csv'), float_precision='round_trip')
    for c in range(K):
        o, n = old.iloc[c], rows1[c]
        check(int(o.cluster) == c and (int(o.answers), int(o.frames), int(o.segments)) ==
              (n['answers'], n['frames'], n['segments']) and float(o.mean_minus_all) == n['diff'] and
              float(o.diff_lo) == n['lo95'] and float(o.diff_hi) == n['hi95'],
              f'cluster {c}: answers, frames, segments, mean - all and its interval = exploratory_reason_clusters.csv '
              f'line {c + 2}')
    check(all(int(cats.answers_mentioned[cats.category == c].iloc[0]) == rows1[K + c - 1]['answers'] for c in range(1, 21)),
          'keyword classes: answers mentioned = exploratory_keywords_categories.csv lines 2-21')

    pd.DataFrame(rows1).to_csv(os.path.join(O, 'exploratory_answer_themes.csv'), index=False)
    pd.DataFrame(rows2).to_csv(os.path.join(O, 'exploratory_answer_within.csv'), index=False)
    pd.DataFrame(rowsf).to_csv(os.path.join(O, 'exploratory_answer_within_frames.csv'), index=False)
    pd.DataFrame(rows_ck).to_csv(os.path.join(O, 'exploratory_answer_check.csv'), index=False)
    with open(os.path.join(O, 'exploratory_answer_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY: answer-level scores by theme, {MODEL} {COND}, main result 1,084 frames (user, 2026-11-14; '
                f'plan in CLAUDE.md "理由分析" -> "逐条答案")\nrun {time.strftime("%Y-%m-%d %H:%M:%S")}; out_dir '
                f'{os.path.abspath(O)}\n')
        f.write(f'program {os.path.abspath(__file__)} sha256 {prog_sha}; numpy {np.__version__}, pandas {pd.__version__}\n')
        f.write(f'inputs: {p_cl} (cluster, risk); {p_kw} (k01..k20, risk); {p_fs}; '
                f'{os.path.join(R1, "exploratory_reason_clusters.csv")}; {os.path.join(R3, "exploratory_keywords_categories.csv")}\n')
        f.write('keyword class mentioned by an answer: status "mentioned" (negated and none are not)\n')
        f.write(f'all answers\' mean score {overall!r}; frames with all 5 answers in one cluster {int(one_cl.sum())} of 1084; '
                f'frames with the same score in all 5 answers {int(same5.sum())} of 1084\n')
        f.write(f'95% intervals: {N_BOOT} resamples of whole segments, default_rng({BOOT_SEED}), drawn segments\' sha256 '
                f'{hh.hexdigest()}; percentiles 2.5 / 97.5 (numpy linear) over the resamples where the value is defined; '
                f'no minimum size\n')

    print('\n' + '=' * 110)
    print(f'all answers\' mean score {overall!r}; frames all 5 in one cluster {int(one_cl.sum())}; frames with one score in '
          f'all 5 answers {int(same5.sum())}')
    print('\nstep 1 (exploratory_answer_themes.csv)')
    for i, d in enumerate(rows1):
        print(f"  line {i + 2}: {d['kind']} {d['item']} {d['name']}: {d['answers']} answers, {d['frames']} frames, "
              f"{d['segments']} segments; mean {d['mean_in']!r}; minus all {d['diff']!r} 95% interval [{d['lo95']!r}, "
              f"{d['hi95']!r}] ({d['side']}; missing {d['missing']})")
    print('\nstep 2 (exploratory_answer_within.csv)')
    for i, d in enumerate(rows2):
        print(f"  line {i + 2}: {d['kind']} {d['item']} {d['name']}: {d['split_frames']} split frames, {d['segments']} "
              f"segments; mean difference {d['mean_diff']!r} 95% interval [{d['lo95']!r}, {d['hi95']!r}] ({d['side']}; "
              f"missing {d['missing']}); same score in all 5: {d['split_frames_same_score_all_5']}")
    for name, rr in (('step 1', rows1), ('step 2', rows2)):
        for kind in ('cluster', 'keyword'):
            s = [d['side'] for d in rr if d['kind'] == kind]
            n = sum(x != 'no interval' for x in s)
            out = sum(x in ('above 0', 'below 0') for x in s)
            print(f'{name} {kind}: {n} intervals, {out} leave out 0, x 0.05 = {n * 0.05!r}')
    print(f'\ndone in {time.time() - t0:.0f} s; checks not met: {len(failed)}')
    for w in failed:
        print('  ', w)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main()
