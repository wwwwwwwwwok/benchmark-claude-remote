"""EXPLORATORY, POST-HOC (user, 2026-11-14, decided after the round-3 results were seen; plan in CLAUDE.md "理由分析" ->
"第三轮" -> "第三轮补充"): round 3's part C (5.3) again with "estimated" limited to distance and speed units. The time
units (s, sec, second(s)) are dropped as a whole: in the thinking they cannot be told apart from the image time labels
("Front-left 2.0s"). Round 3's own C outputs are not changed; this program writes only exploratory_thinking_posthoc_*.
Data as round 3 (reason_keywords.py): Qwen3.6-27B, sep_camera, the main result's 1,084 frames (238 segments), 5 answers
each (5,420); the thinking = the reply's `reasoning` in vlm_scores/qwen3.6-27b/main_qwen3.6-27b.jsonl.gz (a forced
answer's thinking is the reply's own reasoning, cut at max_tokens: checked against the forced record). Matches: round
3's regex (reason_keywords.EST_RE, unchanged) and its kinds; only the distance (m, meter(s), metre(s), ft, feet) and
speed (mph, km/h, kph, m/s) matches are kept. An answer has an estimate when its thinking has one; a frame "estimated"
when 3+ of its 5 answers do (main), 1+ (sensitivity). Per group (as 5.3): frames, segments, mean score (score_mean of the
risk analysis) with its interval, the 9 SSM risk percentile medians (round 2's frame table) with intervals, tau-b over
the frame pairs inside the group (S / sqrt(nx ny)) and the difference estimated - not estimated; no tau-b interval for a
group under 50 frames (a difference too when either group is). Intervals: the main analysis's 2,000 segment resamples
(default_rng(20261021), sha256 48bc1f41...), 2.5 / 97.5 % (numpy linear) over the resamples where defined, the others
counted as missing. Comparison counts (round 3's section 7): intervals read against 0 (or 50), how many leave it out,
count x 0.05. Sample: 20 distance and 20 speed matches (default_rng(20261114), distance first), 15 words either side,
the match in [[ ]], a last column for the user (1 / 0); reading rule: if more than half are not real estimates, numbers
only, no conclusion.
Checks (exit 1 when one is not met): 5,420 answers, 1,084 frames, 238 segments, 5 per frame; frame means = score_mean;
every frame has every SSM; the percentiles = round 2's and a recomputation; the 9 tau-b over all pairs = agreement.csv
lines 1001-1025 within 1e-12; the forced answers' thinking = the forced record's; each answer's distance and speed flags
= round 3's exploratory_keywords_answers.csv; the distance and speed unit counts = round 3's
exploratory_thinking_estimate_units.csv. The run script compares a second run byte for byte.
usage: python -I reason_thinking_posthoc.py <waymo_v143 folder> <round-2 out_dir> <round-3 out_dir> <out_dir>
       [--workers N]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import argparse  # noqa: E402
import collections  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_rules as ar  # noqa: E402
import analyze_vlm_ssm as av  # noqa: E402
import reason_clusters as rc  # noqa: E402
import reason_keywords as rk  # noqa: E402

MODEL, COND = rk.MODEL, rk.COND
N_BOOT, BOOT_SEED, HASH = rk.N_BOOT, rk.BOOT_SEED, rk.HASH
AG_FIRST, AG_STEP, TOL = rk.AG_FIRST, rk.AG_STEP, rk.TOL
SEED, N_SAMPLE, MIN_TAU = 20261114, 20, 50
KINDS = ['distance', 'speed']
PFX = 'exploratory_thinking_posthoc_'
G = {}                                                            # data for the worker processes (fork)


def stats(idx):
    sc = G['score'][idx]
    res = {}
    for name, arr in G['two'].items():
        m = arr[idx]
        for sd_, mm in ((1, m), (0, ~m)):
            res[('mean', name, sd_)] = float(sc[mm].mean()) if mm.any() else np.nan
            for s in av.SSMS:
                res[('pct', name, sd_, s)] = rk.med(G['pct'][s][idx], mm)
    p, q = np.triu_indices(len(idx), 1)
    sx = np.sign(sc[p] - sc[q])
    sy = {s: np.sign(G['risk'][s][idx][p] - G['risk'][s][idx][q]) for s in av.SSMS}
    for name, lab_arr, K in G['groupings']:
        lab = lab_arr[idx]
        code = np.where(lab[p] == lab[q], lab[p], K).astype(np.int64)
        nx = np.bincount(code, weights=np.abs(sx), minlength=K + 1)
        for s in av.SSMS:
            S = np.bincount(code, weights=sx * sy[s], minlength=K + 1)
            ny = np.bincount(code, weights=np.abs(sy[s]), minlength=K + 1)
            for k in range(K):
                res[('tau', name, k, s)] = rk.tau_of(S[k], nx[k], ny[k])
            if name == 'all':
                res[('tau', name, 'all', s)] = rk.tau_of(S.sum(), nx.sum(), ny.sum())
            if K == 2:
                res[('tau', name, 'diff', s)] = res[('tau', name, 1, s)] - res[('tau', name, 0, s)]
    return res


def one(b):
    idx = np.concatenate([G['seg_frames'][s] for s in G['draws'][b]])
    rr = stats(idx)
    return np.array([rr.get(k, np.nan) for k in G['keys']], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('waymo')
    ap.add_argument('round2')
    ap.add_argument('round3')
    ap.add_argument('out')
    ap.add_argument('--workers', type=int, default=1)
    a = ap.parse_args()
    L, R2, R3, O = a.waymo, a.round2, a.round3, a.out
    A, MD, FSSM = rc.paths(L)
    if os.path.exists(os.path.join(O, PFX + 'params.txt')):
        sys.exit(f'{O}/{PFX}params.txt exists already: give a new folder')
    os.makedirs(O, exist_ok=True)
    t0 = time.time()
    failed, rows_ck = [], []

    def check(ok, what):
        print(f'  check {"ok" if ok else "NOT MET"}: {what}', flush=True)
        rows_ck.append(dict(check=what, ok=bool(ok)))
        if not ok:
            failed.append(what)

    with open(os.path.abspath(__file__), 'rb') as f:
        prog_sha = hashlib.sha256(f.read()).hexdigest()
    with open(rk.__file__, 'rb') as f:
        rk_sha = hashlib.sha256(f.read()).hexdigest()
    print(f'round 3, post-hoc C (thinking estimates: distance and speed units only), {MODEL} {COND} (user, 2026-11-14; '
          f'EXPLORATORY, POST-HOC); run {time.strftime("%Y-%m-%d %H:%M:%S")}; program sha256 {prog_sha}; '
          f'reason_keywords.py sha256 {rk_sha}\n\n0. inputs', flush=True)

    # ------------------------------------------------------------------------------------------------- answers, thinking
    td = pd.read_csv(os.path.join(A, 'vlm_requests_tidy.csv'), low_memory=False, keep_default_na=False, na_values=[''],
                     usecols=['model', 'condition', 'segment', 'frame_idx', 'repeat', 'risk', 'forced'])
    td = td[(td.model == MODEL) & (td.condition == COND)].sort_values(['segment', 'frame_idx', 'repeat']) \
        .reset_index(drop=True)
    td['frame_idx'], td['repeat'] = td.frame_idx.astype(int), td['repeat'].astype(int)
    td['forced'] = td.forced.astype(str).str.lower().eq('true')
    keys = {(s, k, r): i for i, (s, k, r) in enumerate(zip(td.segment, td.frame_idx, td['repeat']))}
    main_f = av.jsonl_file(MD, f'main_{MODEL}')
    forced_f = av.jsonl_file(MD, f'main_{MODEL}_forced_answers')
    think = [None] * len(td)
    seen = [False] * len(td)
    with av.open_jsonl(main_f) as f:
        for raw in f:
            d = av.load_light(raw)
            if d['condition'] != COND:
                continue
            i = keys.get((d['segment'], int(d['frame_idx']), int(d['repeat'])))
            if i is None:
                continue
            if seen[i]:
                sys.exit(f'duplicate request {d["segment"]} {d["frame_idx"]} {d["repeat"]}')
            seen[i] = True
            think[i] = d.get('reasoning')
    n_forced = same = 0
    with av.open_jsonl(forced_f) as f:
        for raw in f:
            d = av.load_light(raw)
            if d['condition'] != COND:
                continue
            i = keys.get((d['segment'], int(d['frame_idx']), int(d['repeat'])))
            if i is None or not d.get('parsed'):
                continue
            n_forced += 1
            same += d.get('reasoning') == think[i]
    check(all(seen) and n_forced == int(td.forced.sum()) == same,
          f'{len(td)} answers\' thinking read from {main_f} (field reasoning); the {n_forced} forced answers\' thinking = '
          f'the forced record\'s in {forced_f} ({same} of {n_forced})')
    frames = pd.read_csv(os.path.join(A, 'ssm_target_frames.csv'))[['segment', 'frame_idx']] \
        .sort_values(['segment', 'frame_idx']).reset_index(drop=True)
    n = len(frames)
    mi = pd.MultiIndex.from_frame(frames)
    fpos = pd.Series(np.arange(n), index=mi).reindex(pd.MultiIndex.from_arrays([td.segment, td.frame_idx])).to_numpy()
    reps = collections.defaultdict(dict)
    for i, (fp, rp) in enumerate(zip(fpos, td['repeat'])):
        if not np.isnan(fp):
            reps[int(fp)][int(rp)] = i
    check(len(td) == 5420 and n == 1084 and frames.segment.nunique() == 238 and not np.isnan(fpos.astype(float)).any()
          and all(sorted(reps[f_]) == [0, 1, 2, 3, 4] for f_ in range(n)),
          f'{len(td)} answers, {n} frames, {frames.segment.nunique()} segments, 5 answers per frame (repeats 0-4)')
    rows5 = np.array([[reps[f_].get(r, 0) for r in range(5)] for f_ in range(n)])
    p_fs = os.path.join(A, 'vlm_frame_scores.csv')
    fs = pd.read_csv(p_fs, float_precision='round_trip')
    score = fs[(fs.model == MODEL) & (fs.condition == COND)].set_index(['segment', 'frame_idx'])['score_mean'] \
        .reindex(mi).to_numpy(float)
    fm = td.groupby(['segment', 'frame_idx'], sort=True)['risk'].mean().reindex(mi).to_numpy(float)
    check(not np.isnan(score).any() and np.array_equal(fm, score),
          f'frame means of the 5 answers = {p_fs} score_mean frame by frame')

    # ------------------------------------------------------------------------------------------------- matches
    print('\n1. thinking matches (round 3\'s regex; distance and speed kept)', flush=True)
    has = np.zeros((len(td), 2), bool)
    units = collections.Counter()
    spans = {k: [] for k in KINDS}
    for i, t_ in enumerate(think):
        if not isinstance(t_, str):
            continue
        for m in rk.EST_RE.finditer(t_.lower()):
            kind = rk.EST_GROUP[m.group(1)]
            if kind in KINDS:
                has[i, KINDS.index(kind)] = True
                units[(kind, m.group(1))] += 1
                spans[kind].append((i, m.start(), m.end()))
    est = has.any(1)
    cnt = est[rows5].sum(1)
    p_a3 = os.path.join(R3, 'exploratory_keywords_answers.csv')
    a3 = pd.read_csv(p_a3)
    same_flags = (len(a3) == len(td) and (a3.segment.to_numpy() == td.segment.to_numpy()).all()
                  and (a3.frame_idx.to_numpy() == td.frame_idx.to_numpy()).all()
                  and (a3['repeat'].to_numpy() == td['repeat'].to_numpy()).all()
                  and all(np.array_equal(a3[f'thinking_{k}_estimate'].astype(str).str.lower().eq('true').to_numpy(),
                                         has[:, j]) for j, k in enumerate(KINDS)))
    check(same_flags, f'each answer\'s distance and speed flags = round 3\'s {p_a3}')
    p_u3 = os.path.join(R3, 'exploratory_thinking_estimate_units.csv')
    u3 = pd.read_csv(p_u3, keep_default_na=False)
    u3d = {(r.kind, r.unit): int(r.matches) for r in u3.itertuples() if r.kind in KINDS}
    check(u3d == dict(units), f'the distance and speed unit counts = round 3\'s {p_u3}')
    print(f'  done ({time.time() - t0:.0f} s): answers with distance {int(has[:, 0].sum())}, speed {int(has[:, 1].sum())}, '
          f'either {int(est.sum())}; frames 3+ {int((cnt >= 3).sum())}, 1+ {int((cnt >= 1).sum())}', flush=True)

    # ------------------------------------------------------------------------------------------------- SSMs, resamples
    ssm_df = av.ssm_table(FSSM, frames)
    risk = {s: ssm_df[s + '_risk'].to_numpy(float) for s in av.SSMS}
    check(all(ssm_df[s + '_conflict'].all() and ssm_df[s + '_use'].all() for s in av.SSMS)
          and not any(np.isnan(risk[s]).any() for s in av.SSMS), 'every frame has a value of every SSM')
    p_r2 = os.path.join(R2, 'exploratory_reason_frames.csv')
    r2 = pd.read_csv(p_r2, float_precision='round_trip').set_index(['segment', 'frame_idx']).reindex(mi)
    pct = {s: r2[f'{s}_risk_percentile'].to_numpy(float) for s in av.SSMS}
    check(all(np.array_equal(pct[s], (rankdata(risk[s], method='average') - 1) / (n - 1) * 100) for s in av.SSMS),
          f'the 9 SSM risk percentiles of {p_r2} (used) = a recomputation, frame by frame')
    seg_names = frames.segment.to_numpy()
    seg_list = np.array(sorted(set(seg_names)))
    nb_r, seed_r, frames_r, segs_r = ar.read_params_risk(os.path.join(A, 'params.txt'))
    check((nb_r, seed_r, frames_r) == (N_BOOT, BOOT_SEED, n) and list(seg_list) == segs_r,
          f'{A}/params.txt: {nb_r} resamples, seed {seed_r}, {frames_r} frames, the same {len(seg_list)} segments')
    rng = np.random.default_rng(BOOT_SEED)
    draws = [rng.choice(seg_list, len(seg_list)) for _ in range(N_BOOT)]
    hh = hashlib.sha256()
    for dr in draws:
        hh.update(np.searchsorted(seg_list, dr).astype(np.int32).tobytes())
    check(hh.hexdigest() == HASH, f"drawn segments' sha256 {hh.hexdigest()}")
    two = {'posthoc_est_main': cnt >= 3, 'posthoc_est_sens': cnt >= 1}
    G.update(score=score, pct=pct, risk=risk, two=two, draws=draws,
             groupings=[('all', np.zeros(n, int), 1)] + [(nm, arr.astype(int), 2) for nm, arr in two.items()],
             seg_frames={s: np.flatnonzero(seg_names == s) for s in seg_list})
    point = stats(np.arange(n))
    p_ag = os.path.join(A, 'agreement.csv')
    with open(p_ag, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        ag = {d['ssm']: (rd.line_num, d) for d in rd
              if d['model'] == MODEL and d['condition'] == COND and d['grouping'] == 'all'}
    lines = [ag[s][0] for s in av.SSMS]
    dmax = max(abs(point[('tau', 'all', 'all', s)] - float(ag[s][1]['tau_b_conflict'])) for s in av.SSMS)
    check(lines == list(range(AG_FIRST, AG_FIRST + 9 * AG_STEP, AG_STEP)) and dmax <= TOL,
          f'the 9 tau-b over all frame pairs = {p_ag} lines {lines}: largest |difference| {dmax!r}')
    keys_ = list(point)
    G['keys'] = keys_
    col = {k: i for i, k in enumerate(keys_)}
    print(f'\n2. resamples: {N_BOOT}, {a.workers} worker(s), {len(keys_)} numbers each', flush=True)
    tb = time.time()
    with mp.get_context('fork').Pool(max(a.workers, 1)) as pool:
        boot = np.stack(list(pool.imap(one, range(N_BOOT), chunksize=8)))
    print(f'  done in {time.time() - tb:.0f} s', flush=True)
    comps = []

    def with_ci(k, kind=None, what='', ref=0.0, allow=True):
        lo, hi, nv, miss = rk.ci(boot[:, col[k]]) if allow else (np.nan, np.nan, 0, 0)
        sd = rk.side(lo, hi, ref)
        if kind and allow and not np.isnan(lo):
            comps.append((kind, what, sd))
        return dict(value=point[k], lo=lo, hi=hi, nboot=nv, missing=miss, side=sd)

    # ------------------------------------------------------------------------------------------------- tables
    rows_e = [dict(what=f'answers whose thinking has a {k} estimate', count=int(has[:, j].sum()), of=len(td),
                   share=float(has[:, j].mean())) for j, k in enumerate(KINDS)]
    rows_e.append(dict(what='answers whose thinking has a distance or speed estimate', count=int(est.sum()), of=len(td),
                       share=float(est.mean())))
    for thr, lab in ((3, 'main'), (1, 'sensitivity')):
        rows_e.append(dict(what=f'frames estimated ({thr}+ of 5, {lab})', count=int((cnt >= thr).sum()), of=n,
                           share=float((cnt >= thr).mean())))
        rows_e.append(dict(what=f'frames not estimated ({lab})', count=int((cnt < thr).sum()), of=n,
                           share=float((cnt < thr).mean())))
    rows_u = [dict(kind=k, unit=u, matches=c_) for (k, u), c_ in sorted(units.items())]
    LAB = {'posthoc_est_main': ('estimated, distance or speed (3+ of 5)', 'not estimated'),
           'posthoc_est_sens': ('estimated, distance or speed (1+ of 5)', 'not estimated')}
    rows_g, rows_p, rows_t = [], [], []
    for name, (lab1, lab0) in LAB.items():
        m = two[name]
        for sd_, lab, mm in ((1, lab1, m), (0, lab0, ~m)):
            me = with_ci(('mean', name, sd_))
            rows_g.append(dict(split=name, group=lab, frames=int(mm.sum()), segments=int(len(set(seg_names[mm]))),
                               mean=me['value'], lo=me['lo'], hi=me['hi'], nboot=me['nboot'], missing=me['missing']))
            for s in av.SSMS:
                pp = with_ci(('pct', name, sd_, s), 'SSM percentile median vs 50', f'{name} {lab} {s}', 50.0)
                rows_p.append(dict(split=name, group=lab, ssm=s, frames=int(mm.sum()), median_percentile=pp['value'],
                                   lo=pp['lo'], hi=pp['hi'], nboot=pp['nboot'], missing=pp['missing'], side=pp['side']))
            for s in av.SSMS:
                tt = with_ci(('tau', name, sd_, s), 'tau-b', f'{name} {lab} {s}', allow=mm.sum() >= MIN_TAU)
                rows_t.append(dict(split=name, group=lab, ssm=s, frames=int(mm.sum()), tau_b=tt['value'], lo=tt['lo'],
                                   hi=tt['hi'], nboot=tt['nboot'], missing=tt['missing'], side=tt['side'],
                                   fewer_than_50=bool(mm.sum() < MIN_TAU)))
        for s in av.SSMS:
            tt = with_ci(('tau', name, 'diff', s), 'tau-b difference', f'{name} estimated - not {s}',
                         allow=min(m.sum(), (~m).sum()) >= MIN_TAU)
            rows_t.append(dict(split=name, group='estimated - not estimated', ssm=s, frames=n, tau_b=tt['value'],
                               lo=tt['lo'], hi=tt['hi'], nboot=tt['nboot'], missing=tt['missing'], side=tt['side'],
                               fewer_than_50=False))
    rows_c = []
    for kind in [None] + sorted({x[0] for x in comps}):
        sub = [x for x in comps if kind is None or x[0] == kind]
        rows_c.append(dict(section='C post-hoc', kind=kind or 'all', intervals=len(sub),
                           leaving_out_0_or_50=sum(x[2].startswith(('above', 'below')) for x in sub),
                           by_chance_about=0.05 * len(sub)))
    rs = np.random.default_rng(SEED)
    rows_s = []
    for k in KINDS:
        sp = spans[k]
        pick = np.sort(rs.choice(len(sp), size=min(N_SAMPLE, len(sp)), replace=False)) if sp else []
        for j in pick:
            i, a_, b_ = sp[j]
            t_ = think[i] if len(think[i].lower()) == len(think[i]) else think[i].lower()
            before, after = t_[:a_].split()[-15:], t_[b_:].split()[:15]
            rows_s.append(dict(category=k, name=f'thinking estimate, {k} (post-hoc)', segment=td.segment[i],
                               frame_idx=int(td.frame_idx[i]), repeat=int(td['repeat'][i]),
                               text=' '.join(before) + ' [[' + t_[a_:b_] + ']] ' + ' '.join(after), correct_1_or_0=''))
    fr = frames.copy()
    fr['score_mean'] = score
    fr['distance_estimate_answers'] = has[rows5, 0].sum(1)
    fr['speed_estimate_answers'] = has[rows5, 1].sum(1)
    fr['distance_or_speed_estimate_answers'] = cnt

    def W(name, rows):
        pd.DataFrame(rows).to_csv(os.path.join(O, PFX + name), index=False)

    W('estimates.csv', rows_e)
    W('units.csv', rows_u)
    W('groups.csv', rows_g)
    W('ssm.csv', rows_p)
    W('tau_b.csv', rows_t)
    W('comparisons.csv', rows_c)
    W('comparisons_list.csv', [dict(kind=k_, what=w_, side=s_) for k_, w_, s_ in comps])
    W('sample.csv', rows_s)
    W('check.csv', rows_ck)
    fr.to_csv(os.path.join(O, PFX + 'frames.csv'), index=False)
    with open(os.path.join(O, PFX + 'params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY, POST-HOC: round 3 part C with the thinking estimates limited to distance and speed units '
                f'(user, 2026-11-14, after the round-3 results were seen; plan in CLAUDE.md "第三轮" -> "第三轮补充"); '
                f'round 3\'s own C outputs unchanged\nrun {time.strftime("%Y-%m-%d %H:%M:%S")}; out_dir {os.path.abspath(O)}\n'
                f'program {os.path.abspath(__file__)} sha256 {prog_sha}; reason_keywords.py sha256 {rk_sha}\n')
        f.write('change from round 3 (5.2-5.3): "estimated" uses only the distance units (m, meter(s), metre(s), ft, '
                'feet) and the speed units (mph, km/h, kph, m/s); the time units (s, sec, second(s)) are dropped as a '
                'whole (they cannot be told apart from the image time labels); the rest unchanged (per answer: the '
                'thinking has one; per frame: 3+ of 5 main, 1+ sensitivity)\n')
        f.write(f'thinking: {main_f} field reasoning ({n_forced} forced answers: the reply\'s own reasoning = the forced '
                f'record\'s); regex (round 3) {rk.EST_RE.pattern}; kinds kept {KINDS}\n')
        f.write(f'scores: {p_fs} score_mean; percentiles: {p_r2}; intervals: {N_BOOT} resamples of whole segments, '
                f'default_rng({BOOT_SEED}), sha256 {hh.hexdigest()}; percentiles 2.5 / 97.5 (numpy linear) over the '
                f'resamples where defined; no tau-b interval under {MIN_TAU} frames; sample default_rng({SEED}), '
                f'{N_SAMPLE} distance then {N_SAMPLE} speed matches, 15 words either side\n')
    print('\n' + '=' * 110)
    for r_ in rows_e:
        print(f"  {r_['what']}: {r_['count']} of {r_['of']}")
    for r_ in rows_c:
        print(f"  {r_['kind']}: {r_['intervals']} intervals, {r_['leaving_out_0_or_50']} leave out 0 (or 50), by chance "
              f"about {r_['by_chance_about']:.2f}")
    print(f'\ndone in {time.time() - t0:.0f} s; checks not met: {len(failed)}')
    for w_ in failed:
        print('  ', w_)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
