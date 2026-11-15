"""EXPLORATORY (user, 2026-11-14, follow-up to the HDBSCAN check; text only, no image): one row per cluster of the main
HDBSCAN settings of M1 (answers, min_cluster_size 50, 20 clusters) and M3 (frames, mean vectors, 20, 7 clusters), and
the M1 noise group, so the clusters can be read.
Per row: answers (M1) / frames, segments, mean minus overall and its 95% interval (copied from exploratory_hdbscan_groups.csv);
the KMeans clusters it overlaps (exploratory_hdbscan_crosstab.csv, every non-zero count, largest first); the 8 most
distinctive terms against all other non-noise points (the noise group: against all non-noise points), with the
log-odds ratio with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008) exactly as code/reason_scorebins.py
computes it: main_hazard and reason of each answer cleaned as round 1 (reason_clusters.clean) and counted with
CountVectorizer(stop_words = round 1's custom list + the risk words, ngram_range (1, 2)), the two fields counted apart
and added; alpha_w = the term's count over all 5,420 answers, alpha_0 their sum; delta = log((y_i + a_w) / (n_i + a_0 -
y_i - a_w)) - log((y_j + a_w) / (n_j + a_0 - y_j - a_w)), z = delta / sqrt(1 / (y_i + a_w) + 1 / (y_j + a_w)); eligible:
in 20+ answers and 5+ segments over all 5,420 answers; the 8 highest z (ties: term order); each term is also given with
the number of the group's answers that use it, because with this prior a term used by neither side (here: mostly by
noise answers, which are in neither) still gets z > 0, the more so for a small group. M3 groups are the 5 answers
of each frame of the group. A one-sentence description of the group's 10 examples is Claude's reading, read from a
separate file (docs/reason_hdbscan_descriptions.csv) written after the examples were read; it is not computed. The M1
noise group's 10 examples follow the plan's rule (highest probabilities_, ties by segment, frame_idx, repeat); every
noise point has probability 0, so they are the first 10 noise answers in segment, frame_idx, repeat order.
Checks (exit 1 when one is not met): the replies read = round 1's answers (5,420, risk = the tidy risk, the reasons =
round 1's reasons); the labels files in the answers' / frames' order; group sizes = exploratory_hdbscan_groups.csv.
Outputs (out_dir, prefix exploratory_hdbscan_describe_): .csv (one row per group), _terms.csv (the 8 terms per group with
z, delta and counts), _noise_examples.csv (M1 noise, text cut to 300 characters), _check.csv, _params.txt.
usage: python -I reason_hdbscan_describe.py <waymo_v143> <reason work folder> <hdbscan out_dir> <out_dir>
       [<descriptions csv>]
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_vlm_ssm as av  # noqa: E402
import reason_clusters as rc  # noqa: E402

MODEL, COND = 'qwen3.6-27b', 'sep_camera'
MIN_ANS, MIN_SEG, N_TERMS, N_EX, N_CHARS = 20, 5, 8, 10, 300
REASONS_SHA = 'c718132acf962585cb1463f0fc1736b77d984b598808498f2bd8e5a3c7345373'
MAIN = {'M1': (50, 20261021), 'M3': (20, 20261021)}
failed, rows_ck = [], []


def check(ok, what):
    rows_ck.append(dict(check=what, ok=bool(ok)))
    print(('  check ok: ' if ok else '  CHECK NOT MET: ') + what, flush=True)
    if not ok:
        failed.append(what)


def main():
    from sklearn.feature_extraction.text import CountVectorizer
    t0 = time.time()
    L, WR, H, O = sys.argv[1:5]
    p_desc = sys.argv[5] if len(sys.argv) > 5 else None
    os.makedirs(O, exist_ok=True)
    prog_sha = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()
    print(f'EXPLORATORY: HDBSCAN cluster descriptions; program sha256 {prog_sha}')
    W1 = os.path.join(WR, 'run1')
    td = pd.read_csv(os.path.join(W1, 'answers_base.csv'), keep_default_na=False, na_values=[''],
                     float_precision='round_trip')
    recs = [json.loads(x) for x in open(os.path.join(W1, 'reasons.jsonl'), encoding='utf-8')]
    keys = {(s, int(f), int(r)): i for i, (s, f, r) in enumerate(zip(td.segment, td.frame_idx, td['repeat']))}
    A, MD, _ = rc.paths(L)
    main_f, forced_f = av.jsonl_file(MD, f'main_{MODEL}'), av.jsonl_file(MD, f'main_{MODEL}_forced_answers')
    parsed = [None] * len(td)
    for fn, is_forced in ((main_f, False), (forced_f, True)):
        with av.open_jsonl(fn) as f:
            for raw in f:
                d = av.load_light(raw)
                if d['condition'] != COND:
                    continue
                i = keys.get((d['segment'], int(d['frame_idx']), int(d['repeat'])))
                if i is None:
                    continue
                if not is_forced:
                    parsed[i] = d.get('parsed') or {}
                elif d.get('parsed') and not parsed[i]:
                    parsed[i] = d['parsed']

    def fld(p, k):
        v = (p or {}).get(k)
        return v if isinstance(v, str) else ''

    t_mh = [fld(p, 'main_hazard') for p in parsed]
    t_rs = [fld(p, 'reason') for p in parsed]
    h = hashlib.sha256()
    for r in recs:
        h.update(r['reason'].encode('utf-8') + b'\n')
    check(all(parsed) and np.array_equal(np.array([float(p['risk']) for p in parsed]), td.risk.to_numpy(float))
          and h.hexdigest() == REASONS_SHA and t_rs == [r['reason'] for r in recs],
          f'{len(td)} answers read from {main_f} and {forced_f}; risk = the tidy risk; reasons = round 1\'s')

    l1 = pd.read_csv(os.path.join(H, 'exploratory_hdbscan_labels_m1.csv'), float_precision='round_trip')
    l3 = pd.read_csv(os.path.join(H, 'exploratory_hdbscan_labels_m3.csv'), float_precision='round_trip')
    check(len(l1) == len(td) and (l1.segment.to_numpy() == td.segment.to_numpy()).all()
          and (l1.frame_idx.to_numpy() == td.frame_idx.to_numpy()).all()
          and (l1['repeat'].to_numpy() == td['repeat'].to_numpy()).all(), 'labels_m1 in the answers\' order')
    frames = l3[['segment', 'frame_idx']]
    fidx = pd.Series(np.arange(len(frames)), index=pd.MultiIndex.from_frame(frames)) \
        .reindex(pd.MultiIndex.from_arrays([td.segment, td.frame_idx])).to_numpy()
    check(len(frames) == 1084 and not np.isnan(fidx.astype(float)).any(), 'labels_m3: 1,084 frames, every answer\'s frame')
    fidx = fidx.astype(int)
    lab1 = l1[f'hdbscan_m{MAIN["M1"][0]}_seed{MAIN["M1"][1]}'].to_numpy(int)
    lab3a = l3[f'hdbscan_m{MAIN["M3"][0]}_seed{MAIN["M3"][1]}'].to_numpy(int)[fidx]       # each answer: its frame's label
    gr = pd.read_csv(os.path.join(H, 'exploratory_hdbscan_groups.csv'), float_precision='round_trip')
    ct = pd.read_csv(os.path.join(H, 'exploratory_hdbscan_crosstab.csv'))
    ex = pd.read_csv(os.path.join(H, 'exploratory_hdbscan_examples.csv'), keep_default_na=False)

    stop = sorted(set(rc.CUSTOM_STOP) | set(rc.RISK_WORDS))
    cv = CountVectorizer(stop_words=stop, ngram_range=(1, 2))
    c_mh = [rc.clean(x) for x in t_mh]
    c_rs = [rc.clean(x) for x in t_rs]
    cv.fit(c_mh + c_rs)
    C = (cv.transform(c_mh) + cv.transform(c_rs)).tocsr()
    vocab = cv.get_feature_names_out()
    alpha = np.asarray(C.sum(0)).ravel().astype(float)
    a0 = alpha.sum()
    D = (C > 0).astype(np.int8)
    dfreq = np.asarray(D.sum(0)).ravel()
    seg_list = np.array(sorted(set(td.segment)))
    seg_codes = np.searchsorted(seg_list, td.segment.to_numpy())
    Dc = D.tocsc()
    nseg = np.array([len(set(seg_codes[Dc.indices[Dc.indptr[w]:Dc.indptr[w + 1]]])) for w in range(len(vocab))])
    elig = np.flatnonzero((dfreq >= MIN_ANS) & (nseg >= MIN_SEG))
    print(f'  vocabulary {len(vocab)} terms, eligible {len(elig)}')

    desc = {}
    if p_desc:
        dd = pd.read_csv(p_desc, keep_default_na=False)
        desc = {(r.method, str(r.group)): r.description for r in dd.itertuples()}

    rows, rows_t = [], []
    for m, lab in (('M1', lab1), ('M3', lab3a)):
        cl = sorted(set(lab) - {-1})
        groups = cl + ([-1] if m == 'M1' else [])
        for g in groups:
            gi = lab == g
            gj = (lab != -1) & ~gi if g != -1 else lab != -1
            yi = np.asarray(C[gi].sum(0)).ravel().astype(float)
            yj = np.asarray(C[gj].sum(0)).ravel().astype(float)
            ni, nj = yi.sum(), yj.sum()
            delta = np.log((yi + alpha) / (ni + a0 - yi - alpha)) - np.log((yj + alpha) / (nj + a0 - yj - alpha))
            z = delta / np.sqrt(1.0 / (yi + alpha) + 1.0 / (yj + alpha))
            ai = np.asarray(D[gi].sum(0)).ravel()
            aj = np.asarray(D[gj].sum(0)).ravel()
            order = elig[np.lexsort((elig, -z[elig]))][:N_TERMS]
            name = 'noise' if g == -1 else str(g)
            for r, w in enumerate(order, 1):
                rows_t.append(dict(method=m, group=name, rank=r, term=vocab[w], z=float(z[w]), delta=float(delta[w]),
                                   answers_in_group=int(ai[w]), answers_rest=int(aj[w]), answers_all=int(dfreq[w]),
                                   segments_all=int(nseg[w])))
            g_row = gr[(gr.method == m) & (gr.group.astype(str) == name)]
            check(len(g_row) == 1 and (m != 'M1' or int(g_row.answers.iloc[0]) == int(gi.sum()))
                  and (m != 'M3' or int(g_row.frames.iloc[0]) == int(gi.sum()) // 5),
                  f'{m} {name}: size = exploratory_hdbscan_groups.csv')
            g_row = g_row.iloc[0]
            c_ = ct[(ct.method == m) & (ct.min_cluster_size == MAIN[m][0]) & (ct.hdbscan_cluster.astype(str) == name)]
            c_ = c_[c_['count'] > 0].sort_values(['count', 'kmeans_cluster'], ascending=[False, True])
            unit = 'answers' if m == 'M1' else 'frames'
            rows.append(dict(method=m, group=name, answers=(int(gi.sum()) if m == 'M1' else ''),
                             frames=int(g_row.frames), segments=int(g_row.segments),
                             mean_minus_overall=float(g_row.mean_minus_overall), lo95=float(g_row.lo95),
                             hi95=float(g_row.hi95), missing=int(g_row.missing),
                             kmeans_largest=int(c_.kmeans_cluster.iloc[0]), kmeans_largest_count=int(c_['count'].iloc[0]),
                             kmeans_largest_share=float(c_['count'].iloc[0] / c_['count'].sum()),
                             kmeans_overlap=f'{unit}: ' + '; '.join(f'k{int(a)} {int(b)}' for a, b in
                                                                   zip(c_.kmeans_cluster, c_['count'])),
                             top_8_terms='; '.join(vocab[w] for w in order),
                             top_8_terms_answers_in_group='; '.join(f'{vocab[w]} ({int(ai[w])})' for w in order),
                             terms_in_no_answer_of_group=int(sum(ai[w] == 0 for w in order)),
                             description_claude_reading=desc.get((m, name), '')))
    nz = np.flatnonzero(lab1 == -1)
    nz = sorted(nz, key=lambda i: (td.segment[i], int(td.frame_idx[i]), int(td['repeat'][i])))[:N_EX]
    check(all(float(p) == 0.0 for p in l1.probability_main.to_numpy()[lab1 == -1]),
          'every M1 noise answer has probability 0 (so its examples are the first 10 in segment, frame, repeat order)')
    rows_n = [dict(method='M1', group='noise', rank=r + 1, probability=float(l1.probability_main[i]),
                   segment=td.segment[i], frame_idx=int(td.frame_idx[i]), repeat=int(td['repeat'][i]),
                   score=float(td.risk[i]), text_first_300_characters=t_rs[i][:N_CHARS]) for r, i in enumerate(nz)]
    if p_desc:
        want = {(r['method'], r['group']) for r in rows}
        check(set(desc) == want, f'descriptions file {p_desc}: one description per group ({len(desc)} of {len(want)})')

    pfx = os.path.join(O, 'exploratory_hdbscan_describe')
    pd.DataFrame(rows).to_csv(pfx + '.csv', index=False)
    pd.DataFrame(rows_t).to_csv(pfx + '_terms.csv', index=False)
    pd.DataFrame(rows_n).to_csv(pfx + '_noise_examples.csv', index=False)
    pd.DataFrame(rows_ck).to_csv(pfx + '_check.csv', index=False)
    with open(pfx + '_params.txt', 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY: descriptions of the main HDBSCAN clusters (M1 m 50, M3 m 20; UMAP seed 20261021) and the '
                f'M1 noise group (user, 2026-11-14)\nrun {time.strftime("%Y-%m-%d %H:%M:%S")}; out_dir {os.path.abspath(O)}\n')
        f.write(f'program {os.path.abspath(__file__)} sha256 {prog_sha}\n')
        f.write(f'terms: Monroe et al. log-odds, informative Dirichlet prior, as code/reason_scorebins.py; vocabulary '
                f'{len(vocab)}, eligible {len(elig)} ({MIN_ANS}+ answers, {MIN_SEG}+ segments over all 5,420 answers); '
                f'alpha_0 {int(a0)}; group vs all other non-noise points (noise: vs all non-noise points); M3 groups = '
                f'the 5 answers of each frame\n')
        f.write(f'descriptions: {p_desc or "(none given)"}' + (f' sha256 {hashlib.sha256(open(p_desc, "rb").read()).hexdigest()}'
                                                              if p_desc else '') + ' (Claude\'s reading of the 10 examples)\n')
    print('\n' + '=' * 110)
    for i, r in enumerate(rows):
        print(f"  line {i + 2}: {r['method']} {r['group']}: answers {r['answers']}, frames {r['frames']}, segments "
              f"{r['segments']}; minus overall {r['mean_minus_overall']!r} [{r['lo95']!r}, {r['hi95']!r}]; "
              f"{r['kmeans_overlap']}; terms {r['top_8_terms']}")
    print(f'\ndone in {time.time() - t0:.0f} s; checks not met: {len(failed)}')
    for w in failed:
        print('  ', w)
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    if len(sys.argv) not in (5, 6):
        sys.exit(__doc__)
    main()
