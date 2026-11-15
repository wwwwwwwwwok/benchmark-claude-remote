"""EXPLORATORY, for a figure only (user, 2026-11-14; the plan is in CLAUDE.md "理由分析" -> "二维 UMAP 坐标"): 2D UMAP
coordinates of round 1's 5,420 answer sentence vectors (Qwen3.6-27B, sep_camera, main set; tmp_checks/reason_27b/run1/
embeddings.npy, sha256 30ba0002..., as stored, float32; rows = answers_base.csv). Nothing is re-embedded or re-clustered.
umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=s), the rest default
(compatibility_layout not set, as in the HDBSCAN run), s = 20261021 (umap_x, umap_y) and 20261022 (umap_x_s2, umap_y_s2).
The cluster table (main_1084/exploratory_reason_answers.csv: cluster = kmeans7, risk = score, tsne_x, tsne_y) is matched to
the vectors' rows by segment, frame_idx, repeat, not by row order. No reason text is written.
Checks (exit 1 when one is not met): the vectors' sha256 and shape; every answer once in both tables, matched by key; the
score = answers_base's risk after matching; kmeans7 counts 607, 773, 897, 851, 1190, 614, 488 for clusters 0-6. The run
script compares a second run byte for byte.
Output: <out_dir>/exploratory_umap2d_answers.csv (rows in the vectors' order) and exploratory_umap2d_params.txt.
usage: python -I reason_umap2d.py <waymo_v143 folder> <round-1 work folder> <out_dir>
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
import hashlib  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

EMB1_SHA = '30ba000292417d5047d2279dba1fa1a084883a49ebc9cd379a9aeee6a46296be'
COUNTS = [607, 773, 897, 851, 1190, 614, 488]
SEEDS = (20261021, 20261022)
failed = []


def check(ok, what):
    print(('  check ok: ' if ok else '  CHECK NOT MET: ') + what, flush=True)
    if not ok:
        failed.append(what)


def main():
    import umap
    L, W1, O = sys.argv[1:4]
    os.makedirs(O, exist_ok=True)
    prog_sha = hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()
    print(f'EXPLORATORY (figure only): 2D UMAP of the answer vectors; program sha256 {prog_sha}; umap-learn {umap.__version__}')
    p_e = os.path.join(W1, 'embeddings.npy')
    E = np.load(p_e)
    check(hashlib.sha256(open(p_e, 'rb').read()).hexdigest() == EMB1_SHA and E.shape == (5420, 768),
          f'{p_e}: sha256 = round 1\'s, {E.shape} {E.dtype}')
    key = ['segment', 'frame_idx', 'repeat']
    td = pd.read_csv(os.path.join(W1, 'answers_base.csv'), keep_default_na=False, na_values=[''],
                     float_precision='round_trip')
    p_c = os.path.join(L, 'analysis', 'reason_27b', 'main_1084', 'exploratory_reason_answers.csv')
    ca = pd.read_csv(p_c, float_precision='round_trip')
    ki, kc = pd.MultiIndex.from_frame(td[key]), pd.MultiIndex.from_frame(ca[key])
    check(len(td) == len(E) and ki.is_unique and kc.is_unique and set(ki) == set(kc),
          f'every one of the {len(td)} answers once in answers_base.csv and {p_c} (matched by segment, frame_idx, repeat)')
    if failed:
        sys.exit('input checks not met; nothing written')
    cm = ca.set_index(key).reindex(ki)
    check(np.array_equal(cm.risk.to_numpy(float), td.risk.to_numpy(float)) and not cm.cluster.isna().any(),
          'after matching: the cluster table\'s risk = answers_base\'s risk, answer by answer; every answer has a cluster')
    counts = np.bincount(cm.cluster.to_numpy(int), minlength=7).tolist()
    check(counts == COUNTS, f'kmeans7 counts {counts} (expected {COUNTS})')
    if failed:
        sys.exit('checks not met; nothing written')
    out = td[key].copy()
    out['kmeans7'] = cm.cluster.to_numpy(int)
    out['score'] = cm.risk.to_numpy(float)
    for s, sx in zip(SEEDS, ('', '_s2')):
        t1 = time.time()
        U = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=s).fit_transform(E)
        out[f'umap_x{sx}'], out[f'umap_y{sx}'] = U[:, 0].astype(float), U[:, 1].astype(float)
        print(f'  UMAP seed {s}: {time.time() - t1:.0f} s')
    out['tsne_x'] = cm.tsne_x.to_numpy(float)
    out['tsne_y'] = cm.tsne_y.to_numpy(float)
    out.to_csv(os.path.join(O, 'exploratory_umap2d_answers.csv'), index=False)
    with open(os.path.join(O, 'exploratory_umap2d_params.txt'), 'w', encoding='utf-8') as f:
        f.write(f'EXPLORATORY, figure only: 2D UMAP coordinates of the 5,420 answer vectors (user, 2026-11-14)\nrun '
                f'{time.strftime("%Y-%m-%d %H:%M:%S")}; out_dir {os.path.abspath(O)}\nprogram {os.path.abspath(__file__)} '
                f'sha256 {prog_sha}; umap-learn {umap.__version__}, numpy {np.__version__}, pandas {pd.__version__}\n'
                f'vectors {p_e} sha256 {EMB1_SHA}; cluster table {p_c} matched by segment, frame_idx, repeat\n'
                f'UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=<seed>), the rest default; '
                f'seeds {SEEDS[0]} (umap_x, umap_y), {SEEDS[1]} (umap_x_s2, umap_y_s2)\nkmeans7 counts {counts}\n')
    print(f'done; checks not met: {len(failed)}')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main()
