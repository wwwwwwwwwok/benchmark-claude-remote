"""EXPLORATORY (user, 2026-11-13; CLAUDE.md "理由分析", sections 2 and 3 as revised): the sentence vectors of the reasons,
sentence-transformers/all-mpnet-base-v2 (rev e8c3b32edf5434bc2275fc9bab85f82640a19130, the local folder of 14 files),
one vector per reason as written (no change to the text), normalize_embeddings=True, on the CPU. Runs in the sentence
environment (conda_envs/sentence, kamiak/checks/install_sentence_env.sh); the rest of the analysis is reason_clusters.py.
Input: the reasons file reason_clusters.py extract writes (one JSON per line, key `reason`, in its row order). Output:
<embeddings>.npy (float32, rows in the same order) and <embeddings>.json (the sha256 of the reasons as read, the
model folder, versions, threads, the longest reason in tokens).
usage: python -I reason_embed.py <model folder> <reasons jsonl> <embeddings .npy> [--threads 4]
"""
import os
import sys

THREADS = int(sys.argv[sys.argv.index('--threads') + 1]) if '--threads' in sys.argv else 4
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = str(THREADS)
os.environ['HF_HUB_OFFLINE'] = '1'                       # the local folder only, nothing fetched
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import sentence_transformers  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

REPO, REV = 'sentence-transformers/all-mpnet-base-v2', 'e8c3b32edf5434bc2275fc9bab85f82640a19130'
model_dir, reasons_path, out = sys.argv[1:4]
torch.set_num_threads(THREADS)
t0 = time.time()
texts = []
h = hashlib.sha256()
with open(reasons_path, encoding='utf-8') as f:
    for line in f:
        r = json.loads(line)['reason']
        texts.append(r)
        h.update(r.encode('utf-8') + b'\n')
model = SentenceTransformer(model_dir, device='cpu', model_kwargs={'use_safetensors': True})
emb = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
emb = emb.astype(np.float32)
lens = [len(model.tokenizer(t)['input_ids']) for t in texts]
np.save(out, emb)
meta = dict(repo=REPO, revision=REV, model_dir=os.path.abspath(model_dir), reasons=os.path.abspath(reasons_path),
            n=len(texts), dim=int(emb.shape[1]), reasons_sha256=h.hexdigest(), normalize_embeddings=True,
            max_seq_length=int(model.max_seq_length), longest_reason_tokens=int(max(lens)),
            reasons_over_max_seq_length=int(sum(n > model.max_seq_length for n in lens)),
            norms_min=float(np.linalg.norm(emb, axis=1).min()), norms_max=float(np.linalg.norm(emb, axis=1).max()),
            threads=THREADS, sentence_transformers=sentence_transformers.__version__, transformers=transformers.__version__,
            torch=torch.__version__, numpy=np.__version__, modules=[type(m).__name__ for m in model],
            seconds=round(time.time() - t0, 1))
with open(os.path.splitext(out)[0] + '.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=1)
print(json.dumps(meta, indent=1))
