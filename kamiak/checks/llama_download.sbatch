#!/bin/bash
#SBATCH --job-name=dl-llama
#SBATCH --partition=dtn
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/data/lab/cejli/sy/tmp_checks/llama_download/download_%j.out
# Llama-3.1-8B-Instruct (user approved 2026-11-15): revision 0e9e39f..., top-level files only (original/ excluded).
# The token in HF_HOME is used by the library and never printed. 1. the official file tree with LFS sha256 (an
# unauthenticated listing masks them for this gated repo); 2. dry run; 3. download into models/Llama-3.1-8B-Instruct.
set -euo pipefail
export HF_HOME=/data/lab/cejli/sy/hf_cache
REV=0e9e39f249a16976918f6564b8830bc894c89659
D=/data/lab/cejli/sy/tmp_checks/llama_download
E=/data/lab/cejli/sy/conda_envs/vllm_cu129/bin
OUT=/data/lab/cejli/sy/models/Llama-3.1-8B-Instruct
[ ! -e $OUT ] || { echo "$OUT exists: stop"; exit 1; }
echo "node $(hostname), start $(date '+%F %T'); free in group dir: $(df -B1 /data/lab/cejli/sy | tail -1 | awk '{print $4}') bytes"
$E/python -I - "$REV" "$D/tree_auth.json" <<'PY'
import json, sys
from huggingface_hub import HfApi
rev, out = sys.argv[1:3]
items = []
for t in HfApi().list_repo_tree('meta-llama/Llama-3.1-8B-Instruct', revision=rev, recursive=True, expand=True):
    d = {'type': 'file' if hasattr(t, 'size') else 'directory', 'path': t.path}
    if d['type'] == 'file':
        d['size'] = t.size; d['oid'] = t.blob_id
        if t.lfs: d['lfs'] = {'oid': t.lfs.sha256, 'size': t.lfs.size}
    items.append(d)
json.dump(items, open(out, 'w'), indent=1)
print(f'tree: {len(items)} entries written to {out}')
PY
echo "=== dry run"; $E/hf download meta-llama/Llama-3.1-8B-Instruct --revision $REV --exclude "original/*" --local-dir $OUT --dry-run
echo "=== download $(date +%T)"; $E/hf download meta-llama/Llama-3.1-8B-Instruct --revision $REV --exclude "original/*" --local-dir $OUT
echo "files: $(find $OUT -type f -not -path '*/.cache/*' | wc -l), bytes: $(find $OUT -type f -not -path '*/.cache/*' -printf '%s\n' | awk '{s+=$1} END {print s}')"
echo "end $(date '+%F %T')"
