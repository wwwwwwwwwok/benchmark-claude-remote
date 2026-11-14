#!/bin/bash
# The environment for the reason analysis's sentence vectors (user, 2026-11-13: a new environment with these versions;
# CLAUDE.md "理由分析"). It only computes the sentence vectors of sentence-transformers/all-mpnet-base-v2 and writes them
# to a file; TF-IDF, KMeans, t-SNE and the resamples run in the waymo environment. Python as the waymo environment;
# torch 2.13.0 as the vLLM environment but the CPU build; transformers, tokenizers, huggingface-hub, safetensors as the
# vLLM environment; numpy, scipy, scikit-learn as the waymo environment; sentence-transformers 6.2.0 (2026-10-21).
# Run on a compute node (srun); the existing environments are not touched; pip's cache is not written.
set -eo pipefail                # not -u: conda's scripts use unset variables
E=/data/lab/cejli/sy/conda_envs/sentence
[ ! -e $E ] || { echo "$E exists: stop" >&2; exit 1; }
source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
echo "node $(hostname), start $(date '+%F %T')"
conda create -p $E python=3.13.16 -y
$E/bin/pip install --no-cache-dir torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu
$E/bin/pip install --no-cache-dir sentence-transformers==6.2.0 transformers==5.19.0 tokenizers==0.23.3 \
    huggingface-hub==2.2.0 safetensors==0.8.0 numpy==2.3.5 scipy==1.16.3 scikit-learn==1.9.1
$E/bin/python -I -c "import sys, torch, sentence_transformers, transformers, tokenizers, huggingface_hub, safetensors, numpy, scipy, sklearn; print(sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'sentence-transformers', sentence_transformers.__version__, 'transformers', transformers.__version__, 'tokenizers', tokenizers.__version__, 'huggingface-hub', huggingface_hub.__version__, 'safetensors', safetensors.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'scikit-learn', sklearn.__version__)"
echo "--- pip freeze"
$E/bin/pip freeze
echo "--- size: $(du -sb --apparent-size $E | cut -f1) bytes; end $(date '+%F %T')"
