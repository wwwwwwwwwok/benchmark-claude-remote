#!/bin/bash
# The environment for Qwen-Drive-1.0-4B's official code (user, 2026-10-26), only for the vLLM vs official-code check:
# the versions pinned in the official requirements.txt (verified there with Python 3.12, CUDA 12.8), without
# flash-attn (only for attn_implementation="flash_attention_2"; the VQA path uses sdpa), opencv-python and matplotlib
# (planning, perception and plots only). causal-conv1d from its GitHub release (prebuilt for cu12, torch 2.8, cp312).
# The official package itself is not installed (it would pull flash-attn); its source is put on PYTHONPATH.
# Run on a compute node (sbatch); the existing environments are not touched; pip's cache is not written.
set -eo pipefail                # not -u: conda's scripts use unset variables
E=/data/lab/cejli/sy/conda_envs/qwen_drive
[ ! -e $E ] || { echo "$E exists: stop" >&2; exit 1; }
source /opt/apps/miniconda3/3.12/etc/profile.d/conda.sh
echo "node $(hostname), start $(date '+%F %T')"
conda create -p $E python=3.12 -y
$E/bin/pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 transformers==5.14.1 accelerate==1.12.0 \
    safetensors==0.8.0 numpy==2.2.6 pillow==12.0.0 flash-linear-attention==0.5.1 tqdm==4.67.1
$E/bin/pip install --no-cache-dir --no-deps \
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1%2Bcu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
$E/bin/python -I -c "import sys, torch, torchvision, transformers, accelerate, safetensors, numpy, PIL, fla, tqdm, causal_conv1d; print(sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda, 'torchvision', torchvision.__version__, 'transformers', transformers.__version__, 'accelerate', accelerate.__version__, 'safetensors', safetensors.__version__, 'numpy', numpy.__version__, 'pillow', PIL.__version__, 'fla', fla.__version__, 'causal_conv1d', causal_conv1d.__version__, 'tqdm', tqdm.__version__)"
echo "--- pip freeze"
$E/bin/pip freeze
echo "--- size: $(du -sb --apparent-size $E | cut -f1) bytes; end $(date '+%F %T')"
