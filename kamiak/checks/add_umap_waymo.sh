#!/bin/bash
# Adds umap-learn to the analysis environment conda_envs/waymo (user, 2026-11-14, for the exploratory HDBSCAN check
# of the reason clusters; done as add_sklearn_waymo.sh did scikit-learn). Packages may only be added: every package already there is pinned
# by a constraints file made from pip freeze, so pip stops rather than change one; pip freeze before and after are
# compared and the script fails unless the only differences are added lines. pip's cache is not written.
# Run on a compute node (srun or sbatch), not on a login node.
# usage: bash add_umap_waymo.sh <log folder>   (freeze_before.txt, constraints.txt, freeze_after.txt, added.txt there)
set -euo pipefail
E=/data/lab/cejli/sy/conda_envs/waymo
O=$1
mkdir -p "$O"
[ ! -e "$O/freeze_before.txt" ] || { echo "$O/freeze_before.txt exists: give a new folder" >&2; exit 1; }
echo "node $(hostname), start $(date '+%F %T')"
if $E/bin/python -I -c "import umap" 2>/dev/null; then echo "umap-learn is already in $E: stop" >&2; exit 1; fi
$E/bin/pip freeze --all > "$O/freeze_before.txt"
cp "$O/freeze_before.txt" "$O/constraints.txt"
echo "--- before: $(wc -l < "$O/freeze_before.txt") packages; versions on the index:"
$E/bin/pip index versions umap-learn 2>/dev/null | head -n 2 || true
echo "--- dry run"
$E/bin/pip install --no-cache-dir --dry-run --constraint "$O/constraints.txt" umap-learn
echo "--- install"
$E/bin/pip install --no-cache-dir --constraint "$O/constraints.txt" umap-learn
$E/bin/pip freeze --all > "$O/freeze_after.txt"
comm -23 <(sort "$O/freeze_before.txt") <(sort "$O/freeze_after.txt") > "$O/removed_or_changed.txt"
comm -13 <(sort "$O/freeze_before.txt") <(sort "$O/freeze_after.txt") > "$O/added.txt"
echo "--- added:"; cat "$O/added.txt"
echo "--- removed or changed: $(wc -l < "$O/removed_or_changed.txt")"; cat "$O/removed_or_changed.txt"
$E/bin/python -I -c "import sys, numpy, pandas, scipy, sklearn, numba, llvmlite, pynndescent, umap; from sklearn.cluster import HDBSCAN; print(sys.version.split()[0], 'numpy', numpy.__version__, 'pandas', pandas.__version__, 'scipy', scipy.__version__, 'scikit-learn', sklearn.__version__, 'numba', numba.__version__, 'llvmlite', llvmlite.__version__, 'pynndescent', pynndescent.__version__, 'umap-learn', umap.__version__)"
echo "end $(date '+%F %T')"
[ ! -s "$O/removed_or_changed.txt" ] || { echo "packages removed or changed: FAIL" >&2; exit 1; }
echo "only added: OK"
