#!/bin/bash
# sha256 and size of every real file behind the links in waymo_v143/to_office/ (user, 2026-10-23: the folder for
# taking results to the office computer holds symbolic links only; every file in the linked analysis folder listed).
# Read through the links, so a broken link stops the script. Writes to_office/sha256.txt (not overwritten).
# usage (on a compute node): bash to_office_sha256.sh
set -euo pipefail
T=/data/lab/cejli/sy/waymo_v143/to_office
cd "$T"
[ ! -e sha256.txt ] || { echo "already exists: $T/sha256.txt" >&2; exit 1; }
[ -z "$(find . -type f)" ] || { echo "$T holds real files, expected links only:" >&2; find . -type f >&2; exit 1; }
out=$(mktemp "$T/.sha256.XXXXXX")
n=0; total=0
{
    echo "# to_office/ holds symbolic links only; below, the sha256 and bytes of the real file behind each one"
    echo "# (sha256sum and stat -L read through the links; $(date '+%Y-%m-%d %H:%M:%S %Z') on $(hostname))."
    echo "# links:"
    find . -type l | sed 's|^\./||' | LC_ALL=C sort | while read -r p; do
        echo "#   $p -> $(readlink "$p")$( [ -d "$p" ] && echo ' (folder: every file in it listed)')"
    done
    echo "# sha256  bytes  path under to_office/"
    while read -r p; do
        b=$(stat -L -c %s "$p")
        echo "$(sha256sum < "$p" | cut -c1-64)  $b  $p"
    done < <(find -L . -type f ! -name '.sha256.*' | sed 's|^\./||' | LC_ALL=C sort)
} > "$out"
read -r n total < <(awk '!/^#/ {n++; s += $2} END {printf "%d %.0f\n", n, s}' "$out")
echo "# total: $n files, $total bytes" >> "$out"
mv "$out" sha256.txt
chmod 640 sha256.txt
cat sha256.txt
