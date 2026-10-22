#!/bin/bash
# Download one split of Waymo Open Dataset Perception v1.4.3 (segment-*_with_camera_labels.tfrecord, with maps) from
# Waymo's Google Cloud bucket into a Kamiak scratch workspace. One-time setup (gcloud, your own Google login) and the
# whole sequence: kamiak/README.md. kamiak/waymo_submit.sh submits this together with the later steps.
#
# Usage (repository root, after `mkdir -p logs`):
#   sbatch kamiak/waymo_download.sbatch <workspace> validation     # 202 files
#   sbatch kamiak/waymo_download.sbatch <workspace> training       # 798 files
# Files already downloaded are skipped (--no-clobber), so a rerun resumes. gcloud compares each file's checksum with
# the bucket while downloading; afterwards this script checks that every file in the bucket listing is here with the
# same size, and only then writes <workspace>/raw/<split>/DOWNLOAD_COMPLETE.
#SBATCH --job-name=waymo-dl
#SBATCH --partition=dtn                # data transfer node dtn1 (8 cores); asking for more than 900M means a long queue
#SBATCH --account=cejli
#SBATCH --cpus-per-task=2
#SBATCH --mem=900M
#SBATCH --time=1-00:00:00              # a timed-out run resumes when resubmitted
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
WS=${1:?usage: sbatch kamiak/waymo_download.sbatch <workspace> validation|training}
SPLIT=${2:?usage: sbatch kamiak/waymo_download.sbatch <workspace> validation|training}
case "$SPLIT" in                       # file counts seen in the bucket on 2026-10-21
    validation) EXPECTED=202 ;;
    training)   EXPECTED=798 ;;
    *) echo "split must be validation or training (the test splits have no public labels)"; exit 1 ;;
esac
GCLOUD=/data/lab/cejli/sy/google-cloud-sdk/bin/gcloud   # its bundled Python: dtn1 only has Python 3.9, too old
SRC=gs://waymo_open_dataset_v_1_4_3/individual_files/$SPLIT
DST=$WS/raw/$SPLIT
mkdir -p "$DST"
# dtn1 has little memory to spare: 2 processes x 4 threads
export CLOUDSDK_STORAGE_PROCESS_COUNT=2 CLOUDSDK_STORAGE_THREAD_COUNT=4

# what the bucket holds (size, date, name of every file); kept next to the files
"$GCLOUD" storage ls -l "$SRC/*.tfrecord" > "$DST/bucket_listing.txt"
n=$(grep -c 'gs://' "$DST/bucket_listing.txt" || true)
[[ $n -eq $EXPECTED ]] || { echo "the bucket lists $n files for $SPLIT, expected $EXPECTED"; exit 1; }
echo "$(date '+%F %T') $SPLIT: $(tail -n 1 "$DST/bucket_listing.txt")"

for attempt in 1 2 3; do
    "$GCLOUD" storage cp --no-clobber "$SRC/*.tfrecord" "$DST/" && break
    echo "$(date '+%F %T') attempt $attempt ended with an error"
    [[ $attempt -lt 3 ]] && sleep 60
done
echo "$(date '+%F %T') download finished; checking sizes"

bad=0
while read -r size _ url; do
    f=$DST/${url##*/}
    if [[ ! -f "$f" ]]; then
        echo "missing: ${url##*/}"; bad=$((bad + 1))
    elif [[ $(stat -c %s "$f") != "$size" ]]; then
        echo "size differs from the bucket: ${url##*/}"; bad=$((bad + 1))
    fi
done < <(grep 'gs://' "$DST/bucket_listing.txt")
[[ $bad -eq 0 ]] || { echo "$bad files missing or incomplete; submit again to fetch them"; exit 1; }
total=$(awk '/gs:\/\// {s += $1} END {printf "%.0f", s}' "$DST/bucket_listing.txt")
echo "$n files, $total bytes; every size matches the bucket listing ($(date '+%F %T'))" | tee "$DST/DOWNLOAD_COMPLETE"
