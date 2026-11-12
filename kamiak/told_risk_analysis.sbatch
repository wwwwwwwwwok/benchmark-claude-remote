#!/bin/bash
# EXPLORATORY (user, 2026-11-11): Qwen3.6-27B's risk runs with one true quantity told in the question, against the main
# 27B run with thinking on (base, vlm_scores/qwen3.6-27b). speed = 31628444 (teacher's account, /data/lab/cejli/sy/
# speed_risk); accel = 31681661 (vlm_scores/qwen3.6-27b-accel). CPU only; no image is opened. Steps:
#  1 code/copy_renamed_run.py: the run's main file and forced answers copied with only the model renamed (<name>),
#    checked byte for byte against the source, which is only read;
#  2 in parallel: analyze_vlm_ssm.py, unchanged, on <out>/vlm_scores (base linked, the copy) with analyze_weather.sh's
#    arguments (--run main --scene_labels labels/scene_labels.csv --group_by weather,time_of_day --resample segment
#    --n_boot 2000, seed 20261021 by default; main --closing_min 0.1, sensitivity 0) -> <out>/main_1084 and
#    <out>/sensitivity_1123; code/extract_reasons.py for both models -> <out>/reasons_<model>.csv;
#  3 code/analyze_told_quantity.py on both subfolders: tables exploratory_<run>_*.csv, report exploratory_<run>_report.out
#    (the readings are in its docstring, fixed before the run).
# The accel run's words for the reasons are not decided: give them as WORDS=... in the environment.
# usage (from the repository root): sbatch kamiak/told_risk_analysis.sbatch speed|accel
#SBATCH --job-name=told-risk
#SBATCH --partition=kamiak
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/data/lab/cejli/sy/tmp_checks/told_risk/%x_%j.out
set -euo pipefail
RUN=$1
REPO=$(pwd)
[ -f "$REPO/code/analyze_told_quantity.py" ] || { echo "run it from the repository root" >&2; exit 1; }
PY=/data/lab/cejli/sy/conda_envs/waymo/bin/python
L=/data/lab/cejli/sy/waymo_v143
BASE=qwen3.6-27b
case $RUN in
    speed)
        SRC=/data/lab/cejli/sy/speed_risk/out/main_speed_qwen3.6-27b
        NAME=qwen3.6-27b-speed
        OUT=$L/analysis/speed_risk_27b
        WRAP=/data/lab/cejli/sy/speed_risk/code/run_vlm_speed.py          # the copy the job ran
        TOLD_CSV=/data/lab/cejli/sy/speed_risk/ego_speed_main.csv        # the csv the job read
        COL=v_ego
        WORDS='speed,m/s,meters per second,mph,km/h,fast,slow,slowly' ;;
    accel)
        SRC=$L/vlm_scores/qwen3.6-27b-accel/main_accel_qwen3.6-27b
        NAME=qwen3.6-27b-accel
        OUT=$L/analysis/accel_risk_27b
        WRAP=$REPO/code/run_vlm_accel.py
        TOLD_CSV=$L/quantities/truth_main.csv
        COL=a_ego
        WORDS=${WORDS:?give WORDS=... for the accel run} ;;
    *)
        echo "the first argument is speed or accel" >&2; exit 1 ;;
esac
PREFIX=exploratory_${RUN}_
pick() {
    if [ -f "$1.jsonl.gz" ]; then echo "$1.jsonl.gz"; elif [ -f "$1.jsonl" ]; then echo "$1.jsonl"
    else echo "no $1.jsonl(.gz)" >&2; return 1; fi
}
MAIN=$(pick "$SRC")
FORCED=$(pick "${SRC}_forced_answers")
[ ! -e "$OUT/main_1084" ] || { echo "$OUT/main_1084 exists already: give a new folder" >&2; exit 1; }
[ ! -e "$OUT/sensitivity_1123" ] || { echo "$OUT/sensitivity_1123 exists already: give a new folder" >&2; exit 1; }
echo "node $(hostname), start $(date '+%F %T'); run $RUN: $MAIN, $FORCED -> $OUT; repository at $(git -C "$REPO" rev-parse --short HEAD)"
sha256sum "$REPO"/code/{copy_renamed_run,extract_reasons,analyze_told_quantity,analyze_vlm_ssm,analyze_rules}.py "$WRAP" "$TOLD_CSV"
mkdir -p "$OUT/vlm_scores"
[ -L "$OUT/vlm_scores/$BASE" ] || ln -s "$L/vlm_scores/$BASE" "$OUT/vlm_scores/$BASE"

# 1
$PY -I code/copy_renamed_run.py "$MAIN" "$FORCED" "$BASE" "$NAME" "$OUT/vlm_scores/$NAME"
echo "step 1 done $(date '+%F %T')"

# 2
FSSM=$L/results_lead_final_excl/frame_ssm_validation_training.csv
COMMON=(--run main --scene_labels "$REPO/labels/scene_labels.csv" --group_by weather,time_of_day --resample segment
        --n_boot 2000)
$PY -I code/analyze_vlm_ssm.py "$OUT/vlm_scores" "$FSSM" "$OUT/main_1084" "${COMMON[@]}" --closing_min 0.1 \
    > "$OUT/main_1084.log" 2>&1 &
P1=$!
$PY -I code/analyze_vlm_ssm.py "$OUT/vlm_scores" "$FSSM" "$OUT/sensitivity_1123" "${COMMON[@]}" --closing_min 0 \
    > "$OUT/sensitivity_1123.log" 2>&1 &
P2=$!
$PY -I code/extract_reasons.py "$OUT/vlm_scores/$BASE" main "$OUT/reasons_$BASE.csv" > "$OUT/reasons_$BASE.log" 2>&1 &
P3=$!
$PY -I code/extract_reasons.py "$OUT/vlm_scores/$NAME" main "$OUT/reasons_$NAME.csv" > "$OUT/reasons_$NAME.log" 2>&1 &
P4=$!
RC=0
for p in $P1 $P2 $P3 $P4; do wait "$p" || RC=1; done
tail -n 2 "$OUT/main_1084.log" "$OUT/sensitivity_1123.log" "$OUT/reasons_$BASE.log" "$OUT/reasons_$NAME.log"
[ "$RC" -eq 0 ] || { echo "step 2 failed" >&2; exit 1; }
echo "step 2 done $(date '+%F %T')"

# 3
ARGS=("$NAME" "$BASE" "$OUT/reasons_$NAME.csv" "$OUT/reasons_$BASE.csv" "$WRAP" "$TOLD_CSV" "$COL"
      --words "$WORDS" --prefix "$PREFIX" --workers 4)
$PY -I code/analyze_told_quantity.py "$OUT/main_1084" "${ARGS[@]}" > "$OUT/main_1084_$RUN.log" 2>&1 &
P5=$!
$PY -I code/analyze_told_quantity.py "$OUT/sensitivity_1123" "${ARGS[@]}" > "$OUT/sensitivity_1123_$RUN.log" 2>&1 &
P6=$!
RC=0
wait "$P5" || RC=1
wait "$P6" || RC=1
tail -n 3 "$OUT/main_1084_$RUN.log" "$OUT/sensitivity_1123_$RUN.log"
echo "done $(date '+%F %T'), exit $RC"
exit "$RC"
