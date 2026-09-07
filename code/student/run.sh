#!/usr/bin/env bash
# Train EVIE-4.5B: Prefix-MRL + EVIE-ARD from EVIE-8B, two arms, then mean-merge.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INNER="$SCRIPT_DIR/scripts/launch.sh"
[[ -x "$INNER" ]] || { echo "[fatal] missing launcher: $INNER"; exit 2; }

REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVIE_ROOT="${EVIE_ROOT:-$REPO}"
# shellcheck source=/dev/null
source "$REPO/code/shared/lib.sh"
evie_resolve_python
evie_workdirs

SEED="${SEED:-42}"
export MVT="${MVT:-1024}"
export EPOCHS="${EPOCHS:-1}"
export EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-1}"
export HEAD_DIMS="${HEAD_DIMS:-64,128,256,512,1024,2048}"
export TEACHER_DIR="${TEACHER_DIR:-${EVIE_8B_DIR:-}}"
export TEACHER_MD5="${TEACHER_MD5:-3f9a64a729d9277c20f038a1635203e5}"
export KD_DIMS="${KD_DIMS:-64,128,256,512,1024,2048}"
export CALIBRATION_DIM="${CALIBRATION_DIM:-128}"
export TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.13}"
export STUDENT_TEMPERATURES="${STUDENT_TEMPERATURES:-64:0.13,128:0.13,256:0.13,512:0.13,1024:0.13,2048:0.13}"
export RELATION_WEIGHT="${RELATION_WEIGHT:-1.0}"
export MARGIN_WEIGHT="${MARGIN_WEIGHT:-0.25}"
export ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.25}"
export COLUMN_WEIGHT="${COLUMN_WEIGHT:-1.0}"
export KD_DIRECTIONS="${KD_DIRECTIONS:-both}"
export KD_INCLUDE_HARDNEGS="${KD_INCLUDE_HARDNEGS:-on}"
export ANCHOR_TEACHER="${ANCHOR_TEACHER:-on}"
export TASK_CONSISTENT_BATCHES="${TASK_CONSISTENT_BATCHES:-on}"

ARMS="${ARMS:-a b}"
DO_MERGE="${DO_MERGE:-1}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
WAIT_SECS="${WAIT_SECS:-10800}"

TAG="mvt${MVT}_s${SEED}"
ARM_A="${ARM_A:-evie-4.5b-${TAG}-allpos}"
ARM_B="${ARM_B:-evie-4.5b-${TAG}-judged}"
SOUP="${SOUP:-evie-4.5b-${TAG}}"

declare -A SUBDIR=(
  ["$ARM_A"]="${HARDNEG_A:-allpos}"
  ["$ARM_B"]="${HARDNEG_B:-judged}"
)
declare -A ARM_NAME=(["a"]="$ARM_A" ["b"]="$ARM_B")

[[ -n "$TEACHER_DIR" ]] || { echo "[fatal] set TEACHER_DIR or EVIE_8B_DIR to the EVIE-8B checkpoint"; exit 2; }

rc_all=0
for arm in $ARMS; do
  RUN_NAME="${ARM_NAME[$arm]:-}"
  [[ -n "$RUN_NAME" ]] || { echo "[student][fatal] ARMS must be a and/or b, got: $arm"; exit 2; }
  sub="${SUBDIR[$RUN_NAME]}"
  echo "============================================================"
  echo "[student] $arm $RUN_NAME  (heads=$HEAD_DIMS, hardneg=$sub)"
  echo "============================================================"
  if RUN_NAME="$RUN_NAME" SEED="$SEED" HARDNEG_SUBDIR="$sub" bash "$INNER"; then
    echo "[student] $RUN_NAME done"
  else
    rc=$?
    echo "[student][error] $RUN_NAME failed (exit=$rc)"
    rc_all=1
  fi
done

if [[ "$rc_all" != "0" ]]; then
  echo "[student] an arm failed; skip merge"
  exit $rc_all
fi

if [[ "$DO_MERGE" != "1" ]]; then
  echo "[student] DO_MERGE!=1; not merged (default is one arm; set ARMS=\"a b\" DO_MERGE=1)"
  exit 0
fi

for arm_name in "$ARM_A" "$ARM_B"; do
  [[ -f "$RUNS_DIR/$arm_name/adapter_config.json" ]] ||
    { echo "[student][fatal] merge needs both adapters: missing runs/$arm_name/adapter_config.json"; exit 2; }
done

soup_ready() {
  local d="$1"
  [[ -f "$d/run_config.json" && -f "$d/config.json" ]] && \
    compgen -G "$d/model*.safetensors" >/dev/null
}

BASE_MODEL="${BASE_MODEL:-tencent/EVIE-Preview-4.5B}"
RUNS="$RUNS_DIR"
OUT="$RUNS/$SOUP"
if [[ "$NODE_RANK" == "0" ]]; then
  echo "============================================================"
  echo "[student] merge -> $SOUP"
  echo "============================================================"
  if soup_ready "$OUT"; then
    echo "[student] $OUT already complete, skip merge"
  elif [[ -d "$OUT" && -n "$(ls -A "$OUT" 2>/dev/null || true)" ]]; then
    echo "[student][error] $OUT is non-empty but incomplete; refusing to overwrite"
    echo fail > "${OUT}.failed"
    exit 1
  else
    rm -f "${OUT}.failed"
    export PYTHONPATH="$REPO/colpali:$REPO/code/shared${PYTHONPATH:+:$PYTHONPATH}"
    if ! "$PYTHON" "$REPO/code/shared/merge_seeds.py" \
      --base "$BASE_MODEL" \
      --adapters "$RUNS/$ARM_A" "$RUNS/$ARM_B" \
      --output "$OUT" --dtype float32; then
      echo "[student][error] merge failed"
      echo fail > "${OUT}.failed"
      exit 1
    fi
  fi
else
  echo "[student] waiting for soup $OUT (merge runs on rank 0)"
  deadline=$((SECONDS + WAIT_SECS))
  while (( SECONDS < deadline )); do
    [[ -f "${OUT}.failed" ]] && { echo "[student][error] rank 0 reported merge failure"; exit 1; }
    soup_ready "$OUT" && break
    sleep 10
  done
  soup_ready "$OUT" || { echo "[student][error] timed out waiting for soup"; exit 1; }
fi

if [[ "$EVAL_AFTER_TRAIN" == "1" ]]; then
  echo "[student] eval soup (per head): $SOUP"
  unset NNODES
  export RUN_NAME="$SOUP"
  export MASTER_PORT="${EVAL_MASTER_PORT:-29501}"
  export EVAL_BASE_MODEL="${EVAL_BASE_MODEL:-$BASE_MODEL}"
  bash "$SCRIPT_DIR/scripts/eval_run.sh" || { echo "[student][error] soup eval failed"; exit 1; }
fi
echo "[student] done: a=$ARM_A b=$ARM_B soup=$SOUP"
