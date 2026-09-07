#!/usr/bin/env bash
# Train EVIE-8B: two arms (same recipe, different hard-negative shards), then mean-merge.
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

COL_DIM="${COL_DIM:-4096}"
SEED="${SEED:-42}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5}"
export MVT="${MVT:-1024}"
export EPOCHS="${EPOCHS:-1}"
export EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-1}"
DO_MERGE="${DO_MERGE:-1}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
WAIT_SECS="${WAIT_SECS:-10800}"

TAG="d${COL_DIM}_mvt${MVT}_s${SEED}"
ARM_A="${ARM_A:-evie-8b-${TAG}-judged}"
ARM_B="${ARM_B:-evie-8b-${TAG}-allpos}"
SOUP="${SOUP:-evie-8b-${TAG}}"

declare -A SUBDIR=(
  ["$ARM_A"]="${HARDNEG_A:-judged}"
  ["$ARM_B"]="${HARDNEG_B:-allpos}"
)

rc_all=0
for RUN_NAME in "$ARM_A" "$ARM_B"; do
  sub="${SUBDIR[$RUN_NAME]}"
  echo "============================================================"
  echo "[teacher] $RUN_NAME  (col_dim=$COL_DIM, hardneg=$sub)"
  echo "============================================================"
  if COL_DIM="$COL_DIM" RUN_NAME="$RUN_NAME" SEED="$SEED" \
     HARDNEG_SUBDIR="$sub" bash "$INNER"; then
    echo "[teacher] $RUN_NAME done"
  else
    rc=$?
    echo "[teacher][error] $RUN_NAME failed (exit=$rc)"
    rc_all=1
  fi
done

if [[ "$rc_all" != "0" ]]; then
  echo "[teacher] an arm failed; skip merge"
  exit $rc_all
fi

if [[ "$DO_MERGE" != "1" ]]; then
  echo "[teacher] DO_MERGE!=1; both arms finished, not merged"
  exit 0
fi

soup_ready() {
  local d="$1"
  [[ -f "$d/run_config.json" && -f "$d/config.json" ]] && \
    compgen -G "$d/model*.safetensors" >/dev/null
}

RUNS="$RUNS_DIR"
ALPHA="${ALPHA:-0.4}"
WEIGHT_B="$("$PYTHON" -c "print(round(1.0 - float('$ALPHA'), 4))")"
OUT="$RUNS/$SOUP"
if [[ "$NODE_RANK" == "0" ]]; then
  echo "============================================================"
  echo "[teacher] merge -> $SOUP (alpha=$ALPHA)"
  echo "============================================================"
  if soup_ready "$OUT"; then
    echo "[teacher] $OUT already complete, skip merge"
  elif [[ -d "$OUT" && -n "$(ls -A "$OUT" 2>/dev/null || true)" ]]; then
    echo "[teacher][error] $OUT is non-empty but incomplete; refusing to overwrite"
    echo fail > "${OUT}.failed"
    exit 1
  else
    rm -f "${OUT}.failed"
    export PYTHONPATH="$REPO/colpali:$REPO/code/shared${PYTHONPATH:+:$PYTHONPATH}"
    if ! "$PYTHON" "$REPO/code/shared/merge_seeds.py" \
      --base "$BASE_MODEL" \
      --adapters "$RUNS/$ARM_A" "$RUNS/$ARM_B" \
      --weights "$ALPHA" "$WEIGHT_B" \
      --output "$OUT" --dtype float32; then
      echo "[teacher][error] merge failed"
      echo fail > "${OUT}.failed"
      exit 1
    fi
  fi
else
  echo "[teacher] waiting for soup $OUT (merge runs on rank 0)"
  deadline=$((SECONDS + WAIT_SECS))
  while (( SECONDS < deadline )); do
    [[ -f "${OUT}.failed" ]] && { echo "[teacher][error] rank 0 reported merge failure"; exit 1; }
    soup_ready "$OUT" && break
    sleep 10
  done
  soup_ready "$OUT" || { echo "[teacher][error] timed out waiting for soup"; exit 1; }
fi

if [[ "$EVAL_AFTER_TRAIN" == "1" ]]; then
  echo "[teacher] eval soup: $SOUP"
  unset NNODES
  export RUN_NAME="$SOUP"
  export MASTER_PORT="${EVAL_MASTER_PORT:-29501}"
  bash "$SCRIPT_DIR/scripts/eval_run.sh" || { echo "[teacher][error] soup eval failed"; exit 1; }
fi
echo "[teacher] done: A=$ARM_A B=$ARM_B soup=$SOUP"
