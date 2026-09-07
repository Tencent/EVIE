#!/usr/bin/env bash
# Score the bundled 8-page demo with the checkpoint in this repo.
# Uses every visible GPU (torch.cuda.device_count()).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVIE_ROOT="${EVIE_ROOT:-$REPO}"
# shellcheck source=/dev/null
source "$REPO/code/shared/lib.sh"
evie_resolve_python
PY="$PYTHON"

if [[ ! -f "$REPO/examples/demo/demo/evie_pages/data/test-00000-of-00001.parquet" ]]; then
  echo "[demo] building parquet + PNG pages"
  "$PY" "$REPO/examples/demo/build.py"
fi

export MODEL_DIR="${MODEL_DIR:-$REPO}"
export RUN_NAME="${RUN_NAME:-demo}"
export EVAL_ROOT="$REPO/examples/demo"
export EVAL_DATASETS=demo
export EVAL_MVT="${EVAL_MVT:-1024}"
export BIDIR="${BIDIR:-on}"
export EVAL_OVERWRITE="${EVAL_OVERWRITE:-1}"
export EVAL_OUT="${EVAL_OUT:-$EVIE_ROOT/runs/$RUN_NAME/eval}"
export EVAL_BATCH="${EVAL_BATCH:-2}"
export EVAL_WORKERS="${EVAL_WORKERS:-2}"
export MASTER_PORT="${MASTER_PORT:-29721}"

HEAD="$("$PY" -c "import json; d=json.load(open('$MODEL_DIR/config.json')); hs=d.get('head_dims') or []; print(max(hs) if hs else '')")"
if [[ -n "$HEAD" ]]; then
  export EVAL_HEAD_DIMS="${HEAD_DIM:-$HEAD}"
  echo "[demo] Matryoshka head d$EVAL_HEAD_DIMS"
fi

echo "[demo] MODEL_DIR=$MODEL_DIR  EVAL_ROOT=$EVAL_ROOT"
bash "$REPO/code/shared/eval_run.sh"
echo "[demo] summary: $EVAL_OUT/summary.json"
if [[ -n "${EVAL_HEAD_DIMS:-}" ]]; then
  echo "[demo] summary: $EVAL_OUT/d$EVAL_HEAD_DIMS/summary.json"
fi
