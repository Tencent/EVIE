#!/usr/bin/env bash
# Eval one trained run into runs/$RUN_NAME/eval/ (or eval/d<k>/ for Prefix-MRL).
# Resume by default. Full redo: EVAL_OVERWRITE=1.
set -euo pipefail

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
EVIE_ROOT="${EVIE_ROOT:-$REPO}"
# shellcheck source=/dev/null
source "$REPO/code/shared/lib.sh"
evie_resolve_python
evie_pythonpath
evie_workdirs
unset PYTHONHOME
export PATH="$(dirname "$PYTHON"):$PATH"
PY="$PYTHON"

RUN_NAME="${RUN_NAME:?set RUN_NAME}"
mkdir -p "$LOG_DIR"
LOG_NODE="${NODE_RANK:-${RANK:-0}}"
LOG_FILE="$LOG_DIR/eval_${RUN_NAME}_node${LOG_NODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[log] $LOG_FILE"

EVAL_BATCH="${EVAL_BATCH:-8}"
EVAL_K="${EVAL_K:-1,5,10}"
EVAL_MAX_QUERIES="${EVAL_MAX_QUERIES:-0}"
EVAL_MAX_DOCS="${EVAL_MAX_DOCS:-0}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
EVAL_OVERWRITE="${EVAL_OVERWRITE:-0}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"
EVAL_DDP_TIMEOUT_S="${EVAL_DDP_TIMEOUT_S:-21600}"
EVAL_FINALIZE_TIMEOUT_S="${EVAL_FINALIZE_TIMEOUT_S:-21600}"
evie_nccl
NPROC_PER_NODE="${NPROC_PER_NODE:-$("$PY" -c 'import torch; print(torch.cuda.device_count())')}"

if [[ -n "${MODEL_DIR:-}" ]]; then
  ADAPTER_DIR="$MODEL_DIR"
else
  ADAPTER_DIR="$RUNS_DIR/$RUN_NAME"
fi
HAS_LORA=0
HAS_FULL=0
[[ -f "$ADAPTER_DIR/adapter_model.safetensors" && -f "$ADAPTER_DIR/adapter_config.json" ]] && HAS_LORA=1
[[ -f "$ADAPTER_DIR/config.json" ]] && compgen -G "$ADAPTER_DIR/model*.safetensors" >/dev/null && HAS_FULL=1
if [[ "$HAS_LORA" != "1" && "$HAS_FULL" != "1" ]]; then
  echo "[fatal] no LoRA adapter or full model under $ADAPTER_DIR" >&2
  exit 1
fi

EVAL_BASE_MODEL="${EVAL_BASE_MODEL:-Qwen/Qwen3.5}"
if [[ "$HAS_FULL" == "1" ]]; then
  EVAL_BASE_MODEL="$ADAPTER_DIR"
fi
if [[ "$HAS_LORA" == "1" ]]; then
  TRAINED_ON="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("base_model_name_or_path") or "")' "$ADAPTER_DIR/adapter_config.json" 2>/dev/null || true)"
  if [[ -n "$TRAINED_ON" ]]; then
    want="$(readlink -f "$TRAINED_ON" 2>/dev/null || echo "$TRAINED_ON")"
    got="$(readlink -f "$EVAL_BASE_MODEL" 2>/dev/null || echo "$EVAL_BASE_MODEL")"
    if [[ "$want" != "$got" ]]; then
      echo "[fatal] adapter trained on $want but EVAL_BASE_MODEL=$got" >&2
      exit 2
    fi
  fi
fi

HEAD_DIMS=""
if [[ -f "$ADAPTER_DIR/run_config.json" ]]; then
  HEAD_DIMS="$("$PY" -c 'import json,sys;d=json.load(open(sys.argv[1])).get("head_dims") or [];print(",".join(str(int(x)) for x in d))' "$ADAPTER_DIR/run_config.json" 2>/dev/null || true)"
fi
if [[ -z "$HEAD_DIMS" && -f "$ADAPTER_DIR/config.json" ]]; then
  HEAD_DIMS="$("$PY" -c 'import json,sys;d=json.load(open(sys.argv[1])).get("head_dims") or [];print(",".join(str(int(x)) for x in d))' "$ADAPTER_DIR/config.json" 2>/dev/null || true)"
fi
[[ -n "${EVAL_HEAD_DIMS:-}" ]] && HEAD_DIMS="$EVAL_HEAD_DIMS"

if [[ ! -f "$ADAPTER_DIR/run_config.json" ]]; then
  EVAL_MVT="${EVAL_MVT:-1024}"
  BIDIR="${BIDIR:-on}"
fi

EVAL_OUT_ROOT="${EVAL_OUT:-$RUNS_DIR/$RUN_NAME/eval}"
mkdir -p "$EVIE_TMP"

echo "[eval] tasks under $EVAL_ROOT"
"$PY" - <<PY
from collections import Counter
from eval import discover_tasks
tasks = discover_tasks("${EVAL_ROOT}")
print("[eval]", dict(Counter(t["dataset"] for t in tasks)), "total", len(tasks))
PY

CONTRACT_ARGS=()
[[ -n "${EVAL_MVT:-}" ]] && CONTRACT_ARGS+=(--max-visual-tokens "$EVAL_MVT" --allow-config-override)
[[ -n "${BIDIR:-}" ]] && CONTRACT_ARGS+=(--bidirectional-attention "$BIDIR" --allow-config-override)
[[ -n "${EVAL_DATASETS:-}" ]] && CONTRACT_ARGS+=(--datasets "$EVAL_DATASETS")
MODE_ARGS=(--resume)
[[ "$EVAL_OVERWRITE" == "1" ]] && MODE_ARGS=(--overwrite-output)

run_one_head() {
  local head="$1" out="$2" label token ready head_args=()
  label="${head:-single}"
  [[ -n "$head" ]] && head_args=(--head-dim "$head")
  token="${MASTER_ADDR}_${MASTER_PORT}_${RUN_NAME}_eval_${label}"
  ready="$EVIE_TMP/eval_${RUN_NAME}_${label}.ready"

  if [[ "$NODE_RANK" == "0" ]]; then
    rm -f "$ready"
    if [[ "$EVAL_OVERWRITE" == "1" && -d "$out" && -n "$(ls -A "$out" 2>/dev/null || true)" ]]; then
      mv "$out" "${out}_archive_$(date +%Y%m%d_%H%M%S)"
    fi
    mkdir -p "$out"
    printf '%s\n' "$token" > "$ready"
  else
    local r=""
    for _ in $(seq 1 600); do
      r=""
      IFS= read -r r < "$ready" || true
      [[ "$r" == "$token" ]] && break
      sleep 1
    done
    [[ "$r" == "$token" ]] || { echo "[fatal] timed out waiting for rank0 eval setup"; return 2; }
  fi

  echo "[eval] run=$RUN_NAME head=$label nodes=${NNODES}x${NPROC_PER_NODE} batch=$EVAL_BATCH out=$out"
  "$PY" -m torch.distributed.run \
    --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" \
    "$REPO/code/shared/eval.py" \
    --base-model "$EVAL_BASE_MODEL" --adapter-dir "$ADAPTER_DIR" \
    --eval-root "$EVAL_ROOT" --output-dir "$out" \
    --embed-batch "$EVAL_BATCH" --num-workers "$EVAL_WORKERS" \
    --ks "$EVAL_K" --max-queries "$EVAL_MAX_QUERIES" --max-docs "$EVAL_MAX_DOCS" \
    --run-name "$RUN_NAME" "${MODE_ARGS[@]}" "${head_args[@]}" "${CONTRACT_ARGS[@]}"

  if [[ "$NODE_RANK" != "0" ]]; then
    local deadline=$((SECONDS + EVAL_FINALIZE_TIMEOUT_S))
    while [[ ! -f "$out/summary.json" && "$SECONDS" -lt "$deadline" ]]; do sleep 2; done
  fi
  [[ -f "$out/summary.json" ]] || { echo "[fatal] summary.json missing for head=$label"; return 2; }
  "$PY" - "$out/summary.json" "$label" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"[eval][{sys.argv[2]}] status={s.get('status')} "
      f"{s.get('completed_tasks')}/{s.get('expected_tasks')} headline={s.get('headline')}")
if s.get("n_failed"):
    sys.exit(2)
PY
}

if [[ -z "$HEAD_DIMS" ]]; then
  run_one_head "" "$EVAL_OUT_ROOT"
  echo "== eval done: $RUN_NAME -> $EVAL_OUT_ROOT/summary.json =="
else
  echo "[eval] heads $HEAD_DIMS"
  IFS=',' read -r -a HEAD_LIST <<< "$HEAD_DIMS"
  FAILED=()
  for head in "${HEAD_LIST[@]}"; do
    head="${head// /}"
    [[ -n "$head" ]] || continue
    run_one_head "$head" "$EVAL_OUT_ROOT/d$head" || FAILED+=("$head")
  done
  if [[ "$NODE_RANK" == "0" ]]; then
    "$PY" "$REPO/code/shared/aggregate_heads.py" --eval-root "$EVAL_OUT_ROOT" --heads "$HEAD_DIMS" || true
  fi
  if (( ${#FAILED[@]} > 0 )); then
    echo "[fatal] heads failed: ${FAILED[*]}"
    exit 2
  fi
  echo "== eval done: $RUN_NAME -> $EVAL_OUT_ROOT/{d*,summary_heads.json} =="
fi
