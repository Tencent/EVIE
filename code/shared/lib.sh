# Shared helpers. Source after REPO / EVIE_ROOT are set.
evie_resolve_python() {
  if [[ -z "${PYTHON:-}" ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
      PYTHON="$VIRTUAL_ENV/bin/python"
    else
      PYTHON="$(command -v python3)"
    fi
  fi
  [[ -x "$PYTHON" ]] || { echo "[fatal] set PYTHON or VIRTUAL_ENV"; exit 2; }
}

evie_pythonpath() {
  export PYTHONPATH="$REPO/colpali:$REPO/code/shared${PYTHONPATH:+:$PYTHONPATH}"
}

evie_real() {
  local p="$1"
  if command -v realpath >/dev/null 2>&1; then
    realpath -m "$p" 2>/dev/null || echo "$p"
  else
    echo "$p"
  fi
}

# Refuse dumps inside the venv (the old $EVIE_ROOT/env footgun).
evie_forbid_venv_path() {
  local path="$1" label="$2"
  local child parent roots=() p
  child="$(evie_real "$path")"
  [[ -n "${VIRTUAL_ENV:-}" ]] && roots+=("$(evie_real "$VIRTUAL_ENV")")
  roots+=("$(evie_real "$EVIE_ROOT/env")")
  roots+=("$(evie_real "$EVIE_ROOT/venv")")
  roots+=("$(evie_real "$EVIE_ROOT/.venv")")
  roots+=("$(evie_real "$EVIE_ROOT/train-env")")
  if [[ -n "${PYTHON:-}" && -x "$PYTHON" ]]; then
    parent="$(cd "$(dirname "$PYTHON")/.." && pwd)"
    [[ -f "$parent/pyvenv.cfg" ]] && roots+=("$(evie_real "$parent")")
  fi
  for p in "${roots[@]}"; do
    [[ -n "$p" && "$p" != "/" ]] || continue
    if [[ "$child" == "$p" || "$child" == "$p"/* ]]; then
      echo "[fatal] $label cannot sit inside the Python env ($p): $path" >&2
      echo "[fatal] checkpoints -> \$RUNS_DIR (\$EVIE_ROOT/runs); caches -> \$EVIE_ROOT/.cache" >&2
      exit 2
    fi
  done
}

# Work product roots. All gitignored. Never under env/.
evie_workdirs() {
  export RUNS_DIR="${RUNS_DIR:-$EVIE_ROOT/runs}"
  export LOG_DIR="${LOG_DIR:-$EVIE_ROOT/logs}"
  export EVAL_ROOT="${EVAL_ROOT:-$EVIE_ROOT/data}"
  export EVIE_TMP="${EVIE_TMP:-$EVIE_ROOT/.tmp}"
  export CACHE_DIR="${CACHE_DIR:-$EVIE_ROOT/.cache}"
  export HF_HOME="${HF_HOME:-$CACHE_DIR/huggingface}"
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
  export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
  export TORCH_HOME="${TORCH_HOME:-$CACHE_DIR/torch}"
  evie_forbid_venv_path "$RUNS_DIR" "RUNS_DIR"
  evie_forbid_venv_path "$LOG_DIR" "LOG_DIR"
  evie_forbid_venv_path "$EVAL_ROOT" "EVAL_ROOT"
  evie_forbid_venv_path "$CACHE_DIR" "CACHE_DIR"
  evie_forbid_venv_path "$HF_HOME" "HF_HOME"
  mkdir -p "$RUNS_DIR" "$LOG_DIR" "$EVIE_TMP" "$CACHE_DIR"
}

evie_nccl() {
  evie_workdirs
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
  export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-480}"
  export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
  export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
  export TORCH_FR_BUFFER_SIZE="${TORCH_FR_BUFFER_SIZE:-2000}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
}

evie_batch() {
  NPROC_PER_NODE="${NPROC_PER_NODE:-$("$PYTHON" -c 'import torch; print(torch.cuda.device_count())')}"
  TOTAL_GPUS=$((NNODES * NPROC_PER_NODE))
  MICRO_GLOBAL=$((TOTAL_GPUS * BSZ))
  GRAD_ACCUM="${GRAD_ACCUM:-$((EFF_BATCH / MICRO_GLOBAL))}"
  ACTUAL_GLOBAL=$((MICRO_GLOBAL * GRAD_ACCUM))
}

evie_wandb() {
  if [[ ",$REPORT_TO," != *",wandb,"* ]]; then
    return
  fi
  export WANDB_PROJECT WANDB_NAME WANDB_DIR
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[warn] WANDB_API_KEY absent; wandb offline"
    export WANDB_MODE=offline
  else
    export WANDB_MODE="${WANDB_MODE:-online}"
  fi
}

# Rank 0 prepares OUTPUT_DIR; other nodes wait on a shared ready token.
evie_prepare_output() {
  local token="${MASTER_ADDR}_${MASTER_PORT}_${RUN_NAME}"
  local ready="$EVIE_TMP/launch_${RUN_NAME}.ready"
  evie_forbid_venv_path "$OUTPUT_DIR" "OUTPUT_DIR"
  mkdir -p "$EVIE_TMP" "$RUNS_DIR"
  if [[ "$NODE_RANK" == "0" ]]; then
    rm -f "$ready"
    if [[ -d "$OUTPUT_DIR" && -n "$(ls -A "$OUTPUT_DIR" 2>/dev/null || true)" && -z "${RESUME:-}" ]]; then
      if [[ "${OVERWRITE:-auto}" == "0" ]]; then
        echo "[fatal] non-empty output: $OUTPUT_DIR (OVERWRITE=auto/1)"
        printf 'ERROR:%s\n' "$token" > "$ready"
        exit 2
      fi
      local archive="${OUTPUT_DIR}_archive_$(date +%Y%m%d_%H%M%S)"
      mv "$OUTPUT_DIR" "$archive"
      echo "[output] archived -> $archive"
    fi
    mkdir -p "$OUTPUT_DIR"
    printf '%s\n' "$token" > "$ready"
  else
    local ready_val=""
    for _ in $(seq 1 300); do
      ready_val=""
      IFS= read -r ready_val < "$ready" || true
      [[ "$ready_val" == "$token" ]] && break
      [[ "$ready_val" == "ERROR:$token" ]] && { echo "[fatal] rank0 rejected output setup"; exit 2; }
      sleep 1
    done
    [[ "$ready_val" == "$token" ]] || { echo "[fatal] timed out waiting for rank0 output setup"; exit 2; }
  fi
  export EVIE_OUTPUT_PREPARED=1
}
