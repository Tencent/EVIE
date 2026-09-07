#!/usr/bin/env bash
# HAC on EVIE-4.5B: shipped SKU is d64 × K32 (HEAD_DIM / BUDGET in common.py).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVIE_ROOT="${EVIE_ROOT:-$REPO}"
# shellcheck source=/dev/null
source "$REPO/code/shared/lib.sh"
evie_resolve_python
evie_pythonpath
evie_workdirs
PY="$PYTHON"

export PATH="$(dirname "$PY"):$PATH"
unset PYTHONHOME
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export NCCL_IB_DISABLE=1
export EVAL_DDP_TIMEOUT_S="${EVAL_DDP_TIMEOUT_S:-21600}"
export EVAL_FINALIZE_TIMEOUT_S="${EVAL_FINALIZE_TIMEOUT_S:-21600}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-480}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"

MODEL_DIR="${MODEL_DIR:-${EVIE_4_5B_DIR:-}}"
[[ -n "$MODEL_DIR" && -f "$MODEL_DIR/model.safetensors" ]] || {
  echo "[fatal] set MODEL_DIR or EVIE_4_5B_DIR to an EVIE-4.5B checkpoint"; exit 2
}

read -r HEAD_DIM BUDGET WEIGHT < <("$PY" -c "from common import HEAD_DIM, BUDGET, WEIGHT; print(HEAD_DIM, BUDGET, WEIGHT)")
RUN_NAME="${RUN_NAME:-evie-compress}"
DATASETS="${DATASETS:-}"
MAX_DOCS="${MAX_DOCS:-0}"
MAX_QUERIES="${MAX_QUERIES:-0}"
EXPECTED_TASKS="${EXPECTED_TASKS:-138}"

RUN_DIR="${RUN_DIR:-$RUNS_DIR/$RUN_NAME}"
evie_forbid_venv_path "$RUN_DIR" "RUN_DIR"
DUMP_DIR="$RUN_DIR/dump"
CLUSTER_DIR="$RUN_DIR/cluster"
EVAL_DIR="$RUN_DIR/eval"
EVAL_BATCH="${EVAL_BATCH:-8}"
EVAL_WORKERS="${EVAL_WORKERS:-8}"
CLUSTER_WORKERS="${CLUSTER_WORKERS:-128}"
SKIP_DUMP="${SKIP_DUMP:-0}"
SKIP_RAW="${SKIP_RAW:-0}"
SKIP_CLUSTER="${SKIP_CLUSTER:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29711}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$("$PY" -c 'import torch; print(torch.cuda.device_count())')}"

mkdir -p "$LOG_DIR" "$RUN_DIR"
LOG_NODE="${NODE_RANK}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_NAME}_node${LOG_NODE}_$(date +%Y%m%d_%H%M%S).log}"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[log] $LOG_FILE"
echo "[compress] run=$RUN_NAME soup=$(basename "$MODEL_DIR") d=$HEAD_DIM k=$BUDGET w=$WEIGHT"
echo "[compress] topology=${NNODES}x${NPROC_PER_NODE} dump=$DUMP_DIR"

cd "$SCRIPT_DIR"

run_ddp() {
  local script="$1"; shift
  local port="$1"; shift
  echo "[ddp] $script port=$port $*"
  "$PY" -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$port" \
    "$script" "$@"
}

dump_args=(
  --eval-root "$EVAL_ROOT"
  --dump-dir "$DUMP_DIR"
  --embed-batch "$EVAL_BATCH"
  --num-workers "$EVAL_WORKERS"
  --max-queries "$MAX_QUERIES"
  --max-docs "$MAX_DOCS"
)
[[ -n "$DATASETS" ]] && dump_args+=(--datasets "$DATASETS")

if [[ "$SKIP_DUMP" != "1" ]]; then
  run_ddp dump.py "$MASTER_PORT" "${dump_args[@]}"
  "$PY" - "$DUMP_DIR/DONE.json" "$EXPECTED_TASKS" "$HEAD_DIM" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
want = int(sys.argv[2])
head = int(sys.argv[3])
print("[dump] DONE", d.get("status"), d.get("completed_tasks"), "/", d.get("expected_tasks"), "unique_corpora", d.get("unique_corpora"))
if d.get("status") != "complete" or int(d.get("completed_tasks") or 0) != int(d.get("expected_tasks") or -1):
    sys.exit(2)
if int(d.get("expected_tasks") or 0) != want:
    print("[fatal] expected_tasks mismatch", d.get("expected_tasks"), "want", want, file=sys.stderr)
    sys.exit(2)
if int(d.get("head_dim") or 0) != head:
    print("[fatal] dump head_dim", d.get("head_dim"), "want", head, file=sys.stderr)
    sys.exit(2)
PY
else
  echo "[compress] SKIP_DUMP=1"
fi

CLUSTER_PID=""
cluster_cleanup() {
  if [[ -n "${CLUSTER_PID:-}" ]] && kill -0 "$CLUSTER_PID" 2>/dev/null; then
    echo "[compress] stopping background cluster pid=$CLUSTER_PID"
    kill "$CLUSTER_PID" 2>/dev/null || true
    wait "$CLUSTER_PID" 2>/dev/null || true
  fi
}
trap cluster_cleanup EXIT

n_dump_corpora="$("$PY" -c "from pathlib import Path; print(len(list(Path(r'$DUMP_DIR').joinpath('docs').glob('*.pt'))))")"
WTAG="$("$PY" -c "from common import w_tag; print(w_tag())")"

if [[ "$SKIP_CLUSTER" != "1" ]]; then
  echo "[compress] cluster K=$BUDGET workers=$CLUSTER_WORKERS"
  "$PY" cluster.py --dump-dir "$DUMP_DIR" --cluster-root "$CLUSTER_DIR" --workers "$CLUSTER_WORKERS" &
  CLUSTER_PID=$!
else
  echo "[compress] SKIP_CLUSTER=1"
fi

if [[ "$SKIP_RAW" != "1" ]]; then
  run_ddp eval_index.py "$((MASTER_PORT + 1))" \
    --dump-dir "$DUMP_DIR" \
    --output-dir "$EVAL_DIR/raw" \
    --mode raw \
    --run-name "$RUN_NAME" \
    --resume
else
  echo "[compress] SKIP_RAW=1"
fi

eval_summary_complete() {
  "$PY" - "$1" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file():
    sys.exit(1)
d = json.load(p.open())
if d.get("status") == "complete" and int(d.get("completed_tasks") or 0) == int(d.get("expected_tasks") or -1) and d.get("expected_tasks"):
    sys.exit(0)
sys.exit(1)
PY
}

if [[ "$SKIP_EVAL" != "1" ]]; then
  if [[ "$SKIP_CLUSTER" != "1" ]]; then
    done="$CLUSTER_DIR/$WTAG/DONE.json"
    echo "[compress] waiting for $done"
    while [[ ! -f "$done" ]]; do
      if [[ -n "${CLUSTER_PID:-}" ]] && ! kill -0 "$CLUSTER_PID" 2>/dev/null; then
        rc=0
        wait "$CLUSTER_PID" || rc=$?
        CLUSTER_PID=""
        if [[ ! -f "$done" ]]; then
          echo "[fatal] cluster exited rc=$rc before $done" >&2
          exit 2
        fi
        break
      fi
      if [[ -z "${CLUSTER_PID:-}" && ! -f "$done" ]]; then
        echo "[fatal] no cluster process and missing $done" >&2
        exit 2
      fi
      sleep 5
    done
    if [[ -n "${CLUSTER_PID:-}" ]]; then
      wait "$CLUSTER_PID"
      CLUSTER_PID=""
    fi
    "$PY" - "$done" "$n_dump_corpora" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
want, got = int(sys.argv[2]), int(d.get("n_corpora") or 0)
if got != want:
    print(f"[fatal] {sys.argv[1]} n_corpora={got} want={want}", file=sys.stderr)
    sys.exit(2)
print(f"[compress] cluster ready n_corpora={got}", flush=True)
PY
  fi
  if eval_summary_complete "$EVAL_DIR/k${BUDGET}/summary.json"; then
    echo "[compress] skip eval already complete"
  else
    echo "[compress] eval independent K=$BUDGET w=$WEIGHT"
    run_ddp eval_index.py "$((MASTER_PORT + 2))" \
      --dump-dir "$DUMP_DIR" \
      --cluster-root "$CLUSTER_DIR" \
      --output-dir "$EVAL_DIR/k${BUDGET}" \
      --mode k \
      --w "$WEIGHT" \
      --budget "$BUDGET" \
      --run-name "$RUN_NAME" \
      --resume
  fi
else
  echo "[compress] SKIP_EVAL=1"
  if [[ -n "${CLUSTER_PID:-}" ]]; then
    wait "$CLUSTER_PID"
    CLUSTER_PID=""
  fi
fi

if [[ -n "${CLUSTER_PID:-}" ]]; then
  wait "$CLUSTER_PID"
  CLUSTER_PID=""
fi
trap - EXIT

"$PY" ladder.py --eval-root "$EVAL_DIR"
echo "== compress done: $EVAL_DIR/ladder.json =="
echo "[log] $LOG_FILE"
