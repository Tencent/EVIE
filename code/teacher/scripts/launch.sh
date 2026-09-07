#!/usr/bin/env bash
# Train EVIE-8B (single 4096-d head). Requires DATA_ROOT, HARDNEG_ROOT.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${REPO:-$(cd "$ROOT/../.." && pwd)}"
EVIE_ROOT="${EVIE_ROOT:-$REPO}"
# shellcheck source=/dev/null
source "$REPO/code/shared/lib.sh"
evie_resolve_python
evie_pythonpath
evie_workdirs
export PYTHONPATH="$PYTHONPATH:$ROOT/scripts"
cd "$ROOT"
PY="$PYTHON"

RUN_NAME="${RUN_NAME:-evie-8b}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
HARDNEG_ROOT="${HARDNEG_ROOT:?set HARDNEG_ROOT}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5}"
COL_DIM="${COL_DIM:-4096}"
SOURCES="${SOURCES:-colpali_train_set vdr-multilingual-train VisRAG-Ret-Train-Synthetic-data VisRAG-Ret-Train-In-domain-data tatdqa_train tabfquad_train_set}"
BSZ="${BSZ:-2}"
EFF_BATCH="${EFF_BATCH:-512}"
MVT="${MVT:-1024}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-42}"
BIDIR="${BIDIR:-on}"
GRAD_CHECKPOINTING="${GRAD_CHECKPOINTING:-off}"
LR="${LR:-4.57e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.02}"
WARMUP_RATIO="${WARMUP_RATIO:-0.08}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.197}"
LOSS_TEMPERATURE="${LOSS_TEMPERATURE:-0.02}"
NUM_HARD_NEGS="${NUM_HARD_NEGS:-2}"
USE_HARDNEGATIVES="${USE_HARDNEGATIVES:-on}"
HARDNEG_IN_BATCH_WEIGHT="${HARDNEG_IN_BATCH_WEIGHT:-0.5}"
REPORT_TO="${REPORT_TO:-wandb,tensorboard}"
WANDB_PROJECT="${WANDB_PROJECT:-evie}"
WANDB_NAME="${WANDB_NAME:-$RUN_NAME}"
EVAL_AFTER_TRAIN="${EVAL_AFTER_TRAIN:-1}"
SKIP_IF_TRAINED="${SKIP_IF_TRAINED:-1}"
OVERWRITE="${OVERWRITE:-auto}"
TB_DIR="${TB_DIR:-$RUNS_DIR/$RUN_NAME/tensorboard}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
unset PYTHONHOME
export PATH="$(dirname "$PY"):$PATH"
evie_batch

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${RUN_NAME}_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[log] $LOG_FILE"
echo "[dist] ${NNODES}x${NPROC_PER_NODE} rank=$NODE_RANK ${MASTER_ADDR}:${MASTER_PORT}"

OUTPUT_DIR="${OUTPUT_DIR:-$RUNS_DIR/$RUN_NAME}"
WANDB_DIR="${WANDB_DIR:-$OUTPUT_DIR/wandb}"
SKIP_TRAIN=0
if [[ -f "$OUTPUT_DIR/adapter_model.safetensors" && -f "$OUTPUT_DIR/adapter_config.json" \
      && "$SKIP_IF_TRAINED" == "1" && -z "${RESUME:-}" && "${OVERWRITE:-0}" != "1" ]]; then
  SKIP_TRAIN=1
  echo "[train] skip: adapter already at $OUTPUT_DIR"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  evie_nccl
  export HARDNEG_SUBDIR="${HARDNEG_SUBDIR:-judged}"
  evie_wandb
  RESUME="${RESUME:-}"
  evie_prepare_output
  read -r -a SOURCE_ARGS <<< "$SOURCES"
  ARGS=(
    --base-model "$BASE_MODEL" --col-dim "$COL_DIM" --data-root "$DATA_ROOT"
    --output-dir "$OUTPUT_DIR" --sources "${SOURCE_ARGS[@]}"
    --epochs "$EPOCHS" --seed "$SEED"
    --per-device-batch-size "$BSZ" --grad-accum "$GRAD_ACCUM"
    --learning-rate "$LR" --weight-decay "$WEIGHT_DECAY" --warmup-ratio "$WARMUP_RATIO"
    --max-visual-tokens "$MVT" --dataloader-workers "${DL_WORKERS:-8}"
    --dataloader-prefetch-factor "${DL_PREFETCH:-4}"
    --bidirectional-attention "$BIDIR" --grad-checkpointing "$GRAD_CHECKPOINTING"
    --lora-r "$LORA_R" --lora-alpha "$LORA_ALPHA" --lora-dropout "$LORA_DROPOUT"
    --loss-temperature "$LOSS_TEMPERATURE"
    --num-hard-negs "$NUM_HARD_NEGS" --use-hardnegatives "$USE_HARDNEGATIVES"
    --hardneg-in-batch-weight "$HARDNEG_IN_BATCH_WEIGHT"
    --report-to "$REPORT_TO" --logging-dir "$TB_DIR" --run-name "$RUN_NAME"
    --hardneg-root "$HARDNEG_ROOT"
  )
  [[ -n "$RESUME" ]] && ARGS+=(--resume-from-checkpoint "$RESUME")
  [[ -n "${MAX_STEPS:-}" ]] && ARGS+=(--max-steps "$MAX_STEPS")
  [[ -n "${MAX_SAMPLES:-}" ]] && ARGS+=(--max-samples-per-source "$MAX_SAMPLES")
  echo "[train] $RUN_NAME seed=$SEED ${NNODES}x${NPROC_PER_NODE} global_batch=$ACTUAL_GLOBAL mvt=$MVT"
  "$PY" -m torch.distributed.run \
    --nnodes="$NNODES" --nproc_per_node="$NPROC_PER_NODE" --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" scripts/train.py "${ARGS[@]}"
fi

if [[ "$EVAL_AFTER_TRAIN" == "1" ]]; then
  export MASTER_PORT="${EVAL_MASTER_PORT:-29501}"
  export RUN_NAME NNODES NODE_RANK MASTER_ADDR NPROC_PER_NODE
  export EVAL_BASE_MODEL="${EVAL_BASE_MODEL:-$BASE_MODEL}"
  bash "$REPO/code/shared/eval_run.sh"
fi
echo "== complete: $RUN_NAME =="
