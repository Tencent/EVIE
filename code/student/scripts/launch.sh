#!/usr/bin/env bash
# Train EVIE-4.5B (Prefix-MRL + ARD). Requires DATA_ROOT, HARDNEG_ROOT, TEACHER_DIR.
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

RUN_NAME="${RUN_NAME:-evie-4.5b}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
HARDNEG_ROOT="${HARDNEG_ROOT:?set HARDNEG_ROOT}"
BASE_MODEL="${BASE_MODEL:-tencent/EVIE-Preview-4.5B}"
COL_DIM="${COL_DIM:-2048}"
SOURCES="${SOURCES:-colpali_train_set vdr-multilingual-train VisRAG-Ret-Train-Synthetic-data VisRAG-Ret-Train-In-domain-data tatdqa_train tabfquad_train_set}"
HEAD_DIMS="${HEAD_DIMS:-64,128,256,512,1024,2048}"
ANCHOR_DIM="${ANCHOR_DIM:-128}"
KD_DIMS="${KD_DIMS:-64,128,256,512,1024,2048}"
CALIBRATION_DIM="${CALIBRATION_DIM:-128}"
TEACHER_DIR="${TEACHER_DIR:-${EVIE_8B_DIR:-}}"
TEACHER_MD5="${TEACHER_MD5:-3f9a64a729d9277c20f038a1635203e5}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-0.13}"
STUDENT_TEMPERATURES="${STUDENT_TEMPERATURES:-64:0.13,128:0.13,256:0.13,512:0.13,1024:0.13,2048:0.13}"
RELATION_WEIGHT="${RELATION_WEIGHT:-1.0}"
MARGIN_WEIGHT="${MARGIN_WEIGHT:-0.25}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.25}"
COLUMN_WEIGHT="${COLUMN_WEIGHT:-1.0}"
CONFIDENCE_FLOOR="${CONFIDENCE_FLOOR:-0.1}"
TEACHER_WRONG_FACTOR="${TEACHER_WRONG_FACTOR:-0.25}"
HEAD_WEIGHTS="${HEAD_WEIGHTS:-}"
KD_HEAD_WEIGHTS="${KD_HEAD_WEIGHTS:-}"
KD_DIRECTIONS="${KD_DIRECTIONS:-both}"
KD_INCLUDE_HARDNEGS="${KD_INCLUDE_HARDNEGS:-on}"
ANCHOR_TEACHER="${ANCHOR_TEACHER:-on}"
TASK_CONSISTENT_BATCHES="${TASK_CONSISTENT_BATCHES:-on}"
GRADIENT_TARGET_RATIO="${GRADIENT_TARGET_RATIO:-0.5}"
GRADIENT_CALIBRATION_STEPS="${GRADIENT_CALIBRATION_STEPS:-100}"
GRADIENT_CALIBRATION_INTERVAL="${GRADIENT_CALIBRATION_INTERVAL:-10}"
GRADIENT_SCALE_MIN="${GRADIENT_SCALE_MIN:-0.05}"
GRADIENT_SCALE_MAX="${GRADIENT_SCALE_MAX:-20.0}"
GRADIENT_SCALE_EMA="${GRADIENT_SCALE_EMA:-0.9}"
GRADIENT_DIAGNOSTICS="${GRADIENT_DIAGNOSTICS:-on}"
GRADIENT_DIAGNOSTIC_STEPS="${GRADIENT_DIAGNOSTIC_STEPS:-100}"
GRADIENT_DIAGNOSTIC_INTERVAL="${GRADIENT_DIAGNOSTIC_INTERVAL:-10}"
HEAD_WARMUP_STEPS="${HEAD_WARMUP_STEPS:-100}"
BSZ="${BSZ:-2}"
EFF_BATCH="${EFF_BATCH:-512}"
MVT="${MVT:-1024}"
EPOCHS="${EPOCHS:-1}"
SEED="${SEED:-42}"
BIDIR="${BIDIR:-on}"
GRAD_CHECKPOINTING="${GRAD_CHECKPOINTING:-off}"
LR="${LR:-1.5e-5}"
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
[[ -n "$TEACHER_DIR" ]] || { echo "[fatal] set TEACHER_DIR or EVIE_8B_DIR"; exit 2; }

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
  export HARDNEG_SUBDIR="${HARDNEG_SUBDIR:-allpos}"
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
    --head-dims "$HEAD_DIMS" --anchor-dim "$ANCHOR_DIM"
    --kd-dims "$KD_DIMS" --calibration-dim "$CALIBRATION_DIM"
    --teacher-temperature "$TEACHER_TEMPERATURE"
    --student-temperatures "$STUDENT_TEMPERATURES"
    --relation-weight "$RELATION_WEIGHT" --margin-weight "$MARGIN_WEIGHT"
    --anchor-weight "$ANCHOR_WEIGHT" --column-weight "$COLUMN_WEIGHT"
    --confidence-floor "$CONFIDENCE_FLOOR" --teacher-wrong-factor "$TEACHER_WRONG_FACTOR"
    --kd-directions "$KD_DIRECTIONS" --kd-include-hardnegs "$KD_INCLUDE_HARDNEGS"
    --anchor-teacher "$ANCHOR_TEACHER" --task-consistent-batches "$TASK_CONSISTENT_BATCHES"
    --gradient-target-ratio "$GRADIENT_TARGET_RATIO"
    --gradient-calibration-steps "$GRADIENT_CALIBRATION_STEPS"
    --gradient-calibration-interval "$GRADIENT_CALIBRATION_INTERVAL"
    --gradient-scale-min "$GRADIENT_SCALE_MIN" --gradient-scale-max "$GRADIENT_SCALE_MAX"
    --gradient-scale-ema "$GRADIENT_SCALE_EMA"
    --gradient-diagnostics "$GRADIENT_DIAGNOSTICS"
    --gradient-diagnostic-steps "$GRADIENT_DIAGNOSTIC_STEPS"
    --gradient-diagnostic-interval "$GRADIENT_DIAGNOSTIC_INTERVAL"
    --head-warmup-steps "$HEAD_WARMUP_STEPS"
    --teacher-dir "$TEACHER_DIR"
  )
  [[ -n "$HEAD_WEIGHTS" ]] && ARGS+=(--head-weights "$HEAD_WEIGHTS")
  [[ -n "$KD_HEAD_WEIGHTS" ]] && ARGS+=(--kd-head-weights "$KD_HEAD_WEIGHTS")
  [[ -n "$TEACHER_MD5" ]] && ARGS+=(--teacher-md5 "$TEACHER_MD5")
  [[ -n "$RESUME" ]] && ARGS+=(--resume-from-checkpoint "$RESUME")
  [[ -n "${MAX_STEPS:-}" ]] && ARGS+=(--max-steps "$MAX_STEPS")
  [[ -n "${MAX_SAMPLES:-}" ]] && ARGS+=(--max-samples-per-source "$MAX_SAMPLES")
  echo "[train] $RUN_NAME prefixes=$HEAD_DIMS anchor=$ANCHOR_DIM teacher=$TEACHER_DIR global_batch=$ACTUAL_GLOBAL"
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
