#!/usr/bin/env bash
# Weighted merge of two LoRA adapter runs, then optional eval of each soup.
# ARM_A / ARM_B are directory names under $RUNS_DIR (or absolute paths).
# α is the weight on ARM_A; ARM_B gets 1-α.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVIE_ROOT="${EVIE_ROOT:-$REPO}"
# shellcheck source=/dev/null
source "$REPO/code/shared/lib.sh"
evie_resolve_python
evie_workdirs
MERGE="$REPO/code/shared/merge_seeds.py"
EVAL_SH="$REPO/code/shared/eval_run.sh"
[[ -f "$MERGE" ]] || { echo "[fatal] missing $MERGE"; exit 2; }
[[ -x "$EVAL_SH" ]] || { echo "[fatal] missing $EVAL_SH"; exit 2; }

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5}"
RUNS="$RUNS_DIR"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
EVAL_AFTER="${EVAL_AFTER:-1}"
DTYPE="${DTYPE:-float32}"
WAIT_SECS="${WAIT_SECS:-10800}"
ALPHAS="${ALPHAS:-0.4}"
PREFIX="${PREFIX:-evie-8b}"

resolve_arm() {
  local x="$1"
  if [[ -d "$x" ]]; then
    echo "$x"
  else
    echo "$RUNS/$x"
  fi
}

COL_DIM="${COL_DIM:-4096}"
MVT="${MVT:-1024}"
SEED="${SEED:-42}"
TAG="d${COL_DIM}_mvt${MVT}_s${SEED}"

ARM_A="$(resolve_arm "${ARM_A:-evie-8b-${TAG}-judged}")"
ARM_B="$(resolve_arm "${ARM_B:-evie-8b-${TAG}-allpos}")"

has_lora() {
  local d="$1"
  [[ -f "$d/run_config.json" && -f "$d/adapter_config.json" && -f "$d/adapter_model.safetensors" ]]
}

soup_ready() {
  local d="$1"
  [[ -f "$d/run_config.json" && -f "$d/config.json" ]] && \
    compgen -G "$d/model*.safetensors" >/dev/null
}

eval_done() {
  local d="$1"
  [[ -f "$d/eval/summary.json" ]] || return 1
  "$PYTHON" -c '
import json,sys
p=sys.argv[1]
d=json.load(open(p))
ok=d.get("status")=="complete" and int(d.get("completed_tasks") or 0)==int(d.get("expected_tasks") or -1)
sys.exit(0 if ok else 1)
' "$d/eval/summary.json"
}

pct_of() {
  "$PYTHON" -c "print(int(round(float('$1')*100)))"
}

w_b_of() {
  "$PYTHON" -c "print(f'{1.0-float(\"$1\"):.6f}')"
}

run_name_of() {
  echo "${PREFIX}-a$(pct_of "$1")"
}

echo "============================================================"
echo "[merge-alpha] A=$ARM_A  B=$ARM_B  α=weight on A"
echo "[merge-alpha] ALPHAS=$ALPHAS  prefix=$PREFIX"
echo "============================================================"

missing=0
if has_lora "$ARM_A"; then
  echo "[merge-alpha] ready  $ARM_A"
else
  echo "[merge-alpha] missing LoRA $ARM_A"
  missing=1
fi
if has_lora "$ARM_B"; then
  echo "[merge-alpha] ready  $ARM_B"
else
  echo "[merge-alpha] missing LoRA $ARM_B"
  missing=1
fi


[[ "$missing" == "0" ]] || { echo "[merge-alpha][error] both LoRA adapters required"; exit 2; }

do_merge() {
  local a="$1"
  local name out wb
  name="$(run_name_of "$a")"
  out="$RUNS/$name"
  wb="$(w_b_of "$a")"
  if soup_ready "$out"; then
    echo "[merge-alpha] $out already complete, skip"
    return 0
  fi
  if [[ -d "$out" && -n "$(ls -A "$out" 2>/dev/null || true)" ]]; then
    echo "[merge-alpha][error] $out is non-empty but incomplete; refusing to overwrite"
    echo fail > "${out}.failed"
    return 1
  fi
  rm -f "${out}.failed"
  echo "[merge-alpha] merge α=$a  A=$a  B=$wb  -> $out"
  if PYTHONPATH="$REPO/colpali:$REPO/code/shared${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON" "$MERGE" \
    --base "$BASE_MODEL" \
    --adapters "$ARM_A" "$ARM_B" \
    --weights "$a" "$wb" \
    --output "$out" \
    --dtype "$DTYPE"; then
    return 0
  fi
  echo "[merge-alpha][error] merge failed $out"
  echo fail > "${out}.failed"
  rm -rf "$out"
  return 1
}

wait_soup() {
  local out="$1"
  local deadline=$((SECONDS + WAIT_SECS))
  echo "[merge-alpha] waiting for $out"
  while (( SECONDS < deadline )); do
    [[ -f "${out}.failed" ]] && { echo "[merge-alpha][error] rank 0 reported merge failure $out"; return 1; }
    soup_ready "$out" && return 0
    sleep 10
  done
  echo "[merge-alpha][error] timed out $out"
  return 1
}

eval_one() {
  local name="$1"
  local port="$2"
  local d="$RUNS/$name"
  soup_ready "$d" || { echo "[merge-alpha][error] no soup, skip eval $name"; return 1; }
  if eval_done "$d"; then
    echo "[merge-alpha] $name eval already complete"
    return 0
  fi
  echo "[merge-alpha] eval $name  port=$port"
  RUN_NAME="$name" MASTER_PORT="$port" bash "$EVAL_SH"
}

rc=0
i=0
for a in $ALPHAS; do
  name="$(run_name_of "$a")"
  out="$RUNS/$name"
  port=$((${EVAL_PORT_BASE:-29510} + i))
  i=$((i + 1))
  if [[ "$NODE_RANK" == "0" ]]; then
    do_merge "$a" || { rc=1; continue; }
  else
    echo "[merge-alpha] waiting for $name"
    wait_soup "$out" || { rc=1; continue; }
  fi
  if [[ "$EVAL_AFTER" == "1" ]]; then
    eval_one "$name" "$port" || rc=1
  fi
done

echo "[merge-alpha] done rc=$rc ALPHAS=$ALPHAS"
exit $rc
