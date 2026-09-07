#!/usr/bin/env bash
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${REPO:-$(cd "$ROOT/../.." && pwd)}"
export EVAL_BASE_MODEL="${EVAL_BASE_MODEL:-Qwen/Qwen3.5}"
exec bash "$REPO/code/shared/eval_run.sh"
