#!/usr/bin/env bash
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO="${REPO:-$(cd "$ROOT/../.." && pwd)}"
export EVAL_BASE_MODEL="${EVAL_BASE_MODEL:-tencent/EVIE-Preview-4.5B}"
exec bash "$REPO/code/shared/eval_run.sh"
