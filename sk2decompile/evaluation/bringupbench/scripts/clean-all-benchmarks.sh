#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load config; allow environment overrides
if [[ -f "${EVAL_ROOT}/config.env" ]]; then
  set -a
  source "${EVAL_ROOT}/config.env"
  set +a
fi

BENCH_REPO_ROOT="${BENCH_REPO_ROOT:?Set BENCH_REPO_ROOT in config.env or environment}"

cd "${BENCH_REPO_ROOT}"

echo "==> Running make all-clean"
make all-clean

echo "All benchmarks cleaned."
