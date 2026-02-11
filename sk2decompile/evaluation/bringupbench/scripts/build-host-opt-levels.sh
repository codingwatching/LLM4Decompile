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

for opt in 0 1 2 3; do
  echo "==> Building host binaries with -O${opt}"
  make TARGET=host OPT_CFLAGS="-O${opt} -g" run-tests
  find . -maxdepth 2 -type f -name '*.host' -execdir mv {} {}.O${opt} \;
done

echo "All host optimization builds complete."
