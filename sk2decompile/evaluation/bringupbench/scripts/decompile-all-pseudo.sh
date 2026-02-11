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

IDA_BIN="${IDA_BIN:-/home/bairidreamer/software/IDA-Pro/idat}"
DUMP_SCRIPT="${EVAL_ROOT}/scripts/dump_pseudo.py"

if [[ ! -x "${IDA_BIN}" ]]; then
  echo "error: IDA binary not found or not executable at ${IDA_BIN}" >&2
  exit 1
fi

if [[ ! -f "${DUMP_SCRIPT}" ]]; then
  echo "error: dump script not found at ${DUMP_SCRIPT}" >&2
  exit 1
fi

readarray -t BINARIES < <(
  find "${BENCH_REPO_ROOT}" -mindepth 2 -maxdepth 2 -type f \
    \( -iname '*.o0' -o -iname '*.o1' -o -iname '*.o2' -o -iname '*.o3' \) \
    ! -path "${BENCH_REPO_ROOT}/scripts/*" \
    ! -path "${BENCH_REPO_ROOT}/target/*" \
    ! -path "${BENCH_REPO_ROOT}/common/*" \
    ! -path "${BENCH_REPO_ROOT}/.git/*" \
    | sort
)

if [[ ${#BINARIES[@]} -eq 0 ]]; then
  echo "error: no O0/O1/O2/O3 binaries found under ${BENCH_REPO_ROOT}" >&2
  exit 1
fi

for binary_path in "${BINARIES[@]}"; do
  output_path="${binary_path}.pseudo"
  echo "==> Decompiling ${binary_path#${BENCH_REPO_ROOT}/} -> ${output_path#${BENCH_REPO_ROOT}/}"
  "${IDA_BIN}" -A "-S${DUMP_SCRIPT} ${output_path}" "${binary_path}"
done

echo "All pseudocode dumps are located alongside their binaries."
