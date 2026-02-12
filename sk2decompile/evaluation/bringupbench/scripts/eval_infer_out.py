#!/usr/bin/env python3
"""
Evaluate infer-out-model2 functions by patching benchmark sources inside an
isolated workspace, rebuilding, executing, and collecting structured logs for
every case listed in a JSONL file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _load_config_env() -> dict:
    """Load config.env from the eval project root."""
    eval_root = Path(__file__).resolve().parents[1]
    config_path = eval_root / "config.env"
    config = {}
    if config_path.exists():
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                config[key.strip()] = value.strip()
    return config


def _get_bench_root(cli_value: str | None = None) -> Path:
    """Resolve the benchmark repo root from CLI arg, env var, or config.env."""
    if cli_value:
        return Path(cli_value).resolve()
    env_val = os.environ.get("BENCH_REPO_ROOT")
    if env_val:
        return Path(env_val).resolve()
    config = _load_config_env()
    if "BENCH_REPO_ROOT" in config:
        return Path(config["BENCH_REPO_ROOT"]).resolve()
    sys.exit("error: BENCH_REPO_ROOT not set. Use --bench-root, set the env var, or configure config.env")


@dataclass
class CaseResult:
    """Container for the outcome of processing a single case."""

    case_id: str
    source_path: str
    benchmark_dir: str
    output_dir: str
    workspace_dir: str = ""
    artifact_dir: str = ""
    replacement_applied: bool = False
    build_status: str = "skipped"  # succeeded | failed | skipped
    test_status: str = "skipped"
    notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    log_files: Dict[str, str] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace functions with infer-out-model2 bodies, build, "
        "execute, and record results without modifying the original benchmarks."
    )
    parser.add_argument(
        "jsonl",
        help="Path to the merged.*.jsonl file containing cases to evaluate.",
    )
    parser.add_argument(
        "--bench-root",
        default=None,
        help="Path to the Bringup-Bench repository root (default: from config.env).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on the number of cases to process.",
    )
    parser.add_argument(
        "--target",
        default="host",
        help="Benchmark build target passed as TARGET=<target> (default: host).",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/infer_out_eval",
        help="Directory (relative to eval root) where aggregated reports are written.",
    )
    parser.add_argument(
        "--workspace-root",
        default="reports/infer_out_eval/workspaces",
        help="Directory (relative to eval root) to host temporary build workspaces.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip running 'make clean' inside the workspace (useful when iterating).",
    )
    parser.add_argument(
        "--keep-workspaces",
        action="store_true",
        help="Keep temporary workspaces after each case finishes (default removes them).",
    )
    parser.add_argument(
        "--command-timeout",
        type=int,
        default=20,
        help="Timeout (in seconds) for each make invocation; 0 disables the timeout.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=96,
        help="Number of cases to process in parallel (default: 1).",
    )
    return parser.parse_args()


def canonicalize(text: str) -> str:
    """Normalize newlines for reliable substring matching."""
    return text.replace("\r\n", "\n")


def replace_function_body(
    full_source: str, reference_function: str, inferred_function: str
) -> Tuple[str, bool]:
    """
    Replace the exact reference_function text with inferred_function.

    Returns the updated source and a boolean indicating if replacement happened.
    """
    source_norm = canonicalize(full_source)
    reference_norm = canonicalize(reference_function)
    inferred_norm = canonicalize(inferred_function).rstrip() + "\n"

    candidates = (
        reference_norm,
        reference_norm.rstrip() + "\n",
        reference_norm.strip(),
    )

    for snippet in candidates:
        start_idx = source_norm.find(snippet)
        if start_idx == -1:
            continue
        end_idx = start_idx + len(snippet)
        updated = source_norm[:start_idx] + inferred_norm + source_norm[end_idx:]
        return updated, True
    return full_source, False


def compose_case_id(case: Dict) -> str:
    """Build a stable identifier for a case."""
    return (
        f"{case['source']['path']}::{case['source']['function_name']}"
        f"@{case['pseudo']['address']}"
    )


def ensure_case_output_dir(
    output_root: Path, pseudo_path_str: str, pseudo_address: str, result: CaseResult
) -> Path:
    """Create the per-case output directory, handling file path collisions."""
    pseudo_rel = Path(pseudo_path_str)
    base_dir = output_root / pseudo_rel

    if base_dir.exists() and base_dir.is_file():
        fallback = base_dir.parent / f"{base_dir.name}.infer_eval"
        fallback.mkdir(parents=True, exist_ok=True)
        result.notes.append(
            f"pseudo.path '{pseudo_path_str}' is a file; using '{fallback.relative_to(output_root)}' for logs."
        )
        base_dir = fallback
    else:
        base_dir.mkdir(parents=True, exist_ok=True)

    case_dir = base_dir / pseudo_address
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    return case_dir


def run_command(
    command: List[str],
    cwd: Path,
    log_handle,
    step_name: str,
    timeout: Optional[int],
) -> Optional[int]:
    """Run a command, capture stdout/stderr, and write everything to log_handle."""
    log_handle.write(f"\n[{step_name}] $ {' '.join(command)}\n")
    log_handle.flush()
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout if timeout and timeout > 0 else None,
        )
        log_handle.write(completed.stdout)
        log_handle.write(f"[{step_name}] exit code: {completed.returncode}\n")
        log_handle.flush()
        return completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.output or exc.stdout
        if output:
            if isinstance(output, bytes):
                log_handle.write(output.decode("utf-8", "replace"))
            else:
                log_handle.write(output)
        log_handle.write(
            f"[{step_name}] timed out after {timeout} seconds; terminating process.\n"
        )
        log_handle.flush()
        return None


def write_case_artifacts(
    case_dir: Path,
    case: Dict,
    modified_source: str,
    original_source: str,
) -> None:
    """Persist reusable artifacts for a case."""
    (case_dir / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    (case_dir / "modified_source.c").write_text(modified_source, encoding="utf-8")
    (case_dir / "original_source.c").write_text(original_source, encoding="utf-8")
    (case_dir / "original_function.c").write_text(
        canonicalize(case["source"]["content"]), encoding="utf-8"
    )
    (case_dir / "infer_function.c").write_text(
        canonicalize(case["pseudo"]["content-fix"]), encoding="utf-8"
    )


def sanitize_case_id(case_id: str) -> str:
    """Generate filesystem-safe case identifier."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", case_id)
    return sanitized.strip("_") or "case"


def copy_ignore_eval_dirs(_src: str, names: List[str]) -> List[str]:
    """Ignore helper to skip evaluation artifacts when copying benchmark dirs."""
    ignored: List[str] = []
    for name in names:
        if name.endswith(".infer_eval"):
            ignored.append(name)
    return ignored


def prepare_workspace(
    repo_root: Path,
    benchmark_dir: Path,
    workspace_root: Path,
    case_id: str,
) -> Tuple[Path, Path]:
    """Clone the necessary subset of the repo into a temporary workspace."""
    workspace_case_root = workspace_root / sanitize_case_id(case_id)
    if workspace_case_root.exists():
        shutil.rmtree(workspace_case_root)
    workspace_repo_root = workspace_case_root / "repo"
    workspace_repo_root.mkdir(parents=True, exist_ok=True)

    shutil.copy2(repo_root / "Makefile", workspace_repo_root / "Makefile")
    shutil.copytree(repo_root / "common", workspace_repo_root / "common", dirs_exist_ok=True)
    shutil.copytree(repo_root / "target", workspace_repo_root / "target", dirs_exist_ok=True)
    shutil.copytree(
        benchmark_dir,
        workspace_repo_root / benchmark_dir.name,
        dirs_exist_ok=True,
        ignore=copy_ignore_eval_dirs,
    )
    return workspace_case_root, workspace_repo_root


def relative_to_repo(path: Path, repo_root: Path) -> str:
    """Return a path relative to repo_root when possible."""
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def init_case_result(case: Dict, repo_root: Path) -> CaseResult:
    """Create a CaseResult with basic metadata for the given case."""
    source_rel = Path(case["source"]["path"])
    benchmark_dir_path = (repo_root / source_rel).parent
    try:
        benchmark_rel = str(benchmark_dir_path.relative_to(repo_root))
    except ValueError:
        benchmark_rel = str(benchmark_dir_path)
    return CaseResult(
        case_id=compose_case_id(case),
        source_path=str(source_rel),
        benchmark_dir=benchmark_rel,
        output_dir="",
    )


def snapshot_artifacts(
    case_dir: Path,
    workspace_benchmark_dir: Path,
    eval_root: Path,
    result: CaseResult,
) -> None:
    """Copy the workspace benchmark directory into the case directory."""
    artifacts_dir = case_dir / "artifacts"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    try:
        shutil.copytree(workspace_benchmark_dir, artifacts_dir)
        result.artifact_dir = relative_to_repo(artifacts_dir, eval_root)
    except Exception as exc:  # pragma: no cover - defensive
        result.notes.append(f"Failed to copy artifacts: {exc}")


def process_case(
    case: Dict,
    args: argparse.Namespace,
    repo_root: Path,
    eval_root: Path,
) -> CaseResult:
    """Process a single JSONL entry."""
    case_id = compose_case_id(case)
    source_rel = Path(case["source"]["path"])
    source_path = repo_root / source_rel
    benchmark_dir = source_path.parent

    result = init_case_result(case, repo_root)

    if not source_path.exists():
        result.errors.append(f"Source file '{source_rel}' does not exist.")
        return result

    try:
        case_dir = ensure_case_output_dir(
            eval_root, case["pseudo"]["path"], case["pseudo"]["address"], result
        )
    except Exception as exc:  # pragma: no cover - defensive
        result.errors.append(f"Failed to prepare case directory: {exc}")
        return result

    result.output_dir = str(case_dir.relative_to(eval_root))

    full_source_text = source_path.read_text(encoding="utf-8")
    updated_source, replaced = replace_function_body(
        full_source_text,
        case["source"]["content"],
        case["pseudo"]["content-fix"],
    )

    if not replaced:
        result.errors.append(
            "Could not locate the original function snippet in source file."
        )
        return result

    result.replacement_applied = True
    write_case_artifacts(case_dir, case, updated_source, full_source_text)

    workspace_root = Path(args.workspace_root)
    if not workspace_root.is_absolute():
        workspace_root = eval_root / workspace_root
    workspace_root.mkdir(parents=True, exist_ok=True)

    workspace_case_root: Optional[Path] = None
    try:
        workspace_case_root, workspace_repo_root = prepare_workspace(
            repo_root, benchmark_dir, workspace_root, case_id
        )
        workspace_benchmark_dir = workspace_repo_root / benchmark_dir.name
        artifacts_captured = False

        def capture_artifacts() -> None:
            nonlocal artifacts_captured
            if artifacts_captured:
                return
            snapshot_artifacts(case_dir, workspace_benchmark_dir, eval_root, result)
            artifacts_captured = True

        workspace_source_path = workspace_repo_root / source_rel
        workspace_source_path.write_text(updated_source, encoding="utf-8")

        result.workspace_dir = relative_to_repo(workspace_case_root, eval_root)

        log_path = case_dir / "case.log"
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"Case: {case_id}\n")
            log_handle.write(f"Workspace: {workspace_case_root}\n")
            log_handle.write(f"Benchmark copy: {workspace_benchmark_dir}\n")
            log_handle.write(f"Target: {args.target}\n")
            log_handle.flush()

            if not args.skip_clean:
                clean_rc = run_command(
                    ["make", f"TARGET={args.target}", "clean"],
                    workspace_benchmark_dir,
                    log_handle,
                    "clean",
                    args.command_timeout,
                )
                if clean_rc is None:
                    result.errors.append(
                        f"'make clean' timed out after {args.command_timeout} seconds."
                    )
                    capture_artifacts()
                    result.log_files["case"] = relative_to_repo(log_path, eval_root)
                    return result
                if clean_rc != 0:
                    result.build_status = "failed"
                    result.errors.append("make clean failed.")
                    capture_artifacts()
                    result.log_files["case"] = relative_to_repo(log_path, eval_root)
                    return result
            else:
                log_handle.write("Skipping 'make clean' per --skip-clean flag.\n")

            build_rc = run_command(
                ["make", f"TARGET={args.target}", "build"],
                workspace_benchmark_dir,
                log_handle,
                "build",
                args.command_timeout,
            )

            result.log_files["case"] = relative_to_repo(log_path, eval_root)
            if build_rc is None:
                result.build_status = "failed"
                result.errors.append(
                    f"'make build' timed out after {args.command_timeout} seconds."
                )
                capture_artifacts()
                log_handle.write("Skipping test because build timed out.\n")
                return result
            if build_rc == 0:
                result.build_status = "succeeded"
            else:
                result.build_status = "failed"
                result.errors.append("make build failed.")
                log_handle.write("Skipping test because build failed.\n")
                capture_artifacts()
                return result

            test_rc = run_command(
                ["make", f"TARGET={args.target}", "test"],
                workspace_benchmark_dir,
                log_handle,
                "test",
                args.command_timeout,
            )

            if test_rc is None:
                result.test_status = "failed"
                result.errors.append(
                    f"'make test' timed out after {args.command_timeout} seconds."
                )
            elif test_rc == 0:
                result.test_status = "succeeded"
            else:
                result.test_status = "failed"
                result.errors.append("make test failed.")

            capture_artifacts()

    finally:
        if (
            workspace_case_root
            and workspace_case_root.exists()
            and not args.keep_workspaces
        ):
            shutil.rmtree(workspace_case_root, ignore_errors=True)

    return result


def collect_cases(jsonl_path: Path, limit: Optional[int]) -> Iterable[Dict]:
    """Yield cases from jsonl file respecting the optional limit."""
    processed = 0
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)
            processed += 1
            if limit is not None and processed >= limit:
                break


def compute_summary(results: List[CaseResult]) -> Dict:
    """Aggregate statistics over all case results."""
    total = len(results)
    replacements = sum(1 for r in results if r.replacement_applied)
    build_success = sum(1 for r in results if r.build_status == "succeeded")
    test_success = sum(1 for r in results if r.test_status == "succeeded")

    def frac(passed: int, denom: int) -> float:
        return round(passed / denom, 4) if denom else 0.0

    per_benchmark: Dict[str, Dict[str, float]] = {}
    for r in results:
        stats = per_benchmark.setdefault(
            r.benchmark_dir,
            {
                "cases": 0,
                "replacements": 0,
                "build_success": 0,
                "test_success": 0,
            },
        )
        stats["cases"] += 1
        if r.replacement_applied:
            stats["replacements"] += 1
        if r.build_status == "succeeded":
            stats["build_success"] += 1
        if r.test_status == "succeeded":
            stats["test_success"] += 1

    for stats in per_benchmark.values():
        stats["replacement_rate"] = frac(stats["replacements"], stats["cases"])
        stats["build_rate"] = frac(stats["build_success"], stats["cases"])
        stats["test_rate"] = frac(stats["test_success"], stats["cases"])

    summary = {
        "total_cases": total,
        "replacement_success_count": replacements,
        "replacement_success_rate": frac(replacements, total),
        "compilable_count": build_success,
        "compilable_rate": frac(build_success, total),
        "executable_count": test_success,
        "executable_rate": frac(test_success, total),
        "compilation_failures": [
            r.case_id for r in results if r.build_status == "failed"
        ],
        "execution_failures": [
            r.case_id
            for r in results
            if r.build_status == "succeeded" and r.test_status == "failed"
        ],
        "cases": [asdict(r) for r in results],
        "by_benchmark": per_benchmark,
    }
    return summary


def write_summary(
    eval_root: Path,
    args: argparse.Namespace,
    jsonl_path: Path,
    summary: Dict,
) -> Tuple[Path, Path]:
    """Write JSON and Markdown summary reports."""
    report_root = eval_root / args.report_dir
    report_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_name = f"{jsonl_path.stem}-{args.target}"
    json_report = report_root / f"{base_name}-{timestamp}.json"
    markdown_report = report_root / f"{base_name}-{timestamp}.md"

    json_report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    benchmark_lines = [
        "| Benchmark | Cases | Replacement% | Build% | Exec% |",
        "| --- | --- | --- | --- | --- |",
    ]
    for bench, stats in sorted(summary["by_benchmark"].items()):
        benchmark_lines.append(
            f"| {bench} | {stats['cases']} | "
            f"{stats['replacement_rate']*100:.2f}% | "
            f"{stats['build_rate']*100:.2f}% | "
            f"{stats['test_rate']*100:.2f}% |"
        )
    if len(benchmark_lines) == 2:
        benchmark_lines.append("| (none) | 0 | 0.00% | 0.00% | 0.00% |")

    compilation_items = summary["compilation_failures"] or ["None"]
    execution_items = summary["execution_failures"] or ["None"]

    relative_jsonl = relative_to_repo(jsonl_path, eval_root)

    lines = [
        f"# Infer-Out Model 2 Evaluation ({base_name})",
        "",
        f"- Timestamp: {timestamp}",
        f"- Source JSONL: {relative_jsonl}",
        f"- Target: {args.target}",
        f"- Total cases: {summary['total_cases']}",
        f"- Replacement success: {summary['replacement_success_count']} "
        f"({summary['replacement_success_rate']*100:.2f}%)",
        f"- Compilable: {summary['compilable_count']} "
        f"({summary['compilable_rate']*100:.2f}%)",
        f"- Executable: {summary['executable_count']} "
        f"({summary['executable_rate']*100:.2f}%)",
        "",
        "## Benchmark Breakdown",
        *benchmark_lines,
        "",
        "## Compilation Failures",
    ]
    lines.extend(f"- {cid}" for cid in compilation_items)
    lines.append("")
    lines.append("## Execution Failures")
    lines.extend(f"- {cid}" for cid in execution_items)

    markdown_report.write_text("\n".join(lines), encoding="utf-8")
    return json_report, markdown_report


def main() -> int:
    args = parse_args()
    eval_root = Path(__file__).resolve().parents[1]
    repo_root = _get_bench_root(args.bench_root)
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_absolute():
        jsonl_path = eval_root / jsonl_path

    if not jsonl_path.exists():
        print(f"JSONL file '{jsonl_path}' not found.", file=sys.stderr)
        return 1

    cases = list(collect_cases(jsonl_path, args.limit))
    if not cases:
        print("No cases to process.")
        return 0

    results: List[Optional[CaseResult]] = [None] * len(cases)

    def record_result(idx: int, case_result: CaseResult) -> None:
        results[idx] = case_result
        status = (
            f"build={case_result.build_status}, test={case_result.test_status}"
            if case_result.replacement_applied
            else "replacement_failed"
        )
        print(f"[{idx + 1}] {case_result.case_id}: {status}")

    if args.jobs <= 1:
        for idx, case in enumerate(cases):
            case_result = process_case(case, args, repo_root, eval_root)
            record_result(idx, case_result)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            future_to_idx = {
                executor.submit(process_case, case, args, repo_root, eval_root): idx
                for idx, case in enumerate(cases)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    case_result = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    case_result = init_case_result(cases[idx], repo_root)
                    case_result.errors.append(f"Unhandled exception: {exc}")
                record_result(idx, case_result)

    final_results = [res for res in results if res is not None]

    summary = compute_summary(final_results)
    json_report, markdown_report = write_summary(eval_root, args, jsonl_path, summary)
    print(f"Wrote summary reports:\n - {json_report}\n - {markdown_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
