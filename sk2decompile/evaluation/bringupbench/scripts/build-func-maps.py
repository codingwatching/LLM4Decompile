#!/usr/bin/env python3
"""Generate function-level mappings across source, pseudo, and assembly outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
import subprocess

FUNC_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "do", "case", "else"}

TYPEDEF_MAP = {
    "cpu_set_t": "int",
    "nl_item": "int",
    "__time_t": "int",
    "__mode_t": "unsigned short",
    "__off64_t": "long long",
    "__blksize_t": "long",
    "__ino_t": "unsigned long",
    "__blkcnt_t": "unsigned long long",
    "__syscall_slong_t": "long",
    "__ssize_t": "long int",
    "wchar_t": "unsigned short int",
    "wctype_t": "unsigned short int",
    "__int64": "long long",
    "__int32": "int",
    "__int16": "short",
    "__int8": "char",
    "_QWORD": "uint64_t",
    "_OWORD": "long double",
    "_DWORD": "uint32_t",
    "size_t": "unsigned int",
    "_BYTE": "uint8_t",
    "_TBYTE": "uint16_t",
    "_BOOL8": "uint8_t",
    "gcc_va_list": "va_list",
    "_WORD": "unsigned short",
    "_BOOL4": "int",
    "__va_list_tag": "va_list",
    "_IO_FILE": "FILE",
    "DIR": "int",
    "__fsword_t": "long",
    "__kernel_ulong_t": "int",
    "cc_t": "int",
    "speed_t": "int",
    "fd_set": "int",
    "__suseconds_t": "int",
    "_UNKNOWN": "void",
    "__sighandler_t": "void (*)(int)",
    "__compar_fn_t": "int (*)(const void *, const void *)",
}


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


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _strip_empty(code: str) -> str:
    return "\n".join(line for line in code.splitlines() if line.strip())


def _good_func(func: str) -> bool:
    body = "{".join(func.split("{", 1)[1:]) if "{" in func else func
    total = 0
    for line in body.splitlines():
        if len(line.strip()) >= 3:
            total += 1
    return 3 < total < 300


def _format_with_clang(func: str, style: str = "Google") -> Optional[str]:
    if not func:
        return None
    cmd = ["clang-format", f"--style={style}"]
    try:
        proc = subprocess.run(
            cmd,
            input=func,
            text=True,
            capture_output=True,
            check=True,
            timeout=15,
        )
        return proc.stdout
    except Exception as e:
        print(e)
        return None


def _hex_to_dec(text: str) -> str:
    pattern = re.compile(r"\b(0x[0-9a-fA-F]+)([uUlL]{1,3})?\b")

    def convert(match: re.Match[str]) -> str:
        hex_part = match.group(1)
        suffix = match.group(2) or ""
        return str(int(hex_part, 16)) + suffix

    return pattern.sub(convert, text)


def _remove_keywords(text: str) -> str:
    patterns = [
        r"\b__fastcall\b",
        r"\b__cdecl\b",
        r"\b__ptr32\b",
        r"\b__noreturn\s+noreturn\b",
    ]
    combined = re.compile("|".join(patterns))
    return combined.sub("", text)

def _replace_typedefs(text: str) -> str:
    for alias, original in TYPEDEF_MAP.items():
        pattern = re.compile(rf"\b{re.escape(alias)}\b")
        text = pattern.sub(original, text)
    return text


def _remove_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//.*?$", "", text, flags=re.MULTILINE)
    return text


def _process_code(code_str: str) -> str:
    code_str = _remove_comments(code_str)
    code_str = _hex_to_dec(code_str)
    code_str = _remove_keywords(code_str)
    code_str = _replace_typedefs(code_str)
    return code_str


def _normalize_pseudo(text: str) -> str:
    processed = _process_code(text)
    if not processed.strip():
        return ""
    formatted = _format_with_clang(processed)
    if formatted is None:
        return ""
    cleaned = _strip_empty(formatted)
    if not cleaned or not _good_func(cleaned):
        return ""
    return cleaned


def _strip_comments_and_strings(text: str) -> str:
    result = list(text)
    i = 0
    length = len(text)
    while i < length:
        nxt = text[i : i + 2]
        ch = text[i]
        if nxt == "//":
            end = text.find("\n", i)
            if end == -1:
                end = length
            for j in range(i, end):
                result[j] = " "
            i = end
            continue
        if nxt == "/*":
            end = text.find("*/", i + 2)
            if end == -1:
                end = length - 2
            for j in range(i, end + 2):
                result[j] = " "
            i = end + 2
            continue
        if ch in {'"', "'"}:
            quote = ch
            result[i] = " "
            i += 1
            while i < length:
                c = text[i]
                result[i] = " "
                if c == "\\":
                    i += 2
                    continue
                if c == quote:
                    i += 1
                    break
                i += 1
            continue
        i += 1
    return "".join(result)

def _find_matching_brace(text: str, start_idx: int) -> int:
    depth = 0
    i = start_idx
    length = len(text)
    while i < length:
        nxt = text[i : i + 2]
        ch = text[i]
        if nxt == "//":
            i = text.find("\n", i)
            if i == -1:
                return length - 1
            continue
        if nxt == "/*":
            i = text.find("*/", i + 2)
            if i == -1:
                return length - 1
            i += 2
            continue
        if ch in {'"', "'"}:
            quote = ch
            i += 1
            while i < length:
                c = text[i]
                if c == "\\":
                    i += 2
                    continue
                if c == quote:
                    i += 1
                    break
                i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return length - 1


def _extract_source_functions(path: Path, repo_root: Path) -> Dict[str, Dict[str, str]]:
    text = _read_text(path)
    sanitized = _strip_comments_and_strings(text)
    pattern = re.compile(
        r"(?P<prefix>^|[;\n}])(?P<signature>[^{;}]*?)\b(?P<name>[A-Za-z_][\w]*)\s*\([^;{}]*\)\s*\{",
        re.MULTILINE,
    )
    funcs: Dict[str, Dict[str, str]] = {}
    for match in pattern.finditer(sanitized):
        name = match.group("name")
        if name in FUNC_KEYWORDS:
            continue
        brace_idx = sanitized.find("{", match.start("signature"))
        if brace_idx == -1:
            continue
        end_idx = _find_matching_brace(text, brace_idx)
        if end_idx <= brace_idx:
            continue
        start_idx = match.start("signature")
        content = text[start_idx : end_idx + 1].strip("\n") + "\n"
        funcs.setdefault(
            name,
            {
                "path": str(path.relative_to(repo_root)),
                "function_name": name,
                "content": content,
            },
        )
    return funcs

def _parse_makefile(makefile: Path) -> List[Path]:
    text = _read_text(makefile)
    prog_match = re.search(r"^PROG\s*=\s*(\S+)", text, flags=re.MULTILINE)
    if not prog_match:
        raise RuntimeError(f"PROG not found in {makefile}")
    prog = prog_match.group(1).strip()
    objs_match = re.search(r"^LOCAL_OBJS\s*=\s*(.*)$", text, flags=re.MULTILINE)
    obj_tokens: List[str] = []
    if objs_match:
        obj_tokens = [token for token in objs_match.group(1).split() if token]
    if not obj_tokens:
        obj_tokens = [f"{prog}.o"]
    src_paths: List[Path] = []
    for token in obj_tokens:
        if not token.endswith(".o"):
            continue
        candidate = makefile.parent / token.replace(".o", ".c")
        if candidate.exists():
            src_paths.append(candidate)
    if not src_paths:
        fallback = makefile.parent / f"{prog}.c"
        if fallback.exists():
            src_paths.append(fallback)
    return src_paths


def _collect_source_functions(bench_dir: Path, repo_root: Path) -> Dict[str, Dict[str, str]]:
    makefile = bench_dir / "Makefile"
    srcs = _parse_makefile(makefile)
    func_map: Dict[str, Dict[str, str]] = {}
    for src in srcs:
        func_map.update(_extract_source_functions(src, repo_root))
    return func_map


def _parse_pseudo(pseudo_path: Path, repo_root: Path) -> Dict[str, Dict[str, str]]:
    text = _read_text(pseudo_path)
    lines = text.splitlines()
    pattern = re.compile(r"^/\*\s*(?P<name>[^@]+?)\s*@\s*(?P<addr>0x[0-9a-fA-F]+)\s*\*/$")
    current: Optional[str] = None
    current_addr: Optional[str] = None
    buffer: List[str] = []
    out: Dict[str, Dict[str, str]] = {}
    for raw_line in lines:
        line = raw_line.strip()
        match = pattern.match(line)
        if match:
            if current and buffer:
                content = "\n".join(buffer).strip("\n") + "\n"
                out.setdefault(
                    current,
                    {
                        "path": str(pseudo_path.relative_to(repo_root)),
                        "function_name": current,
                        "address": current_addr,
                        "label": current,
                        "content": content,
                    },
                )
            current = match.group("name").strip()
            current_addr = match.group("addr")
            buffer = []
        else:
            if current is not None:
                buffer.append(raw_line)
    if current and buffer:
        content = "\n".join(buffer).strip("\n") + "\n"
        out.setdefault(
            current,
            {
                "path": str(pseudo_path.relative_to(repo_root)),
                "function_name": current,
                "address": current_addr,
                "label": current,
                "content": content,
            },
        )
    return out

def _clean_instruction(raw: str) -> Optional[str]:
    stripped = raw.strip()
    if not stripped:
        return None
    parts = raw.split("\t")
    if len(parts) >= 3:
        relevant = parts[2:]
    elif len(parts) == 2:
        relevant = parts[1:]
    else:
        relevant = [stripped]
    instr = "\t".join(relevant)
    instr = instr.split("#")[0].strip()
    if not instr:
        return None
    if all(c in "0123456789abcdefABCDEF" for c in instr.replace(" ", "")):
        return None
    return instr


def _clean_asm_block(name: str, lines: List[str]) -> str:
    cleaned = [f"<{name}>:"]
    for raw in lines[1:]:
        instr = _clean_instruction(raw)
        if instr:
            cleaned.append(instr)
    return "\n".join(cleaned) + "\n"


def _parse_assembly(asm_path: Path) -> Dict[str, str]:
    lines = _read_text(asm_path).splitlines()
    header = re.compile(r"^\s*([0-9a-fA-F]+)\s+<([^>]+)>:\s*$")
    current: Optional[str] = None
    buffer: List[str] = []
    result: Dict[str, str] = {}
    for line in lines:
        match = header.match(line)
        if match:
            if current and buffer:
                result.setdefault(current, _clean_asm_block(current, buffer))
            current = match.group(2)
            buffer = [line]
        else:
            if current is not None:
                buffer.append(line)
    if current and buffer:
        result.setdefault(current, _clean_asm_block(current, buffer))
    return result


def _discover_binaries(explicit: Optional[List[str]], repo_root: Path) -> List[Path]:
    if explicit:
        binaries: List[Path] = []
        for entry in explicit:
            candidate = Path(entry)
            if not candidate.is_absolute():
                candidate = repo_root / candidate
            if candidate.exists():
                binaries.append(candidate)
        return binaries
    matches = []
    for path in repo_root.rglob("*.O*"):
        suffix = path.suffix.lower()
        if suffix in {".o0", ".o1", ".o2", ".o3"}:
            matches.append(path)
    return sorted(matches)

def _build_map(binary: Path, repo_root: Path) -> None:
    pseudo_path = Path(str(binary) + ".pseudo")
    asm_path = Path(str(binary) + ".s")
    if not pseudo_path.exists() or not asm_path.exists():
        print(f"[skip] Missing pseudo or assembly for {binary.relative_to(repo_root)}")
        return
    bench_dir = binary.parent
    source_funcs = _collect_source_functions(bench_dir, repo_root)
    pseudo_funcs = _parse_pseudo(pseudo_path, repo_root)
    asm_funcs = _parse_assembly(asm_path)
    common = sorted(set(source_funcs) & set(pseudo_funcs) & set(asm_funcs))
    if not common:
        print(f"[warn] No overlapping functions for {binary.relative_to(repo_root)}")
        return
    output_path = Path(str(binary) + ".func_map.jsonl")
    rel_binary = str(binary.relative_to(repo_root))
    with output_path.open("w", encoding="utf-8") as handle:
        for name in common:
            pseudo_entry = pseudo_funcs[name]
            pseudo_norm = _normalize_pseudo(pseudo_entry.get("content", ""))
            record = {
                "source": source_funcs[name],
                "pseudo": pseudo_entry,
                "pseudo_normalize": pseudo_norm,
                "binary": rel_binary,
                "assembly": asm_funcs[name],
            }
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    print(f"[ok] {output_path.relative_to(repo_root)} -> {len(common)} functions")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Map source/pseudo/assembly per function")
    parser.add_argument(
        "--binary",
        action="append",
        help="Specific binary path (relative to repo) to process; can be repeated.",
    )
    parser.add_argument(
        "--bench-root",
        default=None,
        help="Path to the Bringup-Bench repository root (default: from config.env).",
    )
    args = parser.parse_args(argv)
    repo_root = _get_bench_root(args.bench_root)
    binaries = _discover_binaries(args.binary, repo_root)
    if not binaries:
        print("No binaries found", file=sys.stderr)
        return 1
    for binary in binaries:
        _build_map(binary, repo_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
