"""
Headless IDA/Hex-Rays helper to dump pseudocode for every discovered function.
Usage (from shell):
    idat -A -S"scripts/dump_pseudo.py /path/to/output" /path/to/binary
"""

from __future__ import annotations

import os
import sys

import ida_auto
import ida_funcs
import ida_hexrays
import ida_pro
import idautils
import idc


def _get_output_path() -> str:
    # IDA populates idc.ARGV with the script path at index 0 and the
    # user-provided arguments afterwards.
    if len(idc.ARGV) < 2:
        raise RuntimeError("output path argument missing")
    return os.path.abspath(idc.ARGV[1])


def main() -> None:
    try:
        output_path = _get_output_path()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[dump_pseudo] {exc}", file=sys.stderr)
        ida_pro.qexit(1)
        return

    ida_auto.auto_wait()

    if not ida_hexrays.init_hexrays_plugin():
        print("[dump_pseudo] Hex-Rays decompiler is unavailable", file=sys.stderr)
        ida_pro.qexit(1)
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        for ea in idautils.Functions():
            name = ida_funcs.get_func_name(ea)
            handle.write(f"/* {name} @ 0x{ea:x} */\n")
            try:
                cfunc = ida_hexrays.decompile(ea)
            except ida_hexrays.DecompilationFailure as exc:
                handle.write(f"// decompilation failed: {exc}\n\n")
                continue

            handle.write(str(cfunc))
            handle.write("\n\n")

    ida_pro.qexit(0)


if __name__ == "__main__":
    main()
