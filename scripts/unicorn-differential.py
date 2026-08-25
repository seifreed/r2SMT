#!/usr/bin/env python3
"""Compare radare2 ESIL execution with Unicorn on generated x86-64 snippets.

The Rust ESIL lifter is contract-tested against radare2's emitted ESIL. This
gate adds an independent CPU oracle so arithmetic and flag regressions do not
only compare two r2SMT representations of the same operation.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import tempfile
from pathlib import Path

from keystone import KS_ARCH_X86, KS_MODE_64, Ks
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
from unicorn.x86_const import UC_X86_REG_EFLAGS, UC_X86_REG_RAX, UC_X86_REG_RBX


CASES = (
    "add rax, rbx",
    "xor rax, rbx",
    "and rax, rbx",
    "cmp rax, rbx",
    "test rax, rbx",
)
MASK64 = (1 << 64) - 1
HEX_LINE = re.compile(r"^0x([0-9a-fA-F]+)$")


def assemble(asm: str) -> bytes:
    encoding, _ = Ks(KS_ARCH_X86, KS_MODE_64).asm(asm)
    return bytes(encoding)


def run_unicorn(code: bytes, rax: int, rbx: int) -> tuple[int, int, int]:
    base = 0x1000
    emu = Uc(UC_ARCH_X86, UC_MODE_64)
    emu.mem_map(base, 0x1000)
    emu.mem_write(base, code)
    emu.reg_write(UC_X86_REG_RAX, rax)
    emu.reg_write(UC_X86_REG_RBX, rbx)
    emu.reg_write(UC_X86_REG_EFLAGS, 0)
    emu.emu_start(base, base + len(code))
    return (
        emu.reg_read(UC_X86_REG_RAX) & MASK64,
        emu.reg_read(UC_X86_REG_RBX) & MASK64,
        emu.reg_read(UC_X86_REG_EFLAGS) & 0x8C5,
    )


def radare_esil(r2: str, code: bytes, rax: int, rbx: int, path: Path) -> tuple[int, int, int]:
    path.write_bytes(code + b"\x90" * 16)
    probe = subprocess.run(
        [
            r2,
            "-2q",
            "-a",
            "x86",
            "-b",
            "64",
            "-e",
            "io.va=false",
            "-c",
            "aoj 1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    instruction = json.loads(probe.stdout)[0]
    esil = instruction.get("esil")
    if not esil:
        raise RuntimeError(f"radare2 emitted no ESIL for {instruction}")
    result = subprocess.run(
        [
            r2,
            "-2q",
            "-a",
            "x86",
            "-b",
            "64",
            "-e",
            "io.va=false",
            "-c",
            "aei",
            "-c",
            f"ar rax=0x{rax:x}",
            "-c",
            f"ar rbx=0x{rbx:x}",
            "-c",
            "ar eflags=0",
            "-c",
            f"ae {esil}",
            "-c",
            "ar rax",
            "-c",
            "ar rbx",
            "-c",
            "ar eflags",
            "-c",
            "q",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [int(match.group(1), 16) for line in result.stdout.splitlines() if (match := HEX_LINE.match(line))]
    if len(values) < 3:
        raise RuntimeError(f"could not parse radare2 register output: {result.stdout!r}")
    return values[-3] & MASK64, values[-2] & MASK64, values[-1] & 0x8C5


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--radare2", default="radare2")
    parser.add_argument("--iterations", type=int, default=8)
    args = parser.parse_args()
    rng = random.Random(0x5252534D54)
    checked = 0
    with tempfile.TemporaryDirectory(prefix="r2smt-unicorn-") as work:
        path = Path(work) / "instruction.bin"
        for asm in CASES:
            code = assemble(asm)
            for _ in range(args.iterations):
                rax = rng.getrandbits(64)
                rbx = rng.getrandbits(64)
                expected = run_unicorn(code, rax, rbx)
                actual = radare_esil(args.radare2, code, rax, rbx, path)
                if actual != expected:
                    raise SystemExit(
                        f"differential mismatch for {asm!r}: "
                        f"unicorn={expected!r} radare2-esil={actual!r}"
                    )
                checked += 1
    print(f"unicorn differential ok: {checked} generated executions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
