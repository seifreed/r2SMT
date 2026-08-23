# Compatibility

Run `r2smt doctor` to inspect the exact runtime versions and the active ESIL
flags gate before comparing results from different machines.

## radare2

| Version | Status | ESIL flag-writing instructions |
|---|---|---|
| `< 6.1.0` | Unsupported | Disabled |
| `6.1.x` | Baseline analysis only | Automatically disabled |
| `>= 6.2.0` | Supported | Enabled by default; `--no-esil-flags` opts out |
| Development snapshots | Best effort | Determined from the reported semantic version |

radare2 6.2.0 fixed the ARM `subs` ESIL behavior on which the ARM flag rung
depends. r2SMT compares parsed numeric version components; it does not use
lexical string comparison. Reports record the exact radare2 version.

Supported analysis architectures are x86, x86-64, AArch32/Thumb, and AArch64.
Architecture support means the pipeline recognizes and models a documented
subset of instructions; unsupported instructions fail closed as described in
[SOUNDNESS.md](SOUNDNESS.md).

## Optional tools

| Tool | Purpose | Required |
|---|---|---|
| r2ghidra | P-code ingestion and decompiler context | Only for `--ir pcode|auto` or decompiler output |
| Z3 | Default in-process SMT backend | Built with r2SMT |
| CVC5 | Independent SMT-LIB backend | Only for `--solver cvc5` |
| Bitwuzla | Independent SMT-LIB backend | Only for `--solver bitwuzla` |

`doctor` reports missing optional tools as unavailable. Their absence does not
disable the default ESIL plus Z3 path.

## Build hosts

Release artifacts target Linux, macOS, and Windows on x86-64 and ARM64. Source
builds require Rust 1.95 or newer, CMake, and a C++ toolchain. Runtime behavior
still depends on a compatible radare2 installation on `PATH`.
