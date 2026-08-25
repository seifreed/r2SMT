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

The optional `core_r2smt` integration is a native r2 core plugin. It is built
locally by `r2pm -ci r2smt` (or `make plugin`) against the active radare2 ABI,
then auto-loaded from `R2_USER_PLUGINS`. The plugin supports the same radare2
6.1+ range as the CLI and delegates authoritative analysis to the isolated
`r2smt` executable. Set `R2SMT_CLI` when that executable is outside `PATH`.

## Optional tools

| Tool | Purpose | Required |
|---|---|---|
| r2ghidra | P-code ingestion and decompiler context | Only for `--ir pcode|auto` or decompiler output |
| r2sleigh | Experimental R2IL adapter | Only for the `r2il` subcommand |
| Z3 | Default in-process SMT backend | Built with r2SMT |
| CVC5 | Independent SMT-LIB backend | `--solver cvc5` or `--solver portfolio` |
| Bitwuzla | Independent SMT-LIB backend | `--solver bitwuzla` or `--solver portfolio` |

`--solver portfolio` requires all three backends and emits a decisive verdict
only when Z3, CVC5, and Bitwuzla agree. A missing backend is an error and any
verdict disagreement fails closed to `Unsound`.

`doctor` reports missing optional tools as unavailable. Their absence does not
disable the default ESIL plus Z3 path.

The experimental `r2il` command consumes the external tool's
`run --action lift --format r2cmd` contract and never links r2sleigh. The
LGPL-3.0-only executable is installed and distributed separately from r2SMT's
MIT binaries. The adapter validates each R2IL JSON sidecar and fails closed if
the paired ESIL operation is outside r2SMT's strict subset. Its fixture tracks
r2sleigh master commit `60942f62cdd36717e08544bdfc1dafdd3fa514d9` and R2IL
format v4; future upstream contract changes must update that fixture first.
See [docs/R2SLEIGH.md](docs/R2SLEIGH.md) for the bounded adapter contract and
its current architecture ceiling.

## Analysis worker isolation

`solve`, `annotate`, `at`, `patch`, and every `batch` sample run in a
subprocess group with a 120-second wall-clock deadline, a 2 GiB aggregate RSS
limit, 1 MiB stdout/stderr caps, and explicit function/branch caps. The worker
uses an isolated temporary `HOME`, opens the sample read-only, and returns a
bounded JSON result to the parent.

On macOS the worker requires the system `sandbox-exec`; on Linux it requires
`bubblewrap`. Both deny networking and expose the host filesystem read-only,
with writes allowed only inside the worker's temporary directory. If that
sandbox is unavailable, authoritative analysis fails closed. Windows builds
currently retain inspection and doctor commands, but authoritative analysis is
disabled until an equivalent restricted-token/job-object sandbox is available.
`r2smt doctor` reports the active worker sandbox.

## Build hosts

Release artifacts target Linux, macOS, and Windows on x86-64 and ARM64. Source
builds require Rust 1.95 or newer, CMake, and a C++ toolchain. Runtime behavior
still depends on a compatible radare2 installation on `PATH`.
