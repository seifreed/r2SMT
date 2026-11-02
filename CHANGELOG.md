# Changelog

## 0.6.2 - 2026-11-02

- Fix compiler-free `r2pm -bi r2smt` installation on Windows by isolating
  conditionals after r2pm flattens the manifest block into one command line.
- Validate binary installation and removal against published artifacts on all
  six supported operating-system and architecture combinations.

## 0.6.1 - 2026-11-02

- Add compiler-free `r2pm -bi r2smt` installation mapped to radare2 6.2.4
  on Linux, macOS, and Windows for x86-64 and ARM64.
- Publish self-contained r2pm bundles with the CLI, native core plugin, legacy
  macros, Windows Z3 runtime, checksums, SPDX SBOMs, and build provenance.
- Build and smoke-test native Windows plugins with MSVC, including reliable CLI
  execution from absolute Windows paths.
- Fix current-seek annotation rebasing for universal Mach-O binaries.
- Verify binary installation, CLI/plugin execution, and uninstallation from the
  published GitHub release on every supported platform.

## 0.6.0 - 2026-11-02

- Add centralized structured logging with compact development output and JSON
  production output, validated environment configuration, stable event names,
  and correlation context.
- Instrument CLI lifecycle, analysis, exploration workers, external processes,
  patching, rollback, timeouts, cancellation, and concurrent task boundaries.
- Redact secrets and control characters by default, bound untrusted fields, and
  keep diagnostic logs on stderr so functional CLI output remains stable.
- Capture uncaught panics with structured critical diagnostics and optional
  backtraces while preserving the original panic behavior.
- Add observability contract, concurrency, destination-failure, timeout, panic,
  and security tests, with production configuration guidance in the README.

## 0.5.0 - 2026-10-30

- Isolate Windows analysis workers with Job Objects and outbound firewall rules;
  fail closed if either control cannot be installed.
- Publish reproducible benchmark metrics and a pinned public corpus baseline,
  with an independent validation protocol and Unicorn differential checks.
- Expand CI coverage for Windows sandboxing, radare2 compatibility, benchmark
  fixtures, and dependency security.
- Update Cargo dependencies and GitHub Actions, including Z3 0.22 and rustls
  security fixes.

## 0.4.0 - 2026-08-25

- Add an auto-loaded native radare2 core plugin with direct file/seek/function
  state, parent-session annotations, ABI-local builds, and smoke coverage.
- Replace shell-dependent `$r2smt-*` macros whose seek/function substitutions
  and multi-command patch aliases were invalid on current radare2.
- Thanks to @trufae for preparing the native plugin integration.

## 0.3.0 - 2026-08-23

- Gate radare2 capabilities and verify 6.1.8, 6.2.0, and master in CI.
- Fix r2pm installation and parent-session annotations.
- Add architecture, soundness, compatibility, schema, and provenance contracts.
- Pin CI actions and produce checksums, SBOMs, and artifact provenance.
- Add a labeled multi-ISA corpus, real r2pipe fixtures, and E2E report/patch gates.
- Make patching transactional, hash-preconditioned, verified, and reversible.
- Expose the authoritative analysis pipeline through `r2smt-core::Analyzer`.
- Run authoritative analysis in bounded, network-denied worker sandboxes.
- Add three-way `--solver portfolio` consensus for conservative actions.
- Add cargo-fuzz targets, SSA properties, simplification equivalence,
  metamorphic tests, and radare2 ESIL differential execution.
- Add the optional process-boundary r2sleigh R2IL adapter.

This release completes the actionable engineering roadmap in `changes.md`.
Version 1.0 remains gated on sustained independent analyst usage and measured
field evidence, not additional repository scaffolding.
