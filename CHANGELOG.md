# Changelog

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
