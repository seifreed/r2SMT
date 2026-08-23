# radare2 JSON contracts

`r2-6.2.0/` contains responses captured from radare2 6.2.0 and r2ghidra
6.2.0 against a two-instruction AArch64 executable. Paths, addresses,
symbol names, and unrelated metadata were normalized; command structure and
the fields consumed by r2SMT were retained.

The frozen fixtures catch parser regressions without requiring radare2. The
`r2pipe-contracts` quality gate separately probes the same commands against
supported radare2 releases and `master` in CI.
