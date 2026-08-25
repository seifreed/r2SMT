# r2smt-bench corpus

The corpus is source-first: checked-in C and per-ISA assembly make every case
inspectable and reproducible without trusting opaque binaries. `manifest.json`
declares the supported architecture, compiler, optimization, packaging, and
semantic coverage matrix.

Run the host E2E gate with:

```sh
./scripts/quality-gates.sh real-binaries
```

`r2smt-bench score` publishes every metric derivable from the stable report.
Metrics that require timing, frontend counters, or a patch transaction are
serialized as `null` until those values are present in the report rather than
being fabricated.

The host gate builds and analyzes the ELF cross-target matrix locally. The CI
`msvc-matrix` job builds the same source with Visual C++ for x86 and x86-64 and
publishes the COFF fixtures, so compiler coverage is not limited to the host
toolchain.
