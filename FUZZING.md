# Fuzzing

The fuzz package covers the ESIL and P-code parser/lifter paths, report JSON,
and validated patch manifests. It is intentionally outside the main workspace
so normal stable builds do not compile libFuzzer.

```bash
cargo install cargo-fuzz --version 0.13.2 --locked
cargo +nightly fuzz run esil
cargo +nightly fuzz run pcode
cargo +nightly fuzz run report_json
cargo +nightly fuzz run patch_manifest
```

Crashes are written below `fuzz/artifacts/`; evolving inputs live below
`fuzz/corpus/`. Both directories are ignored. The scheduled fuzz workflow runs
bounded smoke campaigns for every target; longer campaigns can reuse the same
commands without the `-runs` cap.
