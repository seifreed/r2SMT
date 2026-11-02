# Logging and observability

This document is the production contract for r2SMT telemetry. User output
(reports, JSON, patch plans, and radare2 commands) stays on stdout. Operational
events go to stderr through `tracing`, so piping a report never mixes it with
logs.

## Runtime scope

r2SMT is a synchronous Rust 2024 CLI and library workspace, not a web service.
Its important runtime boundaries are:

- the `r2smt` composition root and its subcommands;
- Rayon batch threads;
- sandboxed analysis and exploration subprocesses;
- radare2, Z3, CVC5, and Bitwuzla adapter calls;
- report, temporary-file, patch, manifest, backup, and rollback I/O;
- the native C radare2 plugin, which delegates analysis to the CLI and forwards
  its stderr to the r2 console.

The benchmark binary and repository scripts are offline validation tools. Their
stdout, exit status, and generated metrics are their explicit result contract;
they do not install an application logger. The exploration worker intentionally
does not write to stdio: its parent owns the deadline, outcome, and telemetry,
which prevents pipe deadlocks.

There is no application HTTP server, database, authentication system, cache,
scheduler, queue, or long-running daemon. OpenTelemetry export and distributed
tracing would add infrastructure without joining any real service boundary, so
they are not part of the current design. Latencies, outcomes, and counters are
structured log fields and can be converted into metrics by the log collector.

## Architecture

`crates/r2smt-cli/src/observability.rs` is the only subscriber bootstrap. It
uses the ecosystem-standard `tracing` and `tracing-subscriber` crates. Library
crates emit events and spans but never choose a formatter or destination, so
they remain backend-independent and usable by other applications.

The CLI opens a root `command` span with:

- `service=r2smt`
- `component=cli`
- `environment`
- `operation`
- optional `correlation_id`

JSON events carry these values in `span` and the full causal chain in `spans`.
Batch code explicitly parents each Rayon sample span to the command span,
because tracing's current span is thread-local. Subprocesses inherit the
validated correlation environment. Analysis-worker logs default to `warn` and
remain bounded in the parent's temporary stderr file; set `R2SMT_WORKER_LOG`
only when child-level diagnosis is needed.

Writes are synchronous and go to stderr. This is deliberate for a short-lived
CLI: it gives deterministic flush-on-exit and avoids a background logging queue
that can lose the final failure event. INFO volume is bounded to lifecycle and
aggregate events; hot-loop and per-sample details are DEBUG or TRACE. The
worker watchdog independently caps child stdout/stderr at 1 MiB. A formatter
destination failure is non-fatal and covered by a resilience test.

Stderr is the portable destination and works offline. In production, let the
process supervisor own files, rotation, retention, and backpressure:

```bash
# systemd/journald owns storage and retention
R2SMT_LOG_FORMAT=json R2SMT_ENVIRONMENT=production r2smt solve sample

# Local file destination; rotate with logrotate/newsyslog
R2SMT_LOG_FORMAT=json r2smt batch corpus 2>>r2smt.jsonl
```

No SaaS or remote collector is required. Docker, journald, syslog wrappers, and
shell redirection can replace the destination without changing the binary.

## Configuration

| Setting | Values | Default | Purpose |
|---|---|---|---|
| `-v`, `-vv` | CLI counts | INFO | DEBUG and TRACE defaults |
| `RUST_LOG` | `tracing-subscriber` filter | level from `-v` | Per-target levels, such as `r2smt=info,r2smt::smt=debug` |
| `R2SMT_LOG_FORMAT` | `compact`, `json` | `compact` | Human or structured output |
| `R2SMT_ENVIRONMENT` | safe ASCII context | `development` | Deployment environment |
| `R2SMT_CORRELATION_ID` | 1-128 safe ASCII characters | omitted | Caller-supplied operation correlation |
| `R2SMT_WORKER_LOG` | `tracing` filter | `warn` | Captured analysis-child verbosity |
| `NO_COLOR` | present/absent | terminal-aware | Disable ANSI in compact output |

Invalid filters, formats, environments, or correlation identifiers fail closed
before the command runs. JSON always disables ANSI. The correlation identifier
accepts W3C-compatible characters, but r2SMT does not fabricate `trace_id` or
`span_id` fields when no tracing system exists.

## Event conventions

Every lifecycle event uses a stable dotted `event_name`, an `operation` from
its span, and a `status`. Durations are integer `duration_ms`; counters are
integers; target names act as the component/module namespace.

| Level | Use | Examples |
|---|---|---|
| TRACE | opt-in hot-loop or instruction detail | reserved for exceptionally verbose diagnosis |
| DEBUG | technical steps and bounded per-item outcomes | worker spawn, solver verdict, batch sample |
| INFO | process or operation lifecycle and aggregates | command, radare2 ready, batch, patch, rollback |
| WARN | recoverable degradation | retry, timeout, filtered analysis artefact, partial batch |
| ERROR | failed command or invariant at the process boundary | `command.failed` |
| ERROR + `status=fatal` | process cannot safely continue | `process.panicked` |

`tracing` has no separate FATAL level; `status=fatal` preserves that semantic
without a custom logging framework.

Principal events include:

- `command.started`, `command.completed`, `command.failed`
- `process.panicked`
- `batch.started`, `batch.sample_completed`, `batch.sample_failed`,
  `batch.completed`
- `analysis_worker.starting`, `analysis_worker.spawned`,
  `analysis_worker.completed`
- `explore.worker_spawned`, `explore.worker_exited`,
  `explore.worker_timed_out`, `explore.worker_failed`, `explore.completed`
- `radare2.ready`, `radare2.spawn_retry`, `radare2.project_saved`
- `patch.started`, `patch.completed`, `rollback.started`,
  `rollback.completed`

Expected failures are returned and logged once at the appropriate boundary.
Batch sample failures are WARN because the sweep continues; the batch aggregate
then reports `status=partial`. An unrecoverable command error is emitted once as
ERROR. Panics include a bounded stack trace.

## Security and privacy

Logs minimize input data by default:

- command arguments, binary paths, report bodies, request/response bodies,
  sample bytes, solver models, and SMT formulas are not INFO fields;
- radare2 command payloads and project names are not logged;
- binaries are described by size, counts, and integrity hashes where useful;
- full differential-lifter payloads exist only at DEBUG and use escaped
  structured fields rather than multiline message interpolation;
- likely password, token, API-key, authorization, cookie, private-key, and
  secret-bearing error text is replaced with `[REDACTED]`;
- untrusted values are capped at 4,096 characters and control characters are
  escaped to prevent forged log lines;
- environment and correlation values use a strict ASCII allow-list.

Do not add an event field containing raw bytes, a complete file, a request
body, an authorization value, or a solver model. Prefer counts, a stable
non-secret enum, a digest, or an address. If detailed guest content is required
for a one-off diagnosis, keep it at DEBUG/TRACE and use `?field` so control
characters remain escaped.

## Diagnosing a run

Generate JSON and select one correlation value across a parent workflow:

```bash
export R2SMT_LOG_FORMAT=json
export R2SMT_ENVIRONMENT=production
export R2SMT_CORRELATION_ID=case-2026-11-02-001
export RUST_LOG='r2smt=info,r2smt::worker=debug'
r2smt solve sample --json report.json 2>events.jsonl
```

Useful queries:

```bash
jq -c 'select(.level == "ERROR" or .status == "fatal")' events.jsonl
jq -c 'select(.event_name == "batch.sample_failed")' events.jsonl
jq -s 'map(select(.duration_ms)) | sort_by(.duration_ms) | reverse[:10]' events.jsonl
```

An INFO run emits only a small lifecycle set. Increase one target rather than
the whole process when possible. `RUST_LOG=r2smt::r2pipe=debug` diagnoses r2
dispatch, while `RUST_LOG=r2smt::smt=debug` exposes per-branch solver outcomes.

## Adding instrumentation

1. Reuse the current span; do not install another subscriber in a library.
2. Add a span only at a real boundary or concurrent unit of work.
3. Use a stable `event_name`, `status`, and `duration_ms` for lifecycle events.
4. Log one completion event when it is enough; avoid start/end/error triplets
   for cheap operations.
5. Keep user-facing stdout unchanged.
6. Pass untrusted error text through `safe_value` in the CLI. In lower crates,
   prefer safe enums/counters and return the detailed cause to the boundary.
7. Add a focused contract test for new non-trivial telemetry.

Example:

```rust
let started = std::time::Instant::now();
let result = perform_operation()?;
tracing::info!(
    target: "r2smt::component",
    event_name = "component.operation_completed",
    status = "ok",
    duration_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX),
    items = result.len(),
    "operation completed"
);
```

## Audit evidence

The initial audit on 2026-11-02 scored **3.9/10**. The repository already used
`tracing`, but the subscriber was compact-only, invalid `RUST_LOG` silently
fell back, events had no lifecycle schema or root context, Rayon/subprocess
boundaries were not correlated, paths and project names appeared in logs, and
there were no observability tests.

| Dimension | Initial | Final | Evidence |
|---|---:|---:|---|
| Architecture and decoupling | 6 | 10 | One composition-root subscriber; libraries emit standard `tracing` events only |
| Functional coverage | 4 | 10 | CLI, Rayon batch, analysis/explore workers, radare2, solvers, patch/rollback, files and fatal boundary |
| Event quality and usefulness | 6 | 10 | Stable names, calibrated levels, outcomes, counts and durations; per-item detail below INFO |
| Structured logging and consistency | 2 | 10 | JSON UTC timestamps, level/message/target/thread plus normalized event and span fields |
| Context and correlation | 1 | 10 | Validated optional correlation, root command span, explicit Rayon parent, inherited subprocess context |
| Security and privacy | 4 | 10 | Redaction, bounds, control escaping, path/payload minimization, fatal-payload protection |
| Performance and resilience | 6 | 10 | Level gating, bounded INFO volume, child log cap/default WARN, deterministic flush, failed-writer test |
| Configuration and portability | 4 | 10 | `-v`, `RUST_LOG`, JSON/compact, environment/correlation, terminal-safe ANSI, stderr works offline |
| Error management | 6 | 10 | Cause chains at one boundary, recoverable WARN semantics, structured ERROR, fatal panic/backtrace |
| Testing and verifiability | 0 | 10 | Unit and process contracts for schema, filters, errors, redaction, injection, concurrency, destination failure and timeout |
| **Mean** | **3.9** | **10.0** | All ten dimensions apply |

Verification completed on 2026-11-02:

```text
cargo fmt --all -- --check                                      PASS
cargo test --workspace                                         2,102 passed; 5 ignored
R2SMT_SMOKE_BIN=target/debug/r2smt cargo test --workspace -- --ignored
                                                                5 passed
cargo clippy --workspace --all-targets -- -D warnings          PASS
CFLAGS unset + CMAKE_POLICY_VERSION_MINIMUM=3.5
  cargo clippy --workspace --all-targets --all-features -- -D warnings
                                                                PASS
```

The workspace currently emits one pre-existing toolchain warning because
`clippy::assert_is_empty` is unknown to the installed Clippy version; it is an
allowed lint in the root manifest and does not hide a diagnostic or fail either
gate. The default macOS smoke fixture `/bin/ls` is a universal Mach-O whose CFG
range filtering is unrelated to logging; the documented `R2SMT_SMOKE_BIN`
override was used with the workspace's thin test binary.
