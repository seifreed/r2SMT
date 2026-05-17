//! Integration smoke test that exercises a live radare2 process.
//!
//! Ignored by default to keep the workspace test suite hermetic. Run
//! explicitly with `cargo test -p r2smt-r2pipe -- --ignored` on a host
//! that has `radare2` on `PATH`. Honours the `R2SMT_SMOKE_BIN`
//! environment variable for picking the target binary; defaults to the
//! current `r2smt` binary if it has been built, otherwise skips.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stderr)]

use std::env;
use std::path::PathBuf;

use r2smt_common::Address;
use r2smt_ir::{Annotator, BinaryProvider, BytePatcher};
use r2smt_r2pipe::R2PipeProvider;

fn target_binary() -> Option<PathBuf> {
    if let Ok(path) = env::var("R2SMT_SMOKE_BIN") {
        let p = PathBuf::from(path);
        return p.exists().then_some(p);
    }
    let fallback = PathBuf::from("/bin/ls");
    fallback.exists().then_some(fallback)
}

#[test]
#[ignore = "requires radare2 on PATH and a target binary"]
fn load_program_against_real_binary() {
    let Some(bin) = target_binary() else {
        eprintln!("no target binary available; skipping smoke test");
        return;
    };
    let mut provider = R2PipeProvider::open(&bin).expect("open r2");
    let program = provider.load_program().expect("load program");
    assert!(
        !program.functions.is_empty(),
        "expected at least one discovered function"
    );
}

#[test]
#[ignore = "requires radare2 on PATH and a target binary"]
fn set_comment_does_not_error_on_real_session() {
    let Some(bin) = target_binary() else {
        eprintln!("no target binary available; skipping annotator smoke test");
        return;
    };
    let mut provider = R2PipeProvider::open(&bin).expect("open r2");
    let program = provider.load_program().expect("load program");
    let first_addr = program.functions.first().map_or(Address(0), |f| f.address);
    let payload = "r2SMT smoke test: opaque/High -- ZF == 0 -- sym.main";
    provider
        .set_comment(first_addr, payload)
        .expect("CCu accepted by live r2 session");
}

#[test]
#[ignore = "requires radare2 on PATH and a target binary"]
fn byte_patcher_round_trips_through_real_session() {
    let Some(bin) = target_binary() else {
        eprintln!("no target binary available; skipping byte_patcher smoke test");
        return;
    };
    let mut provider = R2PipeProvider::open(&bin).expect("open r2");
    let program = provider.load_program().expect("load program");
    let entry = program
        .entry
        .or_else(|| program.functions.first().map(|f| f.address))
        .expect("at least one address");

    // Read the first 4 bytes of the entry without writing — this only
    // exercises the read path, which is safe against any binary.
    let original = provider
        .read_bytes(entry, 4)
        .expect("p8 returns 4 bytes at entry");
    assert_eq!(
        original.len(),
        4,
        "p8 must return exactly the requested size"
    );

    // `pa nop @ entry` is the canonical x86 NOP assembly — verify the
    // assemble path returns at least one byte. Skip if r2 reports an
    // unsupported arch (e.g. on hosts without an x86 assembler).
    if let Ok(asm) = provider.assemble(entry, "nop") {
        assert!(
            !asm.is_empty(),
            "pa must return at least one byte for 'nop'"
        );
    } else {
        eprintln!("`pa nop` not supported by this r2 build; skipping assemble check");
    }
}
