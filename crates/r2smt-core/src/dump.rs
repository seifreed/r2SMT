//! `dump-program` use case: ask the provider for the whole program and
//! return it unchanged.

use r2smt_common::Result;
use r2smt_ir::program::Program;
use r2smt_ir::provider::BinaryProvider;
use tracing::info;

/// Load the full program from `provider` and return it.
///
/// Thin wrapper around [`BinaryProvider::load_program`]; exists as a
/// stable use-case entrypoint so the CLI does not import the trait
/// directly.
///
/// # Errors
///
/// Propagates any error returned by the provider.
pub fn dump_program<P: BinaryProvider + ?Sized>(provider: &mut P) -> Result<Program> {
    let program = provider.load_program()?;
    info!(
        target: "r2smt::core",
        functions = program.functions.len(),
        "program dumped"
    );
    Ok(program)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use r2smt_common::{Address, Arch};
    use r2smt_ir::program::{BasicBlock, Function, Program};
    use r2smt_ir::testing::InMemoryProvider;

    use super::*;

    fn fixture() -> Program {
        Program {
            arch: Arch::X86_64,
            bits: 64,
            entry: Some(Address(0x40_1000)),
            functions: vec![Function {
                address: Address(0x40_1000),
                name: Some("sym.main".into()),
                blocks: vec![BasicBlock {
                    address: Address(0x40_1000),
                    instructions: vec![],
                    successors: vec![],
                }],
                is_thumb: false,
            }],
        }
    }

    #[test]
    fn dump_program_returns_provider_output() {
        let prog = fixture();
        let mut provider = InMemoryProvider::new(prog.clone());
        let dumped = dump_program(&mut provider).unwrap();
        assert_eq!(prog, dumped);
    }
}
