#![no_main]

use libfuzzer_sys::fuzz_target;
use r2smt_common::Arch;

fuzz_target!(|data: &[u8]| {
    let input = String::from_utf8_lossy(data);
    let _ = r2smt_pcode::parse_pcode(&input);
    let _ = r2smt_pcode::lift_pcode(&input, Arch::X86_64);
    let _ = r2smt_pcode::lift_pcode(&input, Arch::Aarch64);
});
