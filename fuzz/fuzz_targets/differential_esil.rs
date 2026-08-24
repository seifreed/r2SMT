#![no_main]

use libfuzzer_sys::fuzz_target;
use r2smt_common::Arch;
use r2smt_difflift::build_equivalence_query;

fuzz_target!(|data: &[u8]| {
    let Some(separator) = data.iter().position(|byte| *byte == 0) else {
        return;
    };
    let left = String::from_utf8_lossy(&data[..separator]);
    let right = String::from_utf8_lossy(&data[separator + 1..]);
    let Ok(left) = r2smt_esil::lift_esil(&left, Arch::X86_64) else {
        return;
    };
    let Ok(right) = r2smt_esil::lift_esil(&right, Arch::X86_64) else {
        return;
    };
    let _ = build_equivalence_query(&left.statements, &right.statements, Arch::X86_64);
});
